import argparse
import logging
import os
from collections import defaultdict

import ray
import mmcv
import numpy as np
from tqdm import tqdm

from mmdet.core import eval_map
from mmdet.datasets import DotaDataset
from tools.test_nms import dota_nms


logger = logging.getLogger('nms_ensembler')
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s'))
logger.addHandler(ch)


def _get_nms(pathlist, idx, exist_only, workdir):
    dota = DotaDataset(settype)
    img_id = dota.get_id(idx)
    valid_cnt = 0
    for dirpath, _ in pathlist:
        dumppath = os.path.join(dirpath, settype, 'dump_%s.pkl' % img_id)
        if os.path.exists(dumppath):
            valid_cnt += 1
        else:
            logger.warning('not found, %s in %s, %s' % (img_id, dirpath, dumppath))
    if not exist_only and valid_cnt < len(pathlist):
        raise Exception('no result found, img_id=', img_id, 'cnt=', valid_cnt)
    if valid_cnt == 0:
        raise Exception('no result found, img_id=', img_id, 'cnt=', valid_cnt)

    merged_result = None
    for dirpath, weight in pathlist:
        if not os.path.exists(os.path.join(dirpath, settype, 'dump_%s.pkl' % img_id)):
            continue
        result = mmcv.load(os.path.join(dirpath, settype, 'dump_%s.pkl' % img_id))

        # TOOD
        if 'retina' in dirpath:
            for class_idx in range(len(result)):
                inds = result[class_idx][:, -1] > 0.7
                result[class_idx] = result[class_idx][inds]

        for class_idx in range(len(result)):
            result[class_idx][:, -1] = result[class_idx][:, -1] * weight
        if merged_result is None:
            merged_result = result
        else:
            for class_idx in range(len(merged_result)):
                merged_result[class_idx] = np.concatenate((merged_result[class_idx], result[class_idx]), axis=0)
                assert merged_result[class_idx].shape[1] == 5

    final_result = dota_nms(merged_result, ensemble=True)

    # log result images
    from mmdet.apis import show_result
    img = dota.get_img(idx)
    show_result(img, final_result, dataset='dota', score_thr=0.01, out_file=os.path.join(workdir, 'visnms_%s.jpg' % img_id))

    return final_result


@ray.remote(num_cpus=4)
def get_nms(pathlist, idx, exist_only, workdir):
    return _get_nms(pathlist, idx, exist_only, workdir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MMDet Ensemble Test')
    parser.add_argument('--out', default='./ensemble/')
    parser.add_argument('--settype', default='valid')
    parser.add_argument('--ensemble', default='crcnn-cv5')
    parser.add_argument('--exist-only', action='store_true')
    parser.add_argument('--ray', action='store_true')
    args = parser.parse_args()

    settype = args.settype
    if args.ensemble == 'crcnn3':
        pathlist = [  # crcnn-ensemble3
            ('/data/private/mmdetection-dota/work_dirs/crcnn_190410/full/', 0.8 / 0.9),
            ('/data/private/mmdetection-dota/work_dirs/crcnn_190410/cv5/', 0.9 / 0.9),
            ('/data/private/mmdetection-dota/work_dirs/crcnn_190410/cv1/', 0.7 / 0.9),
        ]
    elif args.ensemble == 'crcnn-cv5':
        pathlist = [    # crcnn-ensemble-cv5
            ('/data/private/mmdetection-dota/work_dirs/crcnn_190410/cv1/', 0.7 / 0.8),
            ('/data/private/mmdetection-dota/work_dirs/crcnn_190410/cv2/', 0.7 / 0.8),
            ('/data/private/mmdetection-dota/work_dirs/crcnn_190410/cv3/', 0.6 / 0.8),
            ('/data/private/mmdetection-dota/work_dirs/crcnn_190410/cv4/', 0.7 / 0.8),
            ('/data/private/mmdetection-dota/work_dirs/crcnn_190410/cv5/', 0.8 / 0.8),
        ]
    elif args.ensemble == 'crcnn-fcv5':
        pathlist = [    # crcnn-ensemble-fcv5
            ('/data/private/mmdetection-dota/work_dirs/crcnn_190410/full/', 0.7 / 0.8),
            ('/data/private/mmdetection-dota/work_dirs/crcnn_190410/cv1/', 0.7 / 0.8),
            ('/data/private/mmdetection-dota/work_dirs/crcnn_190410/cv2/', 0.7 / 0.8),
            ('/data/private/mmdetection-dota/work_dirs/crcnn_190410/cv3/', 0.6 / 0.8),
            ('/data/private/mmdetection-dota/work_dirs/crcnn_190410/cv4/', 0.7 / 0.8),
            ('/data/private/mmdetection-dota/work_dirs/crcnn_190410/cv5/', 0.8 / 0.8),
        ]
    elif args.ensemble == 'crcnn-fcv5+frcnn':
        pathlist = [  # crcnn-ensemble-fcv5
            ('/data/private/mmdetection-dota/work_dirs/frcnn_190411/full300/', 0.7 / 0.8),
            ('/data/private/mmdetection-dota/work_dirs/crcnn_190410/full/', 0.7 / 0.8),
            ('/data/private/mmdetection-dota/work_dirs/crcnn_190410/cv1/', 0.7 / 0.8),
            ('/data/private/mmdetection-dota/work_dirs/crcnn_190410/cv2/', 0.7 / 0.8),
            ('/data/private/mmdetection-dota/work_dirs/crcnn_190410/cv3/', 0.6 / 0.8),
            ('/data/private/mmdetection-dota/work_dirs/crcnn_190410/cv4/', 0.7 / 0.8),
            ('/data/private/mmdetection-dota/work_dirs/crcnn_190410/cv5/', 0.8 / 0.8),  # /data/private/mmdetection-dota/work_dirs/frcnn_r101/full_gsdv2v3avg/valid
        ]
    elif args.ensemble == 'crcnn-fcv5+frcnn2':
        pathlist = [  # crcnn-ensemble-fcv5
            ('/data/private/mmdetection-dota/work_dirs/frcnn_r101/full_gsdv2v3avg/', 0.8 / 0.8),
            ('/data/private/mmdetection-dota/work_dirs/frcnn_190411/full300/', 0.8 / 0.8),
            ('/data/private/mmdetection-dota/work_dirs/crcnn_190410/full/', 0.7 / 0.8),
            ('/data/private/mmdetection-dota/work_dirs/crcnn_190410/cv1/', 0.7 / 0.8),
            ('/data/private/mmdetection-dota/work_dirs/crcnn_190410/cv2/', 0.7 / 0.8),
            ('/data/private/mmdetection-dota/work_dirs/crcnn_190410/cv3/', 0.6 / 0.8),
            ('/data/private/mmdetection-dota/work_dirs/crcnn_190410/cv4/', 0.7 / 0.8),
            ('/data/private/mmdetection-dota/work_dirs/crcnn_190410/cv5/', 0.8 / 0.8),
        ]
    elif args.ensemble == 'crcnn-fcv5+frcnn+retina':
        pathlist = [  # crcnn-ensemble-fcv5
            ('/data/private/mmdetection-dota/work_dirs/retinanet_v190411/full/', 0.6 / 0.8),
            ('/data/private/mmdetection-dota/work_dirs/frcnn_190411/full300/', 0.7 / 0.8),
            ('/data/private/mmdetection-dota/work_dirs/crcnn_190410/full/', 0.7 / 0.8),
            ('/data/private/mmdetection-dota/work_dirs/crcnn_190410/cv1/', 0.7 / 0.8),
            ('/data/private/mmdetection-dota/work_dirs/crcnn_190410/cv2/', 0.7 / 0.8),
            ('/data/private/mmdetection-dota/work_dirs/crcnn_190410/cv3/', 0.6 / 0.8),
            ('/data/private/mmdetection-dota/work_dirs/crcnn_190410/cv4/', 0.7 / 0.8),
            ('/data/private/mmdetection-dota/work_dirs/crcnn_190410/cv5/', 0.8 / 0.8),
        ]
    elif args.ensemble == 'crcnn5+frcnn2/gsd_avgv2v3':
        pathlist = [  # crcnn-ensemble3
            ('/data/private/mmdetection-dota/work_dirs/frcnn_r101/full_gsdv2v3avg/', 0.8 / 0.9),
            ('/data/private/mmdetection-dota/work_dirs/frcnn_190411/full300gsdv2v3avg/', 0.9 / 0.9),
            ('/data/private/mmdetection-dota/work_dirs/crcnn_190410/full_gsdv2v3avg/', 0.8 / 0.9),
            ('/data/private/mmdetection-dota/work_dirs/crcnn_190410/cv1_gsdv2v3avg/', 0.7 / 0.9),
            ('/data/private/mmdetection-dota/work_dirs/crcnn_190410/cv2_gsdv2v3avg/', 0.7 / 0.8),
            ('/data/private/mmdetection-dota/work_dirs/crcnn_190410/cv3_gsdv2v3avg/', 0.6 / 0.8),
            ('/data/private/mmdetection-dota/work_dirs/crcnn_190410/cv4_gsdv2v3avg/', 0.7 / 0.8),
            ('/data/private/mmdetection-dota/work_dirs/crcnn_190410/cv5_gsdv2v3avg/', 0.9 / 0.9),
            # at this moement of running this ensemble, above experiments are not done yet, so below models are added
            # for the baseline.
            ('/data/private/mmdetection-dota/work_dirs/frcnn_190411/full300/', 0.7 / 0.8),
            ('/data/private/mmdetection-dota/work_dirs/crcnn_190410/full/', 0.7 / 0.8),
            ('/data/private/mmdetection-dota/work_dirs/crcnn_190410/cv1/', 0.7 / 0.8),
            ('/data/private/mmdetection-dota/work_dirs/crcnn_190410/cv2/', 0.7 / 0.8),
            ('/data/private/mmdetection-dota/work_dirs/crcnn_190410/cv3/', 0.6 / 0.8),
            ('/data/private/mmdetection-dota/work_dirs/crcnn_190410/cv4/', 0.7 / 0.8),
            ('/data/private/mmdetection-dota/work_dirs/crcnn_190410/cv5/', 0.8 / 0.8),
        ]
    elif args.ensemble == 'crcnn3+frcnn2/gsd_avgv2v3/mix':
        pathlist = [  # crcnn-ensemble3
            ('/data/private/mmdetection-dota/work_dirs/frcnn_r101/full_gsdv2v3avg/', 0.8 / 0.9),
            ('/data/private/mmdetection-dota/work_dirs/frcnn_190411/full300gsdv2v3avg/', 0.9 / 0.9),
            ('/data/private/mmdetection-dota/work_dirs/crcnn_190410/full_gsdv2v3avg/', 0.8 / 0.9),
            ('/data/private/mmdetection-dota/work_dirs/crcnn_190410/cv5_gsdv2v3avg/', 0.9 / 0.9),
            ('/data/private/mmdetection-dota/work_dirs/crcnn_190410/cv1_gsdv2v3avg/', 0.7 / 0.9),
            # at this moement of running this ensemble, above experiments are not done yet, so below models are added
            # for the baseline.
            ('/data/private/mmdetection-dota/work_dirs/frcnn_190411/full300/', 0.7 / 0.8),
            ('/data/private/mmdetection-dota/work_dirs/crcnn_190410/full/', 0.7 / 0.8),
            ('/data/private/mmdetection-dota/work_dirs/crcnn_190410/cv1/', 0.7 / 0.8),
            ('/data/private/mmdetection-dota/work_dirs/crcnn_190410/cv2/', 0.7 / 0.8),
            ('/data/private/mmdetection-dota/work_dirs/crcnn_190410/cv3/', 0.6 / 0.8),
            ('/data/private/mmdetection-dota/work_dirs/crcnn_190410/cv4/', 0.7 / 0.8),
            ('/data/private/mmdetection-dota/work_dirs/crcnn_190410/cv5/', 0.8 / 0.8),
        ]
    elif args.ensemble == 'early_mix':
        pathlist = [
            ('/data/private/mmdetection-dota/work_dirs/frcnn_190411/full300gsdv2v3avg/', 0.9 / 0.9),
            ('/data/private/mmdetection-dota/work_dirs/crcnn_190410/full_gsdv2v3avg/', 0.8 / 0.9),
            ('/data/private/mmdetection-dota/work_dirs/crcnn_190410/cv5_gsdv2v3avg/', 0.9 / 0.9),
            ('/data/private/mmdetection-dota/work_dirs/crcnn_190410/cv1_gsdv2v3avg/', 0.7 / 0.9),
            # at this moement of running this ensemble, above experiments are not done yet, so below models are added
            # for the baseline.
            ('/data/private/mmdetection-dota/work_dirs/frcnn_190411/full300/', 0.7 / 0.8),
            ('/data/private/mmdetection-dota/work_dirs/crcnn_190410/full/', 0.7 / 0.8),
            ('/data/private/mmdetection-dota/work_dirs/crcnn_190410/cv1/', 0.7 / 0.8),
            ('/data/private/mmdetection-dota/work_dirs/crcnn_190410/cv2/', 0.7 / 0.8),
            ('/data/private/mmdetection-dota/work_dirs/crcnn_190410/cv3/', 0.6 / 0.8),
            ('/data/private/mmdetection-dota/work_dirs/crcnn_190410/cv4/', 0.7 / 0.8),
            ('/data/private/mmdetection-dota/work_dirs/crcnn_190410/cv5/', 0.8 / 0.8),
        ]
    else:
        raise ValueError(args.ensemble)

    dota = DotaDataset(settype)
    if args.ray:
        ray.init(redis_address='gpu-cloud-vnode140:8550', log_to_driver=False)

    all_results, test_ids = [], []
    logger.info('getting nms results')
    if not args.ray:
        for idx in tqdm(range(len(dota.index))):
            test_ids.append(idx)
            final_result = _get_nms(pathlist, idx, args.exist_only, args.out)
            all_results.append(final_result)
    else:
        reqs = []
        for idx in tqdm(range(len(dota.index))):
            req = get_nms.remote(pathlist, idx, args.exist_only, args.out)
            reqs.append(req)
        for idx in tqdm(range(len(dota.index))):
            test_ids.append(idx)
            all_results.append(ray.get(reqs[idx]))

    logger.info('getting gt infos...')
    gt_bboxes = []
    gt_labels = []
    for i in test_ids:
        bboxes, labels = dota.get_ann_info(i)

        gt_bboxes.append(bboxes)
        gt_labels.append(labels)

    # save for dota submission
    submission_by_class = defaultdict(list)
    for idx, result in tqdm(zip(test_ids, all_results), desc='save submission file'):
        imgid = dota.get_id(idx)
        if result is None:
            logging.warning('no result idx=%d id=%s' % (idx, imgid))
            continue
        for class_idx in range(len(result)):
            for row in result[class_idx]:
                submission_by_class[class_idx].append('%s %.4f %0.3f %0.3f %0.3f %0.3f' % (
                    imgid, row[4], row[0], row[1], row[2], row[3]
                ))

    for class_idx in submission_by_class:
        class_name = dota.CLASSES[class_idx]
        path = os.path.join(args.out, 'Task2_%s.txt' % class_name)
        with open(path, 'w') as f:
            f.write('\n'.join(submission_by_class[class_idx]))

    print(len(all_results), 'results are saved.', args.out)

    if settype != 'test':
        logger.info('evaluating...')
        mean_ap, eval_results = eval_map(
            all_results,
            gt_bboxes,
            gt_labels,
            scale_ranges=None,
            iou_thr=0.5,
            dataset=dota.CLASSES,
            print_summary=True)
    logger.info('done')
