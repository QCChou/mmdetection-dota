import argparse
import logging
import os
import sys
from collections import defaultdict

import mmcv
import ray
import numpy as np
import torch
from mmcv.parallel import collate
from mmcv.runner import load_checkpoint
from tqdm import tqdm

from mmdet.core import multiclass_nms, eval_map
from mmdet.core.evaluation.bbox_overlaps import bbox_overlaps
from mmdet.datasets import DotaDataset
from mmdet.datasets.dota_devkit.dota_utils import wordname_16
from mmdet.models import build_detector
from mmdet.ops import nms, soft_nms
from mmdet.apis import show_result


logger = logging.getLogger('nms_tester')
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s'))
logger.addHandler(ch)

dump_file = 'dump_%s.pkl'
visl_file = 'vis_%s_nms.jpg'
only_container_crane = False
is_retinanet = False


@ray.remote
def process_nms(idx, settype, path):
    dota = DotaDataset(settype)
    imgid = dota.get_id(idx)
    img = dota.get_img(idx)
    return _process_nms(imgid, img, path)


def _process_nms(imgid, img, path):
    import mmcv
    results = mmcv.load(os.path.join(path, dump_file % imgid))

    # dota nms
    results = dota_nms(results)

    # visualize
    # show_result(img, results, dataset='dota', score_thr=0.01, out_file=os.path.join(path, visl_file % imgid))

    return results


@ray.remote
def inference_if_not_exist(cfg, workdir, checkpoint, settype, img_idx, img_id=None):
    return _inference_if_not_exist(cfg, workdir, checkpoint, settype, img_idx, img_id, use_ray=True)


def _inference_if_not_exist(cfg, workdir, checkpoint, settype, img_idx, img_id=None, use_ray=False):
    if not img_id:
        dota = DotaDataset(settype)
        img_id = dota.get_id(img_idx)

    if os.path.exists(os.path.join(workdir, dump_file % img_id)):
        return True

    if not use_ray:
        return _inference_gpu(cfg, workdir, checkpoint, settype, img_idx)
    else:
        return ray.get(inference_gpu.remote(cfg, workdir, checkpoint, settype, img_idx))


@ray.remote(num_gpus=1, max_calls=1)
def inference_gpu(cfg, workdir, checkpoint, settype, img_idx):
    gpu_ids = ray.get_gpu_ids()
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
    return _inference_gpu(cfg, workdir, checkpoint, settype, img_idx)


def _inference_gpu(cfg, workdir, checkpoint, settype, img_idx):
    cfg = mmcv.Config.fromfile(cfg)
    dota = DotaDataset(settype, img_norm_cfg=cfg.img_norm_cfg, img_scale=cfg.img_scale, test_mode=True)
    img_id = dota.get_id(img_idx)

    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    model = model.cuda()
    load_checkpoint(model, checkpoint)

    model.eval()

    data = dota[img_idx]
    data = collate([data], samples_per_gpu=1)
    data['img_meta'] = [x.data[0] for x in data['img_meta']]

    with torch.no_grad():
        result = model(return_loss=False, rescale=True, lazy=True, **data)

    mmcv.dump(result, os.path.join(workdir, dump_file % img_id))

    # log result images
    from mmdet.apis import show_result
    img = dota.get_img(img_idx)
    show_result(img, result, dataset='dota', score_thr=0.01, out_file=os.path.join(workdir, 'vis_%s.jpg' % img_id))

    return True


def cidx(class_name):
    return wordname_16.index(class_name)


def dota_nms(results, ensemble=False):
    if not ensemble:
        # nms multiclass
        nms_iou_ths = [0.5] * 16
        nms_iou_ths[cidx('plane')] = 0.3
        nms_iou_ths[cidx('harbor')] = 0.8
        nms_iou_ths[cidx('bridge')] = 0.4
        nms_iou_ths[cidx('small-vehicle')] = 0.6
        # nms_iou_ths[cidx('ship')] = 0.8
        # nms_iou_ths[cidx('storage-tank')] = 0.8
        # nms_iou_ths[cidx('roundabout')] = 0.7

        if only_container_crane:
            nms_cnf_ths = [1.] * 16
            nms_cnf_ths[cidx('container-crane')] = 0.05     # TODO 0.95
        elif is_retinanet:
            nms_iou_ths = [0.3] * 16
            nms_cnf_ths = [0.3] * 16
            nms_cnf_ths[cidx('baseball-diamond')] = 0.6  # try2 0.5
            nms_cnf_ths[cidx('ground-track-field')] = 0.4
            nms_cnf_ths[cidx('bridge')] = 0.4
            nms_cnf_ths[cidx('soccer-ball-field')] = 0.5
            nms_cnf_ths[cidx('roundabout')] = 0.4
            nms_cnf_ths[cidx('container-crane')] = 0.1
        else:
            nms_cnf_ths = [0.05] * 16   # 주로 recall은 높고 precision이 아주 많이 떨어지는 class에 대해 대응
            nms_cnf_ths[cidx('bridge')] = 0.7
            nms_cnf_ths[cidx('ground-track-field')] = 0.2
            nms_cnf_ths[cidx('small-vehicle')] = 0.2
            nms_cnf_ths[cidx('roundabout')] = 0.2
            nms_cnf_ths[cidx('container-crane')] = 0.15
    else:
        nms_iou_ths = [0.5] * 16
        nms_iou_ths[cidx('plane')] = 0.3
        nms_iou_ths[cidx('harbor')] = 0.8
        nms_iou_ths[cidx('bridge')] = 0.4
        nms_iou_ths[cidx('small-vehicle')] = 0.6

        nms_cnf_ths = [0.05] * 16  # 주로 recall은 높고 precision이 아주 많이 떨어지는 class에 대해 대응
        nms_cnf_ths[cidx('bridge')] = 0.7
        nms_cnf_ths[cidx('ground-track-field')] = 0.2
        nms_cnf_ths[cidx('small-vehicle')] = 0.2
        nms_cnf_ths[cidx('roundabout')] = 0.2
        nms_cnf_ths[cidx('container-crane')] = 0.15

    results_new = []
    for iou_th, cnf_th, result_cls in zip(nms_iou_ths, nms_cnf_ths, results):
        inds = result_cls[:, -1] > cnf_th
        result_cls = result_cls[inds]

        _, inds = nms(result_cls, iou_th, device_id=None)   # TODO : soft_nms : poor performance?
        result_cls = result_cls[inds]

        results_new.append(result_cls)
    results = results_new

    # iof remove for each classes (inclusive relationship for each classes)
    cnf_certain = 1.0
    iof_ths = [0.0] * 16
    iof_ths[cidx('plane')] = 0.85
    iof_ths[cidx('baseball-diamond')] = 0.85
    iof_ths[cidx('bridge')] = 0.65
    iof_ths[cidx('ground-track-field')] = 0.85
    iof_ths[cidx('small-vehicle')] = 0.85
    iof_ths[cidx('large-vehicle')] = 0.9
    iof_ths[cidx('tennis-court')] = 0.7
    iof_ths[cidx('basketball-court')] = 0.7
    iof_ths[cidx('storage-tank')] = 0.85
    iof_ths[cidx('roundabout')] = 0.85
    iof_ths[cidx('swimming-pool')] = 0.85

    results_new = []
    for iof_th, result_cls in zip(iof_ths, results):
        if iof_th <= 0.:
            results_new.append(result_cls)
            continue

        bboxes, scores = result_cls[:, :4], result_cls[:, -1]
        iofs_all = bbox_overlaps(bboxes, bboxes, mode='iof')

        result_cls_new = []
        for box_idx in range(len(result_cls)):
            if scores[box_idx] > cnf_certain:
                result_cls_new.append(result_cls[box_idx])
                continue

            is_inside = False
            for box_idx2 in range(len(result_cls)):
                if box_idx == box_idx2:
                    continue
                # TODO : score check?
                if iofs_all[box_idx][box_idx2] > iof_th and iof_th > iofs_all[box_idx2][box_idx]:
                    is_inside = True
                    break
            if not is_inside:
                result_cls_new.append(result_cls[box_idx])
        result_cls_new = np.array(result_cls_new).reshape((-1, 5))
        results_new.append(result_cls_new)
    results = results_new

    # iof remove for related(confusing) classes (inclusive relationship between different classes)
    inter_iof_cnf = 0.9
    idx_related = [
        (cidx('small-vehicle'), cidx('large-vehicle'), 0.5),
    ]

    for idx1, idx2, th in idx_related:
        result_cls1 = results[idx1]
        result_cls2 = results[idx2]
        bboxes1, scores1 = result_cls1[:, :4], result_cls1[:, -1]
        bboxes2, scores2 = result_cls2[:, :4], result_cls2[:, -1]
        if len(bboxes1) == 0 or len(bboxes2) == 0:
            continue
        iofs_all = bbox_overlaps(bboxes1, bboxes2, mode='iof')

        bboxes1_new = []
        for result_cls, box, score, iofs in zip(result_cls1, bboxes1, scores1, iofs_all):
            if iofs.max() > th:
                continue
            bboxes1_new.append(result_cls)
        bboxes1_new = np.array(bboxes1_new).reshape((-1, 5))
        results[idx1] = bboxes1_new

    # inter-class NMS
    inter_nms_th = 1.
    if inter_nms_th < 1.:
        merged_results, merged_labels = [], []
        for lb, result in enumerate(results):
            merged_results.append(result)
            merged_labels.append(np.array([lb] * len(result)))
        merged_results = np.concatenate(merged_results, axis=0)
        merged_labels = np.concatenate(merged_labels, axis=0)

        _, inds = nms(merged_results.astype(np.float32), inter_nms_th)
        merged_results = merged_results[inds]
        merged_labels = merged_labels[inds]
        results = [merged_results[np.where(merged_labels == lb)] for lb in range(len(wordname_16))]

    # duplication test : precision이 절반으로 내려감, recall은 그대로.
    # results = [np.concatenate((x, x), axis=0) for x in results]

    return results


if __name__ == '__main__':
    """
    ray nodes:
    ray start --node-ip-address gpu-cloud-vnode140.dakao.io --redis-port 8550 --head --block
    ray start --redis-address 0.0.0.0:8550 --block
    
    run:
    python tools/test_nms.py --ray 1 --settype valid --path /data/private/mmdetection-dota/work_dirs/cascade_rcnn_dconv_c3-c5_r50_fpn_1x_oversample1_multigsd.25.60_try3/test300_valid_gsd.25.40_lowth2/
    python tools/test_nms.py --ray 1 --settype cv_valid
    """
    parser = argparse.ArgumentParser(description='MMDet NMS Test')
    parser.add_argument('--ray', default=0, type=int)
    parser.add_argument('--exp', default='retinanet_cc')

    exps = {
        'retinanet_cc': {
            'cfg': 'configs/dota/retinanet_x101_64x4d_fpn_1x_cc.py',
            'workdir': '/data/private/mmdetection-dota/work_dirs/retinanet_cc/full',
            'checkpoint': 'epoch_300.pth'
        },
        'frcnn_cc': {
            'cfg': 'configs/dota/faster_rcnn_dconv_c3-c5_x101_32x4d_fpn_1x_cc.py',
            'workdir': '/data/private/mmdetection-dota/work_dirs/frcnn_cc/full',
            'checkpoint': 'epoch_100.pth'   # TODO : overfitting @ 300epoch?
        },
        'crcnn_cc': {
            'cfg': 'configs/dota/cascade_rcnn_dconv_c3-c5_r50_fpn_1x_cc.py',
            'workdir': '/data/private/mmdetection-dota/work_dirs/crcnn_cc/full/',
            'checkpoint': 'epoch_100.pth'   # TODO : overfitting @ 300epoch?
        },
        'crcnn_cc2': {
            'cfg': 'configs/dota/cascade_rcnn_dconv_c3-c5_r50_fpn_1x_cc.py',
            'workdir': '/data/private/mmdetection-dota/work_dirs/crcnn_cc2_ov/full/',
            'checkpoint': 'epoch_100.pth'  # TODO : overfitting @ 300epoch?
        },
        'crcnn_cc2_e360': {
            'cfg': 'configs/dota/cascade_rcnn_dconv_c3-c5_r50_fpn_1x_cc.py',
            'workdir': '/data/private/mmdetection-dota/work_dirs/crcnn_cc2_ov/full360/',
            'checkpoint': 'epoch_360.pth'
        },

        # full trained models
        'retinanet_0412': {
            'cfg': 'configs/dota/retinanet_x101_64x4d_fpn_1x.py',
            'workdir': '/data/private/mmdetection-dota-v190410/mmdetection-dota/work_dirs/retinanet_v190411/full',
            'checkpoint': 'epoch_160.pth'   # TODO
        },
        'frcnn_0412_full': {
            'cfg': 'configs/dota/faster_rcnn_dconv_c3-c5_x101_32x4d_fpn_1x.py',
            'workdir': '/data/private/mmdetection-dota/work_dirs/frcnn_190411/full',
            'checkpoint': 'epoch_160.pth'
        },
        'frcnn_0412_full300': {
            'cfg': 'configs/dota/faster_rcnn_dconv_c3-c5_x101_32x4d_fpn_1x.py',
            'workdir': '/data/private/mmdetection-dota/work_dirs/frcnn_190411/full300',
            'checkpoint': 'epoch_300.pth'
        },
        'frcnn_x101w_full': {   # without deformable conv.
            'cfg': 'configs/dota/faster_rcnn_x101_64x4d_fpn_1x.py',
            'workdir': '/data/private/mmdetection-dota/work_dirs/frcnn_r101/full_gsdv2v3avg/',
            'checkpoint': 'epoch_200.pth'
        },

        'crcnn_0412_full': {
            'cfg': 'configs/dota/cascade_rcnn_dconv_c3-c5_r50_fpn_1x.py',
            'workdir': '/data/private/mmdetection-dota/work_dirs/crcnn_190410/full/',
            'checkpoint': 'epoch_400.pth'
        },

        'frcnn_0412_full300_gsd': {
            'cfg': 'configs/dota/faster_rcnn_dconv_c3-c5_x101_32x4d_fpn_1x.py',
            'workdir': '/data/private/mmdetection-dota/work_dirs/frcnn_190411/full300gsdv2v3avg/',
            'checkpoint': 'epoch_300.pth'
        },
        'crcnn_0412_full_gsd': {
            'cfg': 'configs/dota/cascade_rcnn_dconv_c3-c5_r50_fpn_1x.py',
            'workdir': '/data/private/mmdetection-dota/work_dirs/crcnn_190410/full_gsdv2v3avg/',
            'checkpoint': 'epoch_400.pth'
        },
    }
    for cv in range(1, 1+5):
        exps['crcnn_0412_cv%d' % cv] = {
            'cfg': 'configs/dota/cascade_rcnn_dconv_c3-c5_r50_fpn_1x.py',
            'workdir': '/data/private/mmdetection-dota/work_dirs/crcnn_190410/cv%d/' % cv,
            'checkpoint': 'epoch_400.pth'
        }
        exps['crcnn_0412_cv%d_gsd' % cv] = {
            'cfg': 'configs/dota/cascade_rcnn_dconv_c3-c5_r50_fpn_1x.py',
            'workdir': '/data/private/mmdetection-dota/work_dirs/crcnn_190410/cv%d_gsdv2v3avg/' % cv,
            'checkpoint': 'epoch_400.pth'
        }
        exps['crcnn_r50d_aug_cv%d_gsd' % cv] = {
            'cfg': 'configs/dota/cascade_rcnn_dconv_c3-c5_r50_fpn_1x.py',
            'workdir': '/data/private/mmdetection-dota/work_dirs/0414_augment/crcnn-r50-dconv/cv%d/' % cv,
            'checkpoint': 'epoch_240.pth'
        }
        exps['frcnn_x101d_aug_cv%d_gsd' % cv] = {    # work_dirs/0414_augment/frcnn-x101-dconv/cv1
            'cfg': 'configs/dota/faster_rcnn_dconv_c3-c5_x101_32x4d_fpn_1x.py',
            'workdir': '/data/private/mmdetection-dota/work_dirs/0414_augment/frcnn-x101-dconv/cv%d/' % cv,
            'checkpoint': 'epoch_200.pth'
        }

    parser.add_argument('--settype', default='valid')
    parser.add_argument('--exist-only', action='store_true')
    parser.add_argument('--inference-only', action='store_true')
    # parser.add_argument('--path', default='/data/private/mmdetection-dota/work_dirs/cascade_rcnn_dconv_c3-c5_r50_fpn_1x_oversample1_multigsd.25.60_try3/test300_cv_gsd.25.40_lowth2/')
    args = parser.parse_args()

    exp = exps[args.exp]
    assert os.path.exists(os.path.join(exp['cfg'])), os.path.join(exp['cfg'])
    assert os.path.exists(os.path.join(exp['workdir'], exp['checkpoint'])), os.path.join(exp['workdir'], exp['checkpoint'])
    checkpoint = os.path.join(exp['workdir'], exp['checkpoint'])
    exp['workdir'] = os.path.join(exp['workdir'], args.settype)
    if not os.path.exists(exp['workdir']):
        os.mkdir(exp['workdir'])
    only_container_crane = '_cc' in args.exp
    is_retinanet = 'retinanet' in args.exp
    logger.info(exp['workdir'])

    if args.ray:
        ray.init(redis_address='gpu-cloud-vnode140:8550', log_to_driver=False)

    dota = DotaDataset(args.settype)
    if not args.exist_only:
        logger.info('inference all images...')
        if args.ray:
            reqs = []
            for img_idx in tqdm(range(len(dota.index))):
                img_id = dota.get_id(img_idx)
                if os.path.exists(os.path.join(exp['workdir'], dump_file % img_id)):
                    continue
                reqs.append(inference_if_not_exist.remote(exp['cfg'], exp['workdir'], checkpoint, args.settype, img_idx, img_id))
            if len(reqs) > 0:
                tqdm_gen = tqdm(zip(reqs, range(len(dota.index))), total=len(reqs))
                for req, img_idx in tqdm_gen:
                    img_id = dota.get_id(img_idx)
                    tqdm_gen.set_postfix({'idx': img_idx, 'img_id': img_id})
                    ray.get(req)
        else:
            all_results = []
            tqdm_gen = tqdm(range(len(dota.index)))
            for img_idx in tqdm_gen:
                img_id = dota.get_id(img_idx)
                tqdm_gen.set_postfix({'idx': img_idx, 'img_id': img_id})
                _inference_if_not_exist(exp['cfg'], exp['workdir'], checkpoint, args.settype, img_idx, img_id=img_id, use_ray=False)

    if args.inference_only:
        logger.info('inference done.')
        sys.exit(0)

    logger.info('processing nms...')

    if args.ray:
        logger.info('mode=ray')

        reqs, all_results, test_ids = [], [], []
        for idx in tqdm(range(len(dota.index))):
            img_id = dota.get_id(idx)
            if not os.path.exists(os.path.join(exp['workdir'], dump_file % img_id)):
                logger.warning(os.path.exists(os.path.join(exp['workdir'], dump_file % img_id)))
                continue
            test_ids.append(idx)
            reqs.append(process_nms.remote(idx, args.settype, exp['workdir']))
        for req in tqdm(reqs):
            all_results.append(ray.get(req))
    else:
        all_results, test_ids = [], []
        for idx in tqdm(range(len(dota.index))):
            img_id = dota.get_id(idx)
            if not os.path.exists(os.path.join(exp['workdir'], dump_file % img_id)):
                continue
            test_ids.append(idx)
            results = _process_nms(img_id, dota.get_img(idx), exp['workdir'])

            all_results.append(results)

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
                submission_by_class[class_idx].append('%s %.3f %0.2f %0.2f %0.2f %0.2f' % (
                    imgid, row[4], row[0], row[1], row[2], row[3]
                ))

    for class_idx in submission_by_class:
        class_name = dota.CLASSES[class_idx]
        path = os.path.join(exp['workdir'], 'Task2_%s.txt' % class_name)
        with open(path, 'w') as f:
            f.write('\n'.join(submission_by_class[class_idx]))

    print(len(all_results), 'results are saved.', exp['workdir'])

    if args.settype != 'test':
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
