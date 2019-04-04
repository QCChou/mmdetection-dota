import logging
import os
import random

import mmcv
import numpy as np
import torch
import slidingwindow as sw
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from torch.utils.data import Dataset

from mmcv.parallel import DataContainer as DC

from mmdet.core.evaluation.bbox_overlaps import bbox_overlaps
from mmdet.datasets.extra_aug import ExtraAugmentation
from mmdet.datasets.transforms import ImageTransform, BboxTransform, Numpy2Tensor
from mmdet.datasets.utils import to_tensor, random_scale
from mmdet.datasets.dota_devkit.dota_utils import wordname_16, dots4ToRec4
from mmdet.datasets.dota_devkit.DOTA import DOTA


logger = logging.getLogger('DotaDataset')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


_dota_root = '/data/public/rw/team-autolearn/aerial/DOTA/v1.5_hbb_190402/'
_size = 800


class DotaPreprocess:
    def __init__(self, target_gsd=None):
        self.target_gsd = target_gsd


class DotaDataset(Dataset, DotaPreprocess):
    def __init__(self, settype='cv_train', target_gsd=None, extra_aug=None,
                 img_norm_cfg=None, img_scale=(1333, 800), resize_keep_ratio=True, flip_ratio=0.,
                 size_divisor=32, test_mode=False, **kwargs):
        super(DotaDataset, self).__init__(target_gsd)
        print(kwargs.keys())

        self.settype = settype
        self.test_mode = test_mode
        if settype in ['train', 'cv_train', 'cv_valid']:
            root = os.path.join(_dota_root, 'train')
        elif settype == 'valid':
            root = os.path.join(_dota_root, 'valid')
        else:
            raise ValueError('invalid settype=%s' % settype)

        self.dota = DOTA(root)

        x = self.dota.imglist
        if 'cv' in settype:
            # stratified split
            cv = 0  # TODO
            y = []
            cnt_obj = 0
            for id in x:
                anns = self.dota.ImgToAnns[id]
                lbs = list(set([ann['name'] for ann in anns]))
                y.append(one_hot(lbs))
                cnt_obj += len(anns)

            mskf = MultilabelStratifiedKFold(n_splits=5, random_state=0)
            train_index = valid_index = None
            for fold, (train_index, valid_index) in enumerate(mskf.split(x, y)):
                if fold == cv:
                    break
            if 'train' in settype:
                self.index = train_index
            else:
                self.index = valid_index
        else:
            self.index = list(range(len(x)))
        logger.info('settype=%s loaded, size=%d' % (self.settype, len(self.index)))

        # filter out datas without gsd info.
        self.index = list(filter(lambda i: self.dota.gsd(self.dota.imglist[i]) > 0, self.index))
        logger.info('settype=%s filtered, size=%d' % (self.settype, len(self.index)))
        assert len(self.index) > 0

        # transforms
        self.multiscale_mode = 'range'
        self.img_scales = img_scale if isinstance(img_scale, list) else [img_scale]
        if not img_norm_cfg:
            img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
        self.img_transform = ImageTransform(size_divisor=size_divisor, **img_norm_cfg)
        self.bbox_transform = BboxTransform()
        self.numpy2tensor = Numpy2Tensor()

        # gsd normalization
        self.gsd_aug = GSDNormalizedCrop(crop_size=(_size, _size) if not self.test_mode else None)  # TODO : parameters

        # if use extra augmentation
        if extra_aug is not None:
            self.extra_aug = ExtraAugmentation(**extra_aug)
        else:
            self.extra_aug = None

        # image rescale if keep ratio
        self.resize_keep_ratio = resize_keep_ratio

        # flip if provided
        self.flip_ratio = flip_ratio

        # flags
        self.flag = np.zeros(len(self), dtype=np.uint8)

    def get_img(self, index):
        imgid = self.dota.imglist[self.index[index]]
        img = self.dota.loadImgs(imgid)[0]
        return img

    def get_ann_info(self, index):
        imgid = self.dota.imglist[self.index[index]]
        anns = self.dota.loadAnns(imgId=imgid)

        gt_bboxes = np.array([dots4ToRec4(ann['poly']) for ann in anns], dtype=np.int)
        gt_labels = np.array([wordname_16.index(ann['name']) + 1 for ann in anns], dtype=np.int)    # bg : 0
        return gt_bboxes, gt_labels

    def __getitem__(self, index):
        imgid = self.dota.imglist[self.index[index]]

        # load image
        img = self.get_img(index)
        ori_shape = (img.shape[0], img.shape[1], 3)

        # load annotations
        gsd = self.dota.gsd(imgid)
        gt_bboxes, gt_labels = self.get_ann_info(index)

        # skip the image if there is no valid gt bbox
        # if len(gt_bboxes) == 0:
        #     return None

        # gsd normalization & random crop
        if self.gsd_aug:
            img, gt_bboxes, gt_labels, gsd_scale = self.gsd_aug(img, gt_bboxes, gt_labels, gsd)
        else:
            gsd_scale = 1.

        if not self.test_mode:
            # extra augmentation
            if self.extra_aug is not None:
                img, gt_bboxes, gt_labels = self.extra_aug(img, gt_bboxes, gt_labels)

            # apply transforms
            flip = True if np.random.rand() < self.flip_ratio else False

            # randomly sample a scale
            img_scale = random_scale(self.img_scales, self.multiscale_mode)
            img, img_shape, pad_shape, scale_factor = self.img_transform(img, img_scale, flip, keep_ratio=self.resize_keep_ratio)
            img = img.copy()

            gt_bboxes = self.bbox_transform(gt_bboxes, img_shape, scale_factor, flip)

            img_meta = dict(
                imgid=imgid,
                ori_shape=ori_shape,
                img_shape=img_shape,
                pad_shape=pad_shape,
                scale_factor=scale_factor,
                gsd_scale_factor=gsd_scale,
                flip=flip)

            data = dict(
                img=DC(to_tensor(img), stack=True),
                img_meta=DC(img_meta, cpu_only=True),
                gt_bboxes=DC(to_tensor(gt_bboxes)),
                gt_labels=DC(to_tensor(gt_labels))
            )

            return data
        else:
            imgs = []
            img_metas = []
            # for img_scale in self.img_scales:
            # sliding window
            windows = sw.generate(img, sw.DimOrder.HeightWidthChannel, _size, 0.5)
            for window in windows:
                _img_w = img[window.indices()]

                for flip_h in [False]:
                    _img, _img_shape, _pad_shape, _scale_factor = self.img_transform(_img_w, self.img_scales[0], flip_h, keep_ratio=self.resize_keep_ratio)
                    _gt_bboxes = self.bbox_transform(gt_bboxes, _img_shape, _scale_factor, flip_h)

                    imgs.append(to_tensor(_img))
                    img_metas.append(DC(dict(
                        ori_shape=ori_shape,
                        img_shape=(_img.shape[1], _img.shape[2], 3),
                        pad_shape=(_img.shape[1], _img.shape[2], 3),
                        translation=(window.x, window.y),
                        scale_factor=_scale_factor,
                        gsd_scale_factor=gsd_scale,
                        flip=flip_h
                    ), cpu_only=True))

            data = dict(img=imgs, img_meta=img_metas)
            return data

    def __len__(self):
        return len(self.index)


class DotaTestDataset(Dataset):
    def __init__(self, img_norm_cfg=None, img_scale=(1333, 800), flip_ratio=0.):
        pass

    def __getitem__(self, index):
        pass

    def __len__(self):
        pass


class GSDNormalizedCrop(object):

    def __init__(self, crop_size=(800, 800), min_iou=0.3, target_gsd=0.25):
        self.crop_size = crop_size
        self.min_iou = [min_iou] * 9 + [0]
        self.target_gsd = target_gsd

    def __call__(self, img, boxes, labels, gsd):
        # resize (gsd normalization)
        scale = self.target_gsd / gsd
        img = mmcv.imrescale(img, scale)
        boxes = (boxes.astype(np.float64) * scale + 0.5).astype(np.int64)

        if not self.crop_size:
            return img, boxes, labels, scale

        # random crop
        h, w, c = img.shape
        new_w, new_h = self.crop_size
        not_cropped = True
        while not_cropped:
            min_iou = random.choice(self.min_iou)

            for i in range(50):
                # h / w in [0.5, 2]
                if new_h / new_w < 0.5 or new_h / new_w > 2:
                    continue

                left = random.uniform(0, w - new_w)
                top = random.uniform(0, h - new_h)

                patch = np.array((int(left), int(top), int(left + new_w), int(top + new_h)))
                overlaps = bbox_overlaps(patch.reshape(-1, 4), boxes.reshape(-1, 4)).reshape(-1)
                if overlaps.min() < min_iou:
                    continue

                # center of boxes should inside the crop img
                center = (boxes[:, :2] + boxes[:, 2:]) / 2
                mask = (center[:, 0] > patch[0]) * (center[:, 1] > patch[1]) * (center[:, 0] < patch[2]) * (center[:, 1] < patch[3])
                if not mask.any():
                    continue
                boxes = boxes[mask]
                labels = labels[mask]

                # adjust boxes
                img = img[patch[1]:patch[3], patch[0]:patch[2]]
                boxes[:, 2:] = boxes[:, 2:].clip(max=patch[2:])
                boxes[:, :2] = boxes[:, :2].clip(min=patch[:2])
                boxes -= np.tile(patch[:2], 2)
                not_cropped = False
                break

        return img, boxes, labels, scale


def one_hot(labels):
    lb = [wordname_16.index(x) for x in labels]
    lb = np.eye(num_class(), dtype=np.float)[lb].sum(axis=0)
    return lb


def num_class():
    return len(wordname_16)


if __name__ == '__main__':
    # dataset = DotaDataset(settype='cv_train')
    # dataset.__getitem__(0)
    # dataset = DotaDataset(settype='cv_valid')
    dataset = DotaDataset(settype='valid', test_mode=True)
    dataset.__getitem__(0)
