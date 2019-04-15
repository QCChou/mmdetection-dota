import json
import logging
import os
import random

import colored
import fire
import mmcv
import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import DataParallel, AdaptiveAvgPool2d, SmoothL1Loss, AdaptiveMaxPool2d, Module
from torch.utils.data import Dataset
from torchvision.models import resnet18, resnet34
from torchvision.transforms import transforms
from pretrainedmodels.models import *
from tqdm import tqdm

from mmdet.core.evaluation.topk import accuracy_classification
from mmdet.core.utils.SampleDataLoader import SampleParallelDataLoader
from mmdet.core.utils.accumulator import Accumulator
from mmdet.datasets.dota_devkit.DOTA import DOTA

logger = logging.getLogger('GSDEstimator')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


model_path = '/data/public/rw/team-autolearn/aerial/gsd_estimation/models/'
dota_root = '/data/public/rw/team-autolearn/aerial/DOTA/v1.5_hbb_190402/'
input_size = 800
discretize_interval = 0.05
num_class = int(2.0 / discretize_interval) + 1
Image.MAX_IMAGE_PIXELS = None


class DotaDataset(Dataset):
    def __init__(self, settype='train', filter_mode=-1):
        if settype not in ['train', 'valid', 'test']:
            raise ValueError('invalid settype=%s' % settype)

        self.settype = settype
        self.dota = DOTA(os.path.join(dota_root, settype))

        x = self.dota.imglist
        self.index = list(range(len(x)))
        logger.info('settype=%s loaded, size=%d' % (self.settype, len(self.index)))

        # filter out datas without gsd info. filter_mode=-1 for images with gsd, filter_mode=1 for ones without gsd.
        if filter_mode:
            self.index = list(filter(lambda i: filter_mode * self.dota.gsd(self.dota.imglist[i]) < 0, self.index))
            logger.info('settype=%s filtered, size=%d' % (self.settype, len(self.index)))
            assert len(self.index) > 0

        self.transform = transforms.Compose([
            transforms.RandomCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    def get_id(self, index):
        imgid = self.dota.imglist[self.index[index]]
        return imgid

    def get_img(self, index):
        imgid = self.get_id(index)
        img = self.dota.loadImgs(imgid)[0]
        return img

    def __getitem__(self, index):
        imgid = self.get_id(index)
        gsd = self.dota.gsd(imgid)
        img = self.get_img(index)

        h, w = img.shape[:2]
        min_scale = input_size / min(h, w)
        if False and self.settype in ['train']:
            # randomly rescale based_on gsd
            target_gsd = random.uniform(0.1, 2.0)
            scale = gsd / target_gsd
            scale = max(scale, min_scale)
        else:
            scale = max(1.0, min_scale)

        img = mmcv.imrescale(img, scale)
        gsd = gsd / scale

        # if image is too big...
        __size_m = 3 * input_size
        h, w = img.shape[:2]
        assert img.shape[2] == 3

        if h > __size_m:
            s = random.randint(0, h - input_size - 1)
            img = img[s:s + __size_m]
        if w > __size_m:
            s = random.randint(0, w - input_size - 1)
            img = img[:, s:s + __size_m]

        # transform
        img = Image.fromarray(img)
        img = self.transform(img)

        # gsd onehot
        lb = int((gsd + discretize_interval / 2) / discretize_interval)
        lb = min(lb, num_class - 1)
        lb = max(lb, 0)
        lb = np.eye(num_class, dtype=np.int)[lb]
        return img, lb, gsd, scale

    def __len__(self):
        return len(self.index)


class GSDEstimator:
    def __init__(self, backbone='resnet34', model_path='', optimizer='adam', lr=0.001, weight_decay=0.0001):
        self.backbone = backbone
        if backbone == 'resnet18':
            model = resnet18(pretrained=True)
            model.avgpool = AdaptiveAvgPool2d((1, 1))
            model.fc = nn.Sequential(
                nn.Linear(512, 1),
                nn.Sigmoid()
            )
        elif backbone == 'resnet34':
            model = resnet34(pretrained=True)
            model.avgpool = AdaptiveAvgPool2d((1, 1))
            model.fc = nn.Sequential(
                nn.Linear(512, num_class),
                nn.Sigmoid()
            )
        elif backbone == 'resnet50':
            model = resnet34(pretrained=True)
            model.avgpool = AdaptiveAvgPool2d((1, 1))
            model.fc = nn.Sequential(
                nn.Linear(512, num_class),
                nn.Sigmoid()
            )
        elif backbone == 'se_resnext50':
            model = se_resnext50_32x4d(pretrained='imagenet')
            model.avg_pool = AdaptiveAvgPool2d((1, 1))
            model.last_linear = nn.Sequential(
                nn.Linear(512 * 4, num_class),
                nn.Sigmoid()
            )
        elif backbone == 'se_resnext101':
            model = se_resnext101_32x4d(pretrained='imagenet')
            model.avg_pool = AdaptiveAvgPool2d((1, 1))
            model.last_linear = nn.Sequential(
                nn.Linear(512 * 4, num_class),
                nn.Sigmoid()
            )
        else:
            raise ValueError('backbone')
        self.model = DataParallel(model).cuda()
        self.loss_fn = torch.nn.BCELoss()

        if optimizer == 'sgd':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer == 'momentum':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay, nesterov=True)
        elif optimizer == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay, amsgrad=True)
        else:
            raise ValueError(optimizer)

        if model_path:
            self.epoch = self._load(model_path)
        else:
            self.epoch = 0

    def train(self, batch=64, epoch=140, decay_epoch=30):
        trainset = DotaDataset('train')
        trainloader = SampleParallelDataLoader(trainset, batch_size=batch, shuffle=True, pin_memory=True, drop_last=True, num_workers=64)
        scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=decay_epoch, gamma=0.1)
        epoch_gen = range(self.epoch + 1, epoch + 1)

        for i in epoch_gen:
            self.epoch = i
            scheduler.step()

            self.model.train()
            accumulator = Accumulator()
            epoch_gen = tqdm(trainloader, desc='[epoch%03d] lr=%.6f' % (i, self.optimizer.param_groups[0]['lr']), total=len(trainset) // batch)
            for data_idx, (data, label, gsd, scale) in enumerate(epoch_gen):
                data, label = data.cuda(), label.cuda()
                self.optimizer.zero_grad()

                preds = self.model(data)
                loss = self.loss_fn(preds, label.float())
                top1, top3, top5 = accuracy_classification(preds, label, topk=(1, 3, 5))

                nn.utils.clip_grad_norm_(self.model.parameters(), 8)
                loss.backward()
                accumulator.add(loss=loss.item() * len(data), top1=top1.item() * len(data), top3=top3.item() * len(data), top5=top5.item() * len(data), cnt=len(data))
                self.optimizer.step()

                epoch_gen.set_postfix(accumulator / 'cnt')

            if i % decay_epoch == 0 or i == epoch:
                self._save('%s_latest.pth' % self.backbone, epoch=i)
                self.dota('train', min_vote=0)
                self.dota('valid', min_vote=0)

    def _save(self, path, epoch):
        fullpath = os.path.join(model_path, path)
        torch.save({
            'backbone': self.backbone,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epoch': epoch
        }, fullpath)
        logger.info('model saved... %s' % fullpath)

    def _load(self, path):
        fullpath = os.path.join(model_path, path)
        logger.info('model load from... %s' % fullpath)
        checkpoint = torch.load(fullpath)
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
        return epoch

    def inference(self):
        pass

    def dota(self, settype='train', num_patch=5, min_vote=3, filter_mode=0, verbose=0):
        if verbose:
            log = print
        else:
            log = lambda x: x
        dataset = DotaDataset(settype, filter_mode)
        loaders = [SampleParallelDataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True, drop_last=True, num_workers=4) for _ in range(num_patch)]

        self.model.eval()
        accumulator = Accumulator()
        loader_iters = [iter(x) for x in loaders]
        result_json = {}
        for data_idx, datas in enumerate(range(len(dataset))):
            data_id = dataset.get_id(data_idx)
            top1s, top3s, top5s, losses, preds, scales = [], [], [], [], [], []
            orig_gsd = 0.
            for p in range(num_patch):
                data, label, gsd, scale = next(loader_iters[p])
                orig_gsd = gsd * scale
                data, label = data.cuda(), label.cuda()
                pred = self.model(data)
                loss = self.loss_fn(pred, label.float())
                top1, top3, top5 = accuracy_classification(pred, label, topk=(1, 3, 5))

                scales.append(scale)
                top1s.append(top1.item()); top3s.append(top3.item()); top5s.append(top5.item()); losses.append(loss.item()); preds.append(pred.detach().cpu().numpy())

            # estimate gsd
            scales = np.array(scales)
            preds = np.array(np.squeeze(preds))
            preds_sum_raw = np.sum(preds, axis=0)
            preds_cnt = np.zeros_like(preds, dtype=np.int)
            preds_cnt[np.arange(len(preds)), preds.argmax(1)] = 1
            preds_sum = np.sum(preds_cnt, axis=0)
            pred_idx = np.argmax(preds_sum_raw)
            bin_average = np.array([i * discretize_interval for i in range(num_class)])
            if preds_sum[pred_idx] < min_vote:
                log('data=%s failed on estimating gsd, preds_cnt=%d %s' % (data_id, preds_sum[pred_idx], preds_sum))
                estimated_gsd = np.dot(np.sum(preds * bin_average, axis=1), scales) / num_patch
            else:
                indices = slice(max(0, pred_idx - 2), pred_idx + 2, 1)
                estimateds = []
                for pred, pred_onehot, scale in zip(preds, preds_cnt, scales):
                    if pred_onehot[pred_idx] != 1:
                        continue
                    estimated = sum(pred[indices] * bin_average[indices]) * scale
                    estimateds.append(estimated)
                estimated_gsd = np.mean(estimateds)

            delta = abs(orig_gsd - estimated_gsd)
            if orig_gsd < 0:
                attr = colored.fg("blue")
            elif delta > 0.2:
                attr = colored.fg("red")
            else:
                attr = colored.attr('reset')
            log('%s%s top1=%.4f top3=%.4f top5=%.4f loss=%.4f    gsd=%.4f estimated=%.4f delta=%.4f%s' % (
                attr,
                data_id,
                sum(top1s) / num_patch,
                sum(top3s) / num_patch,
                sum(top5s) / num_patch,
                sum(losses) / num_patch,
                orig_gsd, estimated_gsd, delta,
                colored.attr('reset')
            ))
            result_json[data_id] = dict(label=float(orig_gsd), prediction=float(estimated_gsd))

            if orig_gsd > 0:
                accumulator.add(loss=sum(losses) / num_patch, top1=sum(top1s) / num_patch, top3=sum(top3s) / num_patch, top5=sum(top5s) / num_patch, cnt=1)

        if accumulator['cnt'] > 1:
            print(accumulator / 'cnt')

        if verbose:
            json_path = os.path.join(model_path, '%s_%s.json' % (self.backbone, settype))
            with open(json_path, 'w') as f:
                json.dump(result_json, f, indent=4)
            log('saved', json_path)


if __name__ == '__main__':
    fire.Fire(GSDEstimator)
