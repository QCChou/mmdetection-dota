import logging
import os
import random

import fire
import mmcv
import torch
from PIL import Image
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from torch import nn
from torch.nn import DataParallel, AdaptiveAvgPool2d, SmoothL1Loss
from torch.utils.data import Dataset
from torchvision.models import resnet18, resnet34
from torchvision.transforms import transforms
from tqdm import tqdm

from mmdet.core.utils.SampleDataLoader import SampleParallelDataLoader
from mmdet.core.utils.accumulator import Accumulator
from mmdet.datasets.dota import one_hot
from mmdet.datasets.dota_devkit.DOTA import DOTA

logger = logging.getLogger('GSDEstimator')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


model_path = '/data/public/rw/team-autolearn/aerial/GSDEstimator/models/'
dota_root = '/data/public/rw/team-autolearn/aerial/DOTA/v1.5_hbb_190402/'
input_size = 800
Image.MAX_IMAGE_PIXELS = None


class DotaDataset(Dataset):
    def __init__(self, settype='train'):
        if settype not in ['train', 'valid', 'test']:
            raise ValueError('invalid settype=%s' % settype)

        root = os.path.join(dota_root, settype)
        self.settype = settype
        self.dota = DOTA(root)

        x = self.dota.imglist
        self.index = list(range(len(x)))
        logger.info('settype=%s loaded, size=%d' % (self.settype, len(self.index)))

        # filter out datas without gsd info.
        if settype != 'test':
            self.index = list(filter(lambda i: self.dota.gsd(self.dota.imglist[i]) > 0, self.index))
            logger.info('settype=%s filtered, size=%d' % (self.settype, len(self.index)))
            assert len(self.index) > 0

        if self.settype in ['train', 'valid']:
            self.transform = transforms.Compose([
                transforms.RandomCrop(input_size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                # transforms.RandomCrop(input_size, padding=4),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

    def get_img(self, index):
        imgid = self.dota.imglist[self.index[index]]
        img = self.dota.loadImgs(imgid)[0]
        return img

    def __getitem__(self, index):
        imgid = self.dota.imglist[self.index[index]]
        gsd = self.dota.gsd(imgid)
        img = self.get_img(index)

        orig_shape = img.shape
        orig_gsd = gsd
        if self.settype in ['train']:
            # randomly rescale based_on gsd
            h, w = img.shape[:2]        # 278, 1508
            target_gsd = random.uniform(0.1, 2.0)
            scale = gsd / target_gsd    # 0.2306, 0.7632
            min_scale = input_size / min(h, w)
            scale = max(scale, min_scale)
            scale = max(1.0, min_scale) # TODO
            img = mmcv.imrescale(img, scale)

            gsd = gsd / scale

            # TODO : add paddings if image is too small
            pass

        if min(img.shape[:2]) < input_size:
            print(orig_shape, orig_gsd, target_gsd, scale, img.shape)

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

        # import cv2
        # print(gsd, img.shape)
        # cv2.imshow('a', img)
        # cv2.waitKey()

        # transform
        img = Image.fromarray(img)
        img = self.transform(img)

        return img, gsd

    def __len__(self):
        return len(self.index)


class GSDEstimator:
    def __init__(self, backbone='resnet34', model_path=''):
        if backbone == 'resnet18':
            model = resnet18()
            model.avgpool = AdaptiveAvgPool2d((1, 1))
            model.fc = nn.Sequential(
                nn.Linear(512, 1),
                nn.ReLU()   # no negative output
            )
        elif backbone == 'resnet34':
            model = resnet34()
            model.avgpool = AdaptiveAvgPool2d((1, 1))
            model.fc = nn.Sequential(
                nn.Linear(512, 1),
                nn.ReLU()  # no negative output
            )
        else:
            raise ValueError('backbone')
        self.model = DataParallel(model).cuda()
        # self.loss_fn = torch.nn.MSELoss()
        self.loss_fn = SmoothL1Loss()
        pass

    def train(self, batch=128, epoch=50, lr=0.001, optimizer='momentum'):
        trainset = DotaDataset('train')
        trainloader = SampleParallelDataLoader(trainset, batch_size=batch, shuffle=True, pin_memory=True, drop_last=True, num_workers=64)

        if optimizer == 'sgd':
            optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, weight_decay=1e-4)
        elif optimizer == 'momentum':
            optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4, nesterov=True)
        elif optimizer == 'adam':
            optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4, amsgrad=True)
        else:
            raise ValueError(optimizer)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch, eta_min=0.0)

        epoch_gen = range(1, epoch + 1)
        for i in epoch_gen:
            scheduler.step()

            self.model.train()
            accumulator = Accumulator()
            epoch_gen = tqdm(trainloader, desc='[epoch%03d] lr=%.6f' % (i, optimizer.param_groups[0]['lr']), total=len(trainset) // batch)
            for data_idx, (data, label) in enumerate(epoch_gen):
                data = data.cuda()
                label = label.cuda()
                optimizer.zero_grad()

                preds = self.model(data)
                loss = self.loss_fn(preds, label.float())

                nn.utils.clip_grad_norm_(self.model.parameters(), 8)
                loss.backward()
                accumulator.add('loss', loss.item() * len(data))
                accumulator.add('cnt', len(data))
                optimizer.step()

                epoch_gen.set_postfix(accumulator / 'cnt')

            if i % 10 == 0:
                self.dota('train')

    def inference(self):
        pass

    def dota(self, settype='valid'):
        validset = DotaDataset(settype)
        loader = SampleParallelDataLoader(validset, batch_size=1, shuffle=True, pin_memory=True, drop_last=True, num_workers=64)

        self.model.eval()
        accumulator = Accumulator()
        for data, label in loader:
            data, label = data.cuda(), label.cuda()
            preds = self.model(data)
            loss = self.loss_fn(preds, label.float())
            print(label.item(), preds.item(), loss.item())


if __name__ == '__main__':
    fire.Fire(GSDEstimator)
