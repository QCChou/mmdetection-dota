import torch
import torch.nn as nn

from .base import BaseDetector
from .. import builder
from ..registry import DETECTORS
from mmdet.core import bbox2result


@DETECTORS.register_module
class SingleStageDetector(BaseDetector):

    def __init__(self,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(SingleStageDetector, self).__init__()
        self.backbone = builder.build_backbone(backbone)
        if neck is not None:
            self.neck = builder.build_neck(neck)
        self.bbox_head = builder.build_head(bbox_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=None):
        super(SingleStageDetector, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        self.bbox_head.init_weights()

    def extract_feat(self, img):
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        loss_inputs = outs + (gt_bboxes, gt_labels, img_metas, self.train_cfg)
        losses = self.bbox_head.loss(
            *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        return losses

    def _simple_test(self, img, img_meta, proposals=None, rescale=False, return_lb=False):
        if not img.is_cuda:
            img = img.cuda(torch.cuda.current_device())
            created = True
        else:
            created = False

        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        bbox_inputs = outs + (img_meta, self.test_cfg, rescale)
        bbox_list = self.bbox_head.get_bboxes(*bbox_inputs)
        det_bboxes = [x for x, _ in bbox_list]
        det_labels = [x for _, x in bbox_list]

        if created:
            del img
        if return_lb:
            return det_bboxes[0], det_labels[0], None, None
        else:
            def one_hot_embedding(labels, num_classes=17):
                """Embedding labels to one-hot form.

                Args:
                  labels: (LongTensor) class labels, sized [N,].
                  num_classes: (int) number of classes.

                Returns:
                  (tensor) encoded labels, sized [N, #classes].
                """
                y = torch.eye(num_classes).cuda()
                return y[labels + 1]
            return det_bboxes[0][:, :4], one_hot_embedding(det_labels[0]) * det_bboxes[0][:, 4].view(det_bboxes[0][:, 4].shape[0], 1), None, None

    def simple_test(self, img, img_meta, rescale=False):
        det_bboxes, det_labels, _, _ = self._simple_test(img, img_meta, rescale=False, return_lb=True)
        bbox_results = bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
        return bbox_results
