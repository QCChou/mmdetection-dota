import math
import numpy as np
import cv2
import torch
from torch.nn.modules.module import Module


class RotatedRoIPool(Module):
    def _convert_affine(self, rois, feature_shape):
        feature_height, feature_width = feature_shape[-2:]
        idxs, affines = [], []
        for roi in rois:
            idx, x1, y1, x2, y2, theta = roi.cpu().numpy()
            degree = - 180.0 * theta / math.pi
            center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
            width, height = (x2 - x1), (y2 - y1)

            affine = cv2.getRotationMatrix2D(center, degree, self.spatial_scale)
            affine[0, 2] -= center[0]
            affine[1, 2] -= center[1]

            affine[0, :] *= 2.0 / width
            affine[1, :] *= 2.0 / height

            affine_inv = cv2.invertAffineTransform(affine)

            src = np.array([[0, 0, 1], [1, 0, 1], [0, 1, 1]], dtype=np.float32)
            dest = np.matmul(src, affine_inv.T).astype(np.float32)

            src = np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float32)
            dest[:, 0] = 2.0 * dest[:, 0] / feature_width - 1.0
            dest[:, 1] = 2.0 * dest[:, 1] / feature_height - 1.0

            affine = cv2.getAffineTransform(src, dest)

            idxs.append(int(idx))
            affines.append(affine)
        return idxs, torch.from_numpy(np.array(affines, dtype=np.float32))

    def _batch_features(self, features, idxs):
        return torch.stack([features[idx] for idx in idxs])

    def __init__(self, out_size, spatial_scale=1.0, max_pool=False, batch=None):
        super(RotatedRoIPool, self).__init__()

        self.out_size = out_size * (2 if max_pool else 1)
        self.spatial_scale = spatial_scale
        self.max_pool = max_pool
        self.batch = batch

    def forward(self, features, rois):
        # type: (Tensor, Tensor) -> Tensor
        r"""
        Args:
            features: `[batch_size, channels, height, width]` features
            roi: `[candidates, 6]`
                `candidates x [feature_index, x1, y1, x2, y2, theata]`
                theta: CCW radian
        Returns:
            rotated ROI pooled features `[candidates, channels, self.out_size, self.out_size]`
        """
        assert rois.size(1) == 6
        channels = features.size(1)

        idxs, affines = self._convert_affine(rois.detach(), features.shape)
        affines = affines.to(device=features.device)

        size = torch.Size((len(affines), channels, self.out_size, self.out_size))
        grids = torch.nn.functional.affine_grid(affines, size)

        features = self._batch_features(features, idxs)

        output = torch.nn.functional.grid_sample(features, grids)
        if self.max_pool:
            output = torch.nn.functional.max_pool2d(output, 2, 2)
        return output

