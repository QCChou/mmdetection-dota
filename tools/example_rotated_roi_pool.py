# -*- coding: utf-8 -*-
import os
import sys
import logging
import math

import json as json
import numpy as np
import cv2

import torch

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)


def view(name, images):
    images = images.detach().cpu().numpy()
    batch, channels, height, width = images.shape

    cols = math.ceil(math.sqrt(batch))
    canvas = np.zeros((cols*height, cols*width, 3), dtype=np.float32)

    for idx, image in enumerate(images):
        row = idx // cols
        col = idx % cols
        canvas[row*height:(row+1)*height, col*width:(col+1)*width, :] = np.transpose(image, (1, 2, 0))
    cv2.imshow(name, canvas)


def main(args):
    logging.info('args: %s', args)
    device = torch.device('cuda', 0) if torch.cuda.is_available() else torch.device('cpu', 0)

    args.input = [f for f in args.input if '.jpg' in f]

    metas = [json.load(open(fn.replace('.jpg', '.json'), 'r')) for fn in args.input]
    logging.debug('meta: %s', metas)

    images = [cv2.imread(fn).astype(np.float32) / 255.0 for fn in args.input] # h x w x d
    images = np.array(images)
    logging.debug('loaded image: %s', images.shape)
    images = np.transpose(images, (0, 3, 1, 2)) # d x h x w
    logging.debug('transposed image: %s', images.shape)

    tensor = torch.from_numpy(images).to(device=device)
    logging.debug('tensor: %s', tensor.shape)

    logging.info('---------- mmdet roi pool implements ----------')
    from mmdet.ops.roi_pool.modules.roi_pool import RoIPool

    rois = []
    for idx, meta_list in enumerate(metas):
        rois += [[idx, meta['x'], meta['y'], meta['x'] + meta['w'], meta['y'] + meta['h']] for meta in meta_list]
    rois = torch.from_numpy(np.array(rois, dtype=np.float32)).to(device=device)

    module = RoIPool(args.output_size, 1.0)
    output = module.forward(tensor, rois)

    output_mmdet_roi_pool = output

    logging.info('---------- mmdet rotated roi pool implements ----------')
    from mmdet.ops.roi_pool.modules.rotated_roi_pool import RotatedRoIPool

    rois = []
    for idx, meta_list in enumerate(metas):
        rois += [[idx, meta['x'], meta['y'], meta['x'] + meta['w'], meta['y'] + meta['h'], meta['t']] for meta in meta_list]
    rois = torch.from_numpy(np.array(rois, dtype=np.float32)).to(device=device)

    module = RotatedRoIPool(args.output_size, 1.0)
    output = module.forward(tensor, rois)

    output_mmdet_rotated_roi_pool = output

    if args.view:
        view('roi pool', output_mmdet_roi_pool)
        view('rotated roi pool', output_mmdet_rotated_roi_pool)
        cv2.waitKey(-1)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    
    curr_path = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument('-i', '--input', type=str, nargs='+', default=[curr_path + '/roi_samples/sample1.jpg', curr_path + '/roi_samples/sample2.jpg'])
    parser.add_argument('--output-size', type=int, default=128)

    parser.add_argument('--view', action='store_true')

    parser.add_argument('--log-filename', type=str, default='')
    parser.add_argument('--debug', action='store_true')
    parsed_args = parser.parse_args()

    log_format = '[%(asctime)s %(levelname)s] %(message)s'
    level = logging.DEBUG if parsed_args.debug else logging.INFO
    if not parsed_args.log_filename:
        logging.basicConfig(level=level, format=log_format, stream=sys.stderr)
    else:
        logging.basicConfig(level=level, format=log_format, filename=parsed_args.log_filename)

    main(parsed_args)
