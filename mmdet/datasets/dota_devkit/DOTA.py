#The code is used for visulization, inspired from cocoapi
#  Licensed under the Simplified BSD License [see bsd.txt]
import json
import os
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon, Circle
import numpy as np
from collections import defaultdict
import cv2
from mmdet.datasets.dota_devkit.dota_utils import *

def _isArrayLike(obj):
    if type(obj) == str:
        return False
    return hasattr(obj, '__iter__') and hasattr(obj, '__len__')

class DOTA:
    def __init__(self, basepath):
        self.basepath = basepath
        self.labelpath = os.path.join(basepath, 'labelTxt')
        self.imagepath = os.path.join(basepath, 'images')
        self.lbpaths = GetFileFromThisRootDir(self.labelpath, ext='txt')
        self.imgpaths = GetFileFromThisRootDir(self.imagepath, ext='png')
        self.imglist = sorted([custombasename(x) for x in self.imgpaths])
        self.catToImgs = defaultdict(list)
        self.ImgToAnns = defaultdict(list)
        self.ImgToGsd = {}
        self.createIndex()

    def createIndex(self):
        for filename in self.lbpaths:
            gsd, objects = parse_dota_poly(filename)
            imgid = custombasename(filename)
            self.ImgToAnns[imgid] = objects
            for obj in objects:
                cat = obj['name']
                self.catToImgs[cat].append(imgid)
            self.ImgToGsd[imgid] = gsd

    def load_gsd(self, json_path, load_all=False):
        with open(json_path, 'r') as f:
            gsd_estimated = json.load(f)

        for key in gsd_estimated.keys():
            if not load_all and self.ImgToGsd.get(key, -1) > 0:
                continue
            self.ImgToGsd[key] = gsd_estimated[key]['prediction']
            if isinstance(gsd_estimated[key]['prediction'], list):
                for x in gsd_estimated[key]['prediction']:
                    assert x > 0.
            else:
                assert gsd_estimated[key]['prediction'] > 0.
            # print('gsd for %s loaded. %.4f' % (key, self.ImgToGsd[key]))

    def getImgIds(self, catNms=[]):
        """
        :param catNms: category names
        :return: all the image ids contain the categories
        """
        catNms = catNms if _isArrayLike(catNms) else [catNms]
        if len(catNms) == 0:
            return self.imglist
        else:
            imgids = []
            for i, cat in enumerate(catNms):
                if i == 0:
                    imgids = set(self.catToImgs[cat])
                else:
                    imgids &= set(self.catToImgs[cat])
        return list(imgids)

    def gsd(self, imgId):
        return self.ImgToGsd.get(imgId, -1.)

    def loadAnns(self, catNms=[], imgId = None, difficult=None):
        """
        :param catNms: category names
        :param imgId: the img to load anns
        :return: objects
        """
        catNms = catNms if _isArrayLike(catNms) else [catNms]
        objects = self.ImgToAnns[imgId]
        if len(catNms) == 0:
            return objects
        outobjects = [obj for obj in objects if (obj['name'] in catNms)]
        return outobjects

    def showAnns(self, objects, imgId, range):
        """
        :param catNms: category names
        :param objects: objects to show
        :param imgId: img to show
        :param range: display range in the img
        :return:
        """
        img = self.loadImgs(imgId)[0]
        plt.imshow(img)
        plt.axis('off')

        ax = plt.gca()
        ax.set_autoscale_on(False)
        polygons = []
        color = []
        circles = []
        r = 5
        for obj in objects:
            c = (np.random.random((1, 3)) * 0.6 + 0.4).tolist()[0]
            poly = obj['poly']
            polygons.append(Polygon(poly))
            color.append(c)
            point = poly[0]
            circle = Circle((point[0], point[1]), r)
            circles.append(circle)
        p = PatchCollection(polygons, facecolors=color, linewidths=0, alpha=0.4)
        ax.add_collection(p)
        p = PatchCollection(polygons, facecolors='none', edgecolors=color, linewidths=2)
        ax.add_collection(p)
        p = PatchCollection(circles, facecolors='red')
        ax.add_collection(p)

    def loadImgs(self, imgids=[]):
        """
        :param imgids: integer ids specifying img
        :return: loaded img objects
        """
        # print('isarralike:', _isArrayLike(imgids))
        imgids = imgids if _isArrayLike(imgids) else [imgids]
        # print('imgids:', imgids)
        imgs = []
        for imgid in imgids:
            filename = os.path.join(self.imagepath, imgid + '.png')
            # print('filename:', filename)
            img = cv2.imread(filename)
            imgs.append(img)
        return imgs

# if __name__ == '__main__':
#     examplesplit = DOTA('examplesplit')
#     imgids = examplesplit.getImgIds(catNms=['plane'])
#     img = examplesplit.loadImgs(imgids)
#     for imgid in imgids:
#         anns = examplesplit.loadAnns(imgId=imgid)
#         examplesplit.showAnns(anns, imgid, 2)