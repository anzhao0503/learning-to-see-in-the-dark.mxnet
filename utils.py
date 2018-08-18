import numpy as np
import mxnet as mx
import mxnet.ndarray as F
import numbers
import random


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, gt):
        for t in self.transforms:
            img, gt = t(img, gt)
        return img, gt


class RandomCrop(object):
    '''
    Crops the given numpy arrays with size(1, H, W, C) randomly
    :param size: crop size
    :param scale: 2 for Sony 3 for Fuji
    '''
    def __init__(self, size, scale):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.scale = scale

    def __call__(self, img, gt):
        H = img.shape[1]
        W = img.shape[2]
        xx = np.random.randint(0, W - self.size[0])
        yy = np.random.randint(0, H - self.size[1])
        img = img[:, yy: yy + self.size[1], xx: xx + self.size[0], :]
        gt = gt[:, yy * self.scale: (yy + self.size[1]) * self.scale,
             xx * self.scale: (xx + self.size[0]) * self.scale, :]
        return img, gt


class RandomFlipLeftRight(object):
    '''
    LeftRight flip with the probability of p
    '''
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, gt):
        if random.random() < self.p:
            img = np.flip(img, axis=1)
            gt = np.flip(gt, axis=1)
        return img, gt


class RandomFlipTopBottom(object):
    '''
    TopBottom flip with the probability of p
    '''
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, gt):
        if random.random() < self.p:
            img = np.flip(img, axis=0)
            gt = np.flip(gt, axis=0)
        return img, gt


class RandomTranspose(object):
    '''
    Transpose the H and W axis with the probability of p
    '''
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, gt):
        if random.random() < 0.5:
            img = np.transpose(img, (0, 2, 1, 3))
            gt = np.transpose(gt, (0, 2, 1, 3))
        return img, gt


class ToTensor(object):
    def __init__(self):
        super(ToTensor, self).__init__()

    def __call__(self, img, gt):
        img = F.array(np.transpose(img, (0, 3, 1, 2)).astype('float32'))
        gt = F.array(np.transpose(gt, (0, 3, 1, 2)).astype('float32'))
        return img, gt
