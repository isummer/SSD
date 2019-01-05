"""Data augmentation functionality. Passed as callable transformations to
Dataset classes.

The data augmentation procedures were interpreted from @weiliu89's SSD paper
http://arxiv.org/abs/1512.02325

Ellis Brown, Max deGroot
"""

import torch
import cv2
import numpy as np
import random
import math

def matrix_iou(a, b):
    """
    return iou of a and b, numpy version for data augenmentation
    """
    lt = np.maximum(a[:, np.newaxis, :2], b[:, :2])
    rb = np.minimum(a[:, np.newaxis, 2:], b[:, 2:])

    area_i = np.prod(rb - lt, axis=2) * (lt < rb).all(axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)
    return area_i / (area_a[:, np.newaxis] + area_b - area_i)


class Compose(object):
    """Composes several augmentations together.
    Args:
        transforms (List[Transform]): list of transforms to compose.
    Example:
        >>> augmentations.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, boxes=None, labels=None):
        for t in self.transforms:
            img, boxes, labels = t(img, boxes, labels)
        return img, boxes, labels


class Normalize(object):
    def __init__(self, mean, std=(1, 1, 1)):
        self.mean = mean
        self.std = std

    def __call__(self, img, boxes=None, labels=None):
        img = img.astype(np.float32)
        img -= self.mean
        img /= self.std

        return img, boxes, labels


class Resize(object):
    def __init__(self, size=(300, 300)):
        self.size = size

    def __call__(self, img, boxes=None, labels=None):
        im_h, im_w = img.shape[:2]
        interp_methods = [
            cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_NEAREST,
            cv2.INTER_LANCZOS4
        ]
        interp_method = interp_methods[random.randrange(5)]
        img_t = cv2.resize(img, self.size, interpolation=interp_method)

        boxes = boxes.copy()
        boxes[:, 0::2] /= im_w
        boxes[:, 1::2] /= im_h

        return img_t, boxes, labels


class Crop(object):

    def __call__(self, img, boxes, labels):
        im_h, im_w = img.shape[:2]

        while True:
            mode = random.choice((
                None,
                (0.1, None),
                (0.3, None),
                (0.5, None),
                (0.7, None),
                (0.9, None),
                (None, None),
            ))

            if mode is None:
                return img, boxes, labels

            min_iou, max_iou = mode
            if min_iou is None:
                min_iou = float('-inf')
            if max_iou is None:
                max_iou = float('inf')

            for _ in range(50):
                scale = random.uniform(0.3, 1.)
                min_ratio = max(0.5, scale * scale)
                max_ratio = min(2, 1. / scale / scale)
                ratio = math.sqrt(random.uniform(min_ratio, max_ratio))
                w = int(scale * ratio * im_w)
                h = int((scale / ratio) * im_h)

                l = random.randrange(im_w - w)
                t = random.randrange(im_h - h)
                roi = np.array((l, t, l + w, t + h))

                iou = matrix_iou(boxes, roi[np.newaxis])

                if not (min_iou <= iou.min() and iou.max() <= max_iou):
                    continue

                centers = (boxes[:, :2] + boxes[:, 2:]) / 2
                mask = np.logical_and(roi[:2] < centers, centers < roi[2:]) \
                         .all(axis=1)
                if len(mask) == 0:
                    continue
                
                img = img[roi[1]:roi[3], roi[0]:roi[2]]
                boxes = boxes[mask].copy()
                labels = labels[mask].copy()

                boxes[:, :2] = np.maximum(boxes[:, :2], roi[:2])
                boxes[:, :2] -= roi[:2]
                boxes[:, 2:] = np.minimum(boxes[:, 2:], roi[2:])
                boxes[:, 2:] -= roi[:2]

                return img, boxes, labels

        return img, boxes, labels


class Distort(object):
    def __init__(self, p=.5):
        self.p = p

    def __call__(self, img, boxes, labels):

        def _convert(img, alpha=1, beta=0):
            tmp = img.astype(float) * alpha + beta
            tmp[tmp < 0] = 0
            tmp[tmp > 255] = 255
            img[:] = tmp

        img = img.copy()

        if random.random() > self.p:
            _convert(img, beta=random.uniform(-32, 32))

        if random.random() > self.p:
            _convert(img, alpha=random.uniform(0.5, 1.5))

        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        if random.random() > self.p:
            tmp = img[:, :, 0].astype(int) + random.randint(-18, 18)
            tmp %= 180
            img[:, :, 0] = tmp

        if random.random() > self.p:
            _convert(img[:, :, 1], alpha=random.uniform(0.5, 1.5))

        img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)

        return img, boxes, labels



class Expand(object):
    def __init__(self, mean, p=.5):
        self.mean = mean
        self.p = p

    def __call__(self, img, boxes, labels):
        if random.random() < self.p:
            im_h, im_w, c = img.shape
            for _ in range(50):
                scale = random.uniform(1, 4)
                min_ratio = max(0.5, 1. / scale / scale)
                max_ratio = min(2, scale * scale)
                ratio = math.sqrt(random.uniform(min_ratio, max_ratio))
                ws = scale * ratio
                hs = scale / ratio
                if ws < 1 or hs < 1:
                    continue

                w = int(ws * im_w)
                h = int(hs * im_h)
                left = random.randint(0, w - im_w)
                top = random.randint(0, h - im_h)

                expand_img = np.empty((h, w, c), dtype=img.dtype)
                expand_img[:, :, :] = self.mean
                expand_img[top:top + im_h, left:left + im_w] = img
                img = expand_img

                boxes = boxes.copy()
                boxes[:, :2] += (left, top)
                boxes[:, 2:] += (left, top)

                return img, boxes, labels

        return img, boxes, labels


class Mirror(object):
    def __init__(self, p=.5):
        self.p = p

    def __call__(self, img, boxes, labels):
        if random.random() > self.p:
            im_w = img.shape[1]
            img = img[:, ::-1]
            boxes = boxes.copy()
            boxes[:, 0::2] = im_w - boxes[:, 2::-2]

        return img, boxes, labels


class SSDAugmentation(object):
    def __init__(self, size=(300, 300), mean=(104, 117, 123), std=(1, 1, 1), p=.2):
        self.mean = mean
        self.std = std
        self.size = size
        self.p = p
        self.augment = Compose([
            Crop(),
            Distort(),
            Expand(self.mean, self.p),
            Mirror()
        ])
        self.resize = Resize(self.size)
        self.normalize = Normalize(self.mean, self.std)

    def __call__(self, img, boxes, labels):
        img_o, boxes_o, labels_o = self.resize(img, boxes, labels)
        img, boxes, labels = self.augment(img, boxes, labels)
        img, boxes, labels = self.resize(img, boxes, labels)

        boxes = boxes.copy()
        labels = labels.copy()
        boxes_w = (boxes[:, 2] - boxes[:, 0])
        boxes_h = (boxes[:, 3] - boxes[:, 1])
        keep = np.minimum(boxes_w, boxes_h) > 0.01
        boxes = boxes[keep]
        labels = labels[keep]
 
        if len(boxes) == 0:
            img, boxes, labels = self.normalize(img_o, boxes_o, labels_o)
        else:
            img, boxes, labels = self.normalize(img, boxes, labels)
        return img, boxes, labels
