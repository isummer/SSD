import os
import sys
import numpy as np
import argparse
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data as data

from libs.datasets import *
from libs.transforms import SSDAugmentation
from libs.utils import Config
from libs.utils import Timer


cfgs = Config.fromfile("./cfgs/refine320_voc.json")

dataset = VOCDetection(root=cfgs.data.dataset_root,
                       transform=SSDAugmentation(tuple(cfgs.coder.input_shape), cfgs.train.aug_params.bgr_mean))

data_loader = data.DataLoader(dataset, cfgs.train.batch_size,
                              num_workers=cfgs.train.num_workers,
                              shuffle=True, collate_fn=detection_collate,
                              pin_memory=True)


for _iter, (images, targets) in enumerate(data_loader):
    images = images.to('cuda')
    targets = [ann.to('cuda') for ann in targets]
    img = images.data.cpu().numpy()[0, ...].transpose(1, 2, 0)
    targets_b = targets[0].data.cpu().numpy()
    boxes, labels = targets_b[:, :4], targets_b[:, -1]

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img += np.array([104, 117, 123])
    img = img.astype(np.uint8)
    for bbox in boxes:
        x1, y1, x2, y2 = [int(320*x) for x in bbox]
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 0), 2)
    cv2.imshow("img", img)
    cv2.waitKey(1)
