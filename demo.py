import os
import sys
import numpy as np
import cv2
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from libs.datasets import VOCDetection, VOCAnnotationTransform
from libs.datasets import VOC_CLASSES as labelmap
from libs.models import SSD
from libs.layers.functions import PriorBox
from libs.utils import Config
from libs.utils import Timer

cfgs = Config.fromfile('cfgs/ssd300_voc.json')
# print(cfgs)

colors = plt.cm.hsv(np.linspace(0, 1, cfgs.model.num_classes))
colors = (255.0 * colors[:, :3]).tolist()

# here we specify year (07 or 12) and dataset ('test', 'val', 'train') 
testset = VOCDetection('/home/pengwu/data/VOCdevkit/',
                       [('2007', 'trainval')], None, 
                       VOCAnnotationTransform())

priorbox = PriorBox(cfgs.coder)
priors = priorbox.forward().cuda()

net = SSD(cfgs.model)
if torch.cuda.is_available():
    net = net.to('cuda')

# net.load_state_dict(torch.load('./model/ssd300_voc_mAP_77.43.pth'))

from collections import OrderedDict
pretrained_weights = torch.load('./weights/ssd300_voc_epoch_210.pth')
new_state_dict = OrderedDict()
for k, v in pretrained_weights.items():
    name = k[7:] # remove 'module'
    new_state_dict[name] = v
net.load_state_dict(new_state_dict)
net.eval()

timer = Timer()

for img_id in range(10, len(testset), 10):
    cv2_img = testset.pull_image(img_id)
    # print(im_ori.shape)
    im = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    im_h, im_w = im.shape[:2]
    im = cv2.resize(im, (300, 300))
    im_to_show = cv2_img
    im = im.astype(np.float32)
    im -= (104.0, 117.0, 123.0)
    im = im[:, :, ::-1].copy()
    im_tensor = torch.from_numpy(im).permute(2, 0, 1)

    if torch.cuda.is_available():
        im_tensor = im_tensor.to('cuda')

    torch.cuda.synchronize()
    timer.tic()

    with torch.no_grad():
        loc_preds, cls_preds = net(im_tensor.unsqueeze(0))

    torch.cuda.synchronize()
    print("cost time:", timer.toc())
    predictions = (loc_preds, cls_preds, priors)
    boxes, labels, scores = net.nms_decode(predictions, 0.6, 0.45)

    for bbox, label, score in zip(boxes, labels, scores):
        bbox[0::2] *= im_w
        bbox[1::2] *= im_h
        x1, y1, x2, y2 = [int(p) for p in bbox]
        label_name = labelmap[label]
        cv2.rectangle(im_to_show, (x1,y1), (x2,y2), colors[label], 2)
        cv2.putText(im_to_show, label_name, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, colors[label], 1)

    im_scale = 0.25
    im_to_show = cv2.resize(im_to_show, None, None, fx=im_scale, fy=im_scale)
    cv2.imshow("result", im_to_show)
    cv2.waitKey(1)
