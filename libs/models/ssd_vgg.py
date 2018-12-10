'''SSD model with VGG16 as feature extractor.'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from .backbones.vgg import vgg16
from .heads.multibox_head import MultiBoxLayer
from libs.utils import box_utils as T


class L2Norm(nn.Module):
    '''L2Norm layer across all channels.'''
    def __init__(self, in_features, scale):
        super(L2Norm, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_features))
        self.reset_parameters(scale)

    def reset_parameters(self, scale):
        nn.init.constant_(self.weight, scale)

    def forward(self, x):
        x = F.normalize(x, dim=1)
        scale = self.weight[None,:,None,None]
        return scale * x


def add_extras(cfg, i, batch_norm=False):
    layers = nn.ModuleList()
    in_channels = i
    flag = False
    for k, v in enumerate(cfg):
        if in_channels != 'S':
            if v == 'S':
                layers.append(nn.Conv2d(in_channels, cfg[k+1],
                              kernel_size=(1, 3)[flag], stride=2, padding=1))
            else:
                layers.append(nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag]))
            flag = not flag
        in_channels = v
    return layers


class SSD(nn.Module):

    def __init__(self, cfgs):
        super(SSD, self).__init__()
        self.cfgs = cfgs
        # print(self.cfgs)
        self.num_classes = self.cfgs['num_classes']
        self.base = vgg16(self.cfgs['pretrained_weights'])

        self.norm4 = L2Norm(512, 20)

        self.extras = add_extras(cfgs['extras'], 1024)
        self.multibox = MultiBoxLayer(cfgs['multibox'], self.num_classes)

        print("Init weights and load pretrained weights...")
        for m in self.modules():
            if m in self.base.modules():
                continue
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        c4, c5, c6 = self.base(x)
        sources = []
        sources.append(self.norm4(c4))
        sources.append(c6)

        x = c6
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                sources.append(x)

        loc, cls = self.multibox(sources)
        return loc, cls


    def nms_decode(self, predictions, score_thresh=0.6, nms_thresh=0.45):
        loc, cls, priors = predictions
        loc = loc.squeeze()
        cls_preds = F.softmax(cls.squeeze(), dim=1)

        variances = [0.1, 0.2]

        cxcy = loc[:, :2] * variances[0] * priors[:, 2:] + priors[:, :2]
        wh = torch.exp(loc[:, 2:] * variances[1]) * priors[:, 2:]

        boxes_pred = torch.cat([cxcy - wh / 2, cxcy + wh / 2], 1)

        boxes, labels, scores = [], [], []
        num_classes = cls_preds.size(1)
        for c in range(num_classes-1):
            scores_c = cls_preds[:, c+1]  # class c corresponds to (c+1) column

            mask = scores_c > score_thresh
            if not mask.any():
                continue

            boxes_c = boxes_pred[mask]
            scores_c = scores_c[mask]

            keep = T.nms(boxes_c, scores_c, nms_thresh)
            boxes_c = boxes_c[keep]
            scores_c = scores_c[keep]

            boxes.append(boxes_c)
            labels.append(torch.LongTensor(len(boxes_c)).fill_(c+1))
            scores.append(scores_c)

        if len(boxes) > 0:
            boxes = torch.cat(boxes, 0)
            labels = torch.cat(labels, 0)
            scores = torch.cat(scores, 0)

        return boxes, labels, scores
        

def test():

    VOC_300 = {
        "model": {
            "num_classes": 21,
            "extras": [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
            "multibox": {
                "num_anchors": [4, 6, 6, 6, 4, 4],
                "feat_channels": [512, 1024, 512, 256, 256, 256]
            }
        }
    }

    net = SSD(VOC_300['model'])
    loc_preds, cls_preds = net(Variable(torch.randn(1, 3, 300, 300)))
    print(loc_preds.size(), cls_preds.size())

# test()

