import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn.init as init

from .backbones.vgg import vgg16
from .heads.multibox_head import RefineMultiBoxLayer

from libs.utils import box_utils as T


class FpnAdapter(nn.Module):
    def __init__(self, block, oup=256):
        super(FpnAdapter, self).__init__()
        self.trans, self.ups, self.latents = \
            [nn.ModuleList() for _ in range(3)]
        for inp in block:
            self.trans.append(nn.Sequential(
                nn.Conv2d(inp, oup, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(oup, oup, 3, padding=1)
            ))
            self.ups.append(nn.Upsample(scale_factor=2, mode='bilinear'))
            self.latents.append(nn.Conv2d(oup, oup, 3, padding=1))

    def forward(self, features):
        trans_list = list()
        for (p, t) in zip(features, self.trans):
            trans_list.append(t(p))

        fpn_out = list()
        last = F.relu(
            self.latents[-1](trans_list[-1]), inplace=True)
        # last layer
        fpn_out.append(last)
        _up = self.ups[-1](last)

        for i in range(len(trans_list) - 2, -1, -1):
            q = F.relu(trans_list[i] + _up, inplace=True)
            q = F.relu(self.latents[i](q), inplace=True)
            fpn_out.append(q)
            if i > 0:
                _up = self.ups[i - 1](q)
        fpn_out = fpn_out[::-1]
        return fpn_out


class L2Norm(nn.Module):
    def __init__(self, n_channels, scale):
        super(L2Norm, self).__init__()
        self.n_channels = n_channels
        self.gamma = scale or None
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))
        self.reset_parameters()

    def reset_parameters(self):
        init.constant_(self.weight, self.gamma)

    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt() + self.eps
        x = x / norm
        out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(
            x) * x
        return out


def add_extras():
    layers = nn.Sequential(
        nn.Conv2d(1024, 256, 1),
        nn.ReLU(inplace=True),
        nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
        nn.ReLU(inplace=True)
    )
    return layers


class RefineDet(nn.Module):
    def __init__(self, cfgs):
        super(RefineDet, self).__init__()
        self.cfgs = cfgs
        self.num_classes = self.cfgs['num_classes']
        self.base = vgg16(cfgs.pretrained_weights)

        self.norm4 = L2Norm(512, 10)
        self.norm5 = L2Norm(512, 8)

        self.extras = add_extras()
        self.fpn = FpnAdapter(cfgs['fpn'], 256)
        self.multibox = RefineMultiBoxLayer(cfgs['multibox'], self.num_classes)
        
        print("Init weights and load pretrained weights...")
        for m in self.modules():
            if m in self.base.modules():
                continue
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_uniform_(m.weight.data)
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def forward(self, x):
        c4, c5, c6 = self.base(x)

        arm_sources = list()
        # 40x40
        arm_sources.append(self.norm4(c4))
        # 20x20
        arm_sources.append(self.norm5(c5))
        # 10x10      
        arm_sources.append(c6)
        # 5x5
        x = self.extras(c6)
        arm_sources.append(x)

        odm_sources = self.fpn(arm_sources)
        arm_loc, arm_cls, odm_loc, odm_cls = self.multibox(arm_sources, odm_sources)

        return arm_loc, arm_cls, odm_loc, odm_cls


    def nms_decode(self, predictions, score_thresh=0.6, nms_thresh=0.45):
        odm_loc, odm_cls, arm_loc, arm_cls, priors = predictions
        arm_loc = arm_loc.squeeze()
        arm_cls = F.softmax(arm_cls.squeeze(), dim=1)
        odm_loc = odm_loc.squeeze()
        odm_cls = F.softmax(odm_cls.squeeze(), dim=1)

        variances = [0.1, 0.2]

        if arm_loc is not None:
            default = torch.cat([
                arm_loc[:, :2] * variances[0] * priors[:, 2:] + priors[:, :2],
                torch.exp(arm_loc[:, 2:] * variances[1]) * priors[:, 2:]], 1)
        else:
            default = priors

        cxcy = odm_loc[:, :2] * variances[0] * default[:, 2:] + default[:, :2]
        wh = torch.exp(odm_loc[:, 2:] * variances[1]) * default[:, 2:]

        boxes_pred = torch.cat([cxcy - wh / 2, cxcy + wh / 2], 1)
        cls_preds = odm_cls

        if arm_cls is not None:
            score, obj_sel = arm_cls.max(1)
            cls_preds[obj_sel == 0] = 0

        boxes, labels, scores = [], [], []
        num_classes = cls_preds.size(1)
        for c in range(num_classes-1):
            scores_c = cls_preds[:, c+1]  # class c corresponds to (c+1) column

            mask = scores_c > score_thresh
            if not mask.any():
                continue

            boxes_c = boxes_pred[mask]
            scores_c = scores_c[mask]

            keep, _ = T.nms(boxes_c, scores_c, nms_thresh)
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

    Refine_320 = {
        "model": {
            "num_classes": 21,
            "fpn": [512, 512, 1024, 512],
            "multibox": {
                "num_anchors": [3, 3, 3, 3],
                "arm_channels": [512, 512, 1024, 512],
                "odm_channels": [256, 256, 256, 256]
            }

        }
    }

    net = RefineDet(Refine_320['model'])
    arm_loc_preds, arm_cls_preds, odm_loc_preds, odm_cls_preds = \
        net(Variable(torch.randn(1, 3, 320, 320)))

    print(arm_loc_preds.size(), arm_cls_preds.size())
    print(odm_loc_preds.size(), odm_cls_preds.size())

if __name__ == '__main__':
    test()
