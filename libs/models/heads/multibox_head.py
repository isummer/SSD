import torch
import torch.nn as nn


class MultiBoxLayer(nn.Module):

    def __init__(self, cfg, num_classes):
        super(MultiBoxLayer, self).__init__()
        feat_channels = cfg['feat_channels']
        self.num_anchors = cfg['num_anchors']
        self.num_classes = num_classes

        self.loc_layers = nn.ModuleList()
        self.cls_layers = nn.ModuleList()
        for i in range(len(feat_channels)):
            in_channels = feat_channels[i]
            self.loc_layers.append(
                nn.Conv2d(in_channels, self.num_anchors[i] * 4, kernel_size=3, padding=1)
            )
            self.cls_layers.append(
                nn.Conv2d(in_channels, self.num_anchors[i] * self.num_classes, kernel_size=3, padding=1)
            )

    @staticmethod
    def reshape_layer(x, d):
        batch_size = x.size(0)
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(batch_size, -1, d)
        return x

    def forward(self, features):
        loc = list()
        cls = list()
        for i, x in enumerate(features):
            loc_l = self.loc_layers[i](x)
            loc_l = self.reshape_layer(loc_l, 4)
            loc.append(loc_l)
            cls_l = self.cls_layers[i](x)
            cls_l = self.reshape_layer(cls_l, self.num_classes)
            cls.append(cls_l)

        loc_preds = torch.cat(loc, 1)
        cls_preds = torch.cat(cls, 1)

        return loc_preds, cls_preds


class RefineMultiBoxLayer(nn.Module):

    def __init__(self, cfg, num_classes):
        super(RefineMultiBoxLayer, self).__init__()
        arm_channels = cfg['arm_channels']
        odm_channels = cfg['odm_channels']
        self.num_anchors = cfg['num_anchors']
        self.num_classes = num_classes

        self.arm_loc_layers = nn.ModuleList()
        self.arm_cls_layers = nn.ModuleList()
        self.odm_loc_layers = nn.ModuleList()
        self.odm_cls_layers = nn.ModuleList()
        for i in range(len(arm_channels)):
            in_channels = arm_channels[i]
            self.arm_loc_layers.append(
                nn.Conv2d(in_channels, self.num_anchors[i] * 4, kernel_size=3, padding=1)
            )
            self.arm_cls_layers.append(
                nn.Conv2d(in_channels, self.num_anchors[i] * 2, kernel_size=3, padding=1)
            )

        for i in range(len(odm_channels)):
            in_channels = odm_channels[i]
            self.odm_loc_layers.append(
                nn.Conv2d(in_channels, self.num_anchors[i] * 4, kernel_size=3, padding=1)
            )
            self.odm_cls_layers.append(
                nn.Conv2d(in_channels, self.num_anchors[i] * self.num_classes, kernel_size=3, padding=1)
            )

    @staticmethod
    def reshape_layer(x, d):
        batch_size = x.size(0)
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(batch_size, -1, d)
        return x

    def forward(self, dwfs, upfs):
        arm_loc, arm_cls = list(), list()
        odm_loc, odm_cls = list(), list()
        for i, x in enumerate(dwfs):
            arm_loc_l = self.arm_loc_layers[i](x)
            arm_loc_l = self.reshape_layer(arm_loc_l, 4)
            arm_loc.append(arm_loc_l)
            arm_cls_l = self.arm_cls_layers[i](x)
            arm_cls_l = self.reshape_layer(arm_cls_l, 2)
            arm_cls.append(arm_cls_l)
       
        arm_loc_preds = torch.cat(arm_loc, 1)
        arm_cls_preds = torch.cat(arm_cls, 1)

        for i, x in enumerate(upfs):
            odm_loc_l = self.odm_loc_layers[i](x)
            odm_loc_l = self.reshape_layer(odm_loc_l, 4)
            odm_loc.append(odm_loc_l)
            odm_cls_l = self.odm_cls_layers[i](x)
            odm_cls_l = self.reshape_layer(odm_cls_l, self.num_classes)
            odm_cls.append(odm_cls_l)

        odm_loc_preds = torch.cat(odm_loc, 1)
        odm_cls_preds = torch.cat(odm_cls, 1)

        return arm_loc_preds, arm_cls_preds, odm_loc_preds, odm_cls_preds
