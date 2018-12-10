import torch
import torch.nn as nn
import torch.nn.functional as F


class VGG(nn.Module):
    def __init__(self, cfgs, batch_norm=False):
        self.inplanes = 3
        super(VGG, self).__init__()
        self.batch_norm = batch_norm

        self.conv1 = self._make_layer(cfgs[0])
        self.conv2 = self._make_layer(cfgs[1])
        self.conv3 = self._make_layer(cfgs[2])
        self.conv4 = self._make_layer(cfgs[3])
        self.conv5 = self._make_layer(cfgs[4])

        self.pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(512, 1024, 3, padding=6, dilation=6)
        self.conv7 = nn.Conv2d(1024, 1024, 1)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, planes):
        layers = []
        for v in planes:
            conv2d = nn.Conv2d(self.inplanes, v, kernel_size=3, padding=1)
            if self.batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            self.inplanes = v
        return nn.Sequential(*layers)

    def forward(self, x):
        C1 = self.conv1(x)
        x = self.maxpool(C1)
        C2 = self.conv2(x)
        x = self.maxpool(C2)
        C3 = self.conv3(x)
        x = self.maxpool(C3)
        C4 = self.conv4(x)
        x = self.maxpool(C4)
        C5 = self.conv5(x)
        # x = self.maxpool(C5)
        x = self.pool5(C5)
        x = F.relu(self.conv6(x), inplace=True)
        C6 = F.relu(self.conv7(x), inplace=True)

        return C4, C5, C6


def vgg16(pretrained_weights=None):
    cfgs = [[64, 64], [128, 128], [256, 256, 256], [512, 512, 512], [512, 512, 512]]
    model = VGG(cfgs)
    if pretrained_weights is not None:
        mapping = {
            'conv1': 0,
            'conv2': 5,
            'conv3': 10,
            'conv4': 17,
            'conv5': 24,
            'conv6': 31,
            'conv7': 33
        }
        pretrained_dict = torch.load(pretrained_weights)
        state_dict = model.state_dict()
        for k in state_dict.keys():
            splits = k.split('.')
            if splits[0] in ['conv6', 'conv7']:
                block_name = splits[0]
                name = str(mapping[block_name]) + '.' + splits[1]
            else:
                block_name, layer_id = splits[0], int(splits[1])
                name = str(mapping[block_name] + layer_id) + '.' + splits[2]
            state_dict[k] = pretrained_dict[name]
        model.load_state_dict(state_dict)
    return model

if __name__ == '__main__':
    model = vgg16()
