import math
import torch
import torch.nn as nn


class RetinaClsHead(nn.Module):

    def __init__(self, inplanes, num_anchors, num_classes, num_layers=4):
        super(RetinaClsHead, self).__init__()
        layers = []
        for _ in range(num_layers):
            layers.append(
                nn.Conv2d(inplanes,
                          inplanes,
                          kernel_size=3,
                          stride=1,
                          padding=1))
            layers.append(nn.ReLU(inplace=True))
        self.cls_head = nn.Sequential(*layers)
        self.cls_out = nn.Conv2d(inplanes,
                                 num_anchors * num_classes,
                                 kernel_size=3,
                                 stride=1,
                                 padding=1)
        self.sigmoid = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, val=0)

        prior = 0.01
        b = -math.log((1 - prior) / prior)
        self.cls_out.bias.data.fill_(b)

    def forward(self, x):
        x = self.cls_head(x)
        x = self.cls_out(x)
        x = self.sigmoid(x)

        return x


class RetinaRegHead(nn.Module):

    def __init__(self, inplanes, num_anchors, num_layers=4):
        super(RetinaRegHead, self).__init__()
        layers = []
        for _ in range(num_layers):
            layers.append(
                nn.Conv2d(inplanes,
                          inplanes,
                          kernel_size=3,
                          stride=1,
                          padding=1))
            layers.append(nn.ReLU(inplace=True))
        self.reg_head = nn.Sequential(*layers)
        self.reg_out = nn.Conv2d(inplanes,
                                 num_anchors * 4,
                                 kernel_size=3,
                                 stride=1,
                                 padding=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, val=0)

    def forward(self, x):
        x = self.reg_head(x)
        x = self.reg_out(x)

        return x
