import math
import torch
import torch.nn as nn
import torchvision.ops

class DeformableConv2d(nn.Module):
    def __init__(self,
                 inplanes,
                 planes,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 dilation=1,
                 groups=1,
                 bias=True):
        super(DeformableConv2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        self.weight = nn.Parameter(
            torch.Tensor(planes, inplanes // groups, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(planes))
        else:
            self.bias = None

        self.offset_conv = nn.Conv2d(inplanes,
                                     2 * groups * kernel_size * kernel_size,
                                     kernel_size=self.kernel_size,
                                     stride=self.stride,
                                     padding=self.padding,
                                     bias=True)

        self.mask_conv = nn.Conv2d(inplanes,
                                   1 * groups * kernel_size * kernel_size,
                                   kernel_size=self.kernel_size,
                                   stride=self.stride,
                                   padding=self.padding,
                                   bias=True)

        n = inplanes * kernel_size * kernel_size
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if bias:
            self.bias.data.zero_()

        nn.init.constant_(self.offset_conv.weight, 0.)
        nn.init.constant_(self.offset_conv.bias, 0.)
        nn.init.constant_(self.mask_conv.weight, 0.)
        nn.init.constant_(self.mask_conv.bias, 0.)

    def forward(self, x):
        offset = self.offset_conv(x)
        mask = torch.sigmoid(self.mask_conv(x))

        x = torchvision.ops.deform_conv2d(input=x,
                                          offset=offset,
                                          weight=self.weight,
                                          bias=self.bias,
                                          stride=self.stride,
                                          padding=self.padding,
                                          dilation=self.dilation,
                                          mask=mask)

        return x


class CenterNetHetRegWhHead(nn.Module):

    def __init__(self,
                 inplanes,
                 num_classes,
                 planes=[256, 128, 64],
                 num_layers=3):
        super(CenterNetHetRegWhHead, self).__init__()
        self.inplanes = inplanes
        layers = []
        for i in range(num_layers):
            layers.append(
                DeformableConv2d(self.inplanes,
                                 planes[i],
                                 kernel_size=3,
                                 stride=1,
                                 padding=1,
                                 dilation=1,
                                 groups=1,
                                 bias=False))
            layers.append(nn.BatchNorm2d(planes[i]))
            layers.append(nn.ReLU(inplace=True))
            layers.append(
                nn.ConvTranspose2d(planes[i],
                                   planes[i],
                                   kernel_size=4,
                                   stride=2,
                                   padding=1,
                                   output_padding=0,
                                   bias=False))
            layers.append(nn.BatchNorm2d(planes[i]))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes[i]

        self.public_deconv_head = nn.Sequential(*layers)

        self.heatmap_head = nn.Sequential(
            nn.Conv2d(planes[-1],
                      planes[-1],
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes[-1],
                      num_classes,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=True),
        )
        self.offset_head = nn.Sequential(
            nn.Conv2d(planes[-1],
                      planes[-1],
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes[-1],
                      2,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=True),
        )
        self.wh_head = nn.Sequential(
            nn.Conv2d(planes[-1],
                      planes[-1],
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes[-1],
                      2,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=True),
        )
        self.sigmoid = nn.Sigmoid()

        for m in self.public_deconv_head.modules():
            if isinstance(m, nn.ConvTranspose2d):
                w = m.weight.data
                f = math.ceil(w.size(2) / 2)
                c = (2 * f - 1 - f % 2) / (2. * f)
                for i in range(w.size(2)):
                    for j in range(w.size(3)):
                        w[0, 0, i, j] = (1 - math.fabs(i / f - c)) * (
                            1 - math.fabs(j / f - c))
                for c in range(1, w.size(0)):
                    w[c, 0, :, :] = w[0, 0, :, :]

        self.heatmap_head[-1].bias.data.fill_(-2.19)

        for m in self.offset_head.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                nn.init.constant_(m.bias, 0)

        for m in self.wh_head.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.public_deconv_head(x)

        heatmap_output = self.heatmap_head(x)
        offset_output = self.offset_head(x)
        wh_output = self.wh_head(x)

        heatmap_output = self.sigmoid(heatmap_output)

        return heatmap_output, offset_output, wh_output