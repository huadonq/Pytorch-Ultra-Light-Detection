import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ActivationBlock(nn.Module):

    def __init__(self, act_type='silu', inplace=True):
        super(ActivationBlock, self).__init__()
        assert act_type in ['silu', 'relu',
                            'leakyrelu'], 'Unsupport activation function!'
        if act_type == 'silu':
            self.act = nn.SiLU(inplace=inplace)
        elif act_type == 'relu':
            self.act = nn.ReLU(inplace=inplace)
        elif act_type == 'leakyrelu':
            self.act = nn.LeakyReLU(0.1, inplace=inplace)

    def forward(self, x):
        x = self.act(x)

        return x


class ConvBnActBlock(nn.Module):

    def __init__(self,
                 inplanes,
                 planes,
                 kernel_size,
                 stride,
                 padding,
                 groups=1,
                 has_bn=True,
                 has_act=True,
                 act_type='silu'):
        super(ConvBnActBlock, self).__init__()
        bias = False if has_bn else True

        self.layer = nn.Sequential(
            nn.Conv2d(inplanes,
                      planes,
                      kernel_size,
                      stride=stride,
                      padding=padding,
                      groups=groups,
                      bias=bias),
            nn.BatchNorm2d(planes) if has_bn else nn.Sequential(),
            ActivationBlock(act_type=act_type, inplace=True)
            if has_act else nn.Sequential(),
        )

    def forward(self, x):
        x = self.layer(x)

        return x


class DWConvBnActBlock(nn.Module):

    def __init__(self,
                 inplanes,
                 planes,
                 kernel_size,
                 stride,
                 padding,
                 groups,
                 has_bn=True,
                 has_act=True,
                 act_type='silu'):
        super(DWConvBnActBlock, self).__init__()

        self.depthwise_conv = ConvBnActBlock(inplanes,
                                             inplanes,
                                             kernel_size=kernel_size,
                                             stride=stride,
                                             padding=padding,
                                             groups=inplanes,
                                             has_bn=has_bn,
                                             has_act=has_act,
                                             act_type=act_type)
        self.pointwise_conv = ConvBnActBlock(inplanes,
                                             planes,
                                             kernel_size=1,
                                             stride=1,
                                             padding=0,
                                             groups=groups,
                                             has_bn=has_bn,
                                             has_act=has_act,
                                             act_type=act_type)

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)

        return x


class YOLOXBottleneck(nn.Module):

    def __init__(self,
                 inplanes,
                 planes,
                 block=ConvBnActBlock,
                 reduction=0.5,
                 shortcut=True,
                 act_type='silu'):
        super(YOLOXBottleneck, self).__init__()
        squeezed_planes = max(1, int(planes * reduction))
        self.conv = nn.Sequential(
            ConvBnActBlock(inplanes,
                           squeezed_planes,
                           kernel_size=1,
                           stride=1,
                           padding=0,
                           groups=1,
                           has_bn=True,
                           has_act=True,
                           act_type=act_type),
            block(squeezed_planes,
                  planes,
                  kernel_size=3,
                  stride=1,
                  padding=1,
                  groups=1,
                  has_bn=True,
                  has_act=True,
                  act_type=act_type))

        self.shortcut = True if shortcut and inplanes == planes else False

    def forward(self, x):
        out = self.conv(x)

        if self.shortcut:
            out = out + x

        del x

        return out


class YOLOXCSPBottleneck(nn.Module):
    '''
    CSP Bottleneck with 3 convolution layers
    CSPBottleneck:https://github.com/WongKinYiu/CrossStagePartialNetworks
    '''

    def __init__(self,
                 inplanes,
                 planes,
                 bottleneck_nums=1,
                 bottleneck_block_type=ConvBnActBlock,
                 reduction=0.5,
                 shortcut=True,
                 act_type='silu'):
        super(YOLOXCSPBottleneck, self).__init__()
        squeezed_planes = max(1, int(planes * reduction))
        self.conv1 = ConvBnActBlock(inplanes,
                                    squeezed_planes,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    groups=1,
                                    has_bn=True,
                                    has_act=True,
                                    act_type=act_type)
        self.conv2 = ConvBnActBlock(inplanes,
                                    squeezed_planes,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    groups=1,
                                    has_bn=True,
                                    has_act=True,
                                    act_type=act_type)
        self.conv3 = ConvBnActBlock(2 * squeezed_planes,
                                    planes,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    groups=1,
                                    has_bn=True,
                                    has_act=True,
                                    act_type=act_type)

        self.bottlenecks = nn.Sequential(*[
            YOLOXBottleneck(squeezed_planes,
                            squeezed_planes,
                            block=bottleneck_block_type,
                            reduction=1.0,
                            shortcut=shortcut,
                            act_type=act_type) for _ in range(bottleneck_nums)
        ])

    def forward(self, x):
        y1 = self.conv1(x)
        y1 = self.bottlenecks(y1)
        y2 = self.conv2(x)

        del x

        out = torch.cat([y1, y2], axis=1)
        out = self.conv3(out)

        del y1, y2

        return out




class YOLOXFPN(nn.Module):

    def __init__(self,
                 inplanes,
                 csp_nums=3,
                 csp_shortcut=False,
                 block=ConvBnActBlock,
                 act_type='silu'):
        super(YOLOXFPN, self).__init__()
        # inplanes:[C3_inplanes,C4_inplanes,C5_inplanes]

        self.p5_reduce_conv = ConvBnActBlock(inplanes[2],
                                             inplanes[1],
                                             kernel_size=1,
                                             stride=1,
                                             padding=0,
                                             groups=1,
                                             has_bn=True,
                                             has_act=True,
                                             act_type=act_type)
        self.p4_conv1 = YOLOXCSPBottleneck(int(inplanes[1] * 2),
                                           inplanes[1],
                                           bottleneck_nums=csp_nums,
                                           bottleneck_block_type=block,
                                           reduction=0.5,
                                           shortcut=csp_shortcut,
                                           act_type=act_type)
        self.p4_reduce_conv = ConvBnActBlock(inplanes[1],
                                             inplanes[0],
                                             kernel_size=1,
                                             stride=1,
                                             padding=0,
                                             groups=1,
                                             has_bn=True,
                                             has_act=True,
                                             act_type=act_type)
        self.p3_conv1 = YOLOXCSPBottleneck(int(inplanes[0] * 2),
                                           inplanes[0],
                                           bottleneck_nums=csp_nums,
                                           bottleneck_block_type=block,
                                           reduction=0.5,
                                           shortcut=csp_shortcut,
                                           act_type=act_type)
        self.p3_up_conv = ConvBnActBlock(inplanes[0],
                                         inplanes[0],
                                         kernel_size=3,
                                         stride=2,
                                         padding=1,
                                         groups=1,
                                         has_bn=True,
                                         has_act=True,
                                         act_type=act_type)
        self.p4_conv2 = YOLOXCSPBottleneck(int(inplanes[0] * 2),
                                           inplanes[1],
                                           bottleneck_nums=csp_nums,
                                           bottleneck_block_type=block,
                                           reduction=0.5,
                                           shortcut=csp_shortcut,
                                           act_type=act_type)
        self.p4_up_conv = ConvBnActBlock(inplanes[1],
                                         inplanes[1],
                                         kernel_size=3,
                                         stride=2,
                                         padding=1,
                                         groups=1,
                                         has_bn=True,
                                         has_act=True,
                                         act_type=act_type)
        self.p5_conv1 = YOLOXCSPBottleneck(int(inplanes[1] * 2),
                                           inplanes[2],
                                           bottleneck_nums=csp_nums,
                                           bottleneck_block_type=block,
                                           reduction=0.5,
                                           shortcut=csp_shortcut,
                                           act_type=act_type)

    def forward(self, inputs):
        [C3, C4, C5] = inputs

        P5 = self.p5_reduce_conv(C5)

        del C5

        P5_upsample = F.interpolate(P5,
                                    size=(C4.shape[2], C4.shape[3]),
                                    mode='bilinear',
                                    align_corners=True)
        P4 = torch.cat([C4, P5_upsample], axis=1)

        del C4, P5_upsample

        P4 = self.p4_conv1(P4)
        P4 = self.p4_reduce_conv(P4)

        P4_upsample = F.interpolate(P4,
                                    size=(C3.shape[2], C3.shape[3]),
                                    mode='bilinear',
                                    align_corners=True)
        P3 = torch.cat([C3, P4_upsample], axis=1)

        del C3, P4_upsample

        P3_out = self.p3_conv1(P3)

        P3_up = self.p3_up_conv(P3_out)
        P4 = torch.cat([P3_up, P4], axis=1)
        P4_out = self.p4_conv2(P4)

        del P4

        P4_up = self.p4_up_conv(P4_out)
        P5 = torch.cat([P4_up, P5], axis=1)
        P5_out = self.p5_conv1(P5)

        del P5

        return [P3_out, P4_out, P5_out]