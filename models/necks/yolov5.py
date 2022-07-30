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


class Bottleneck(nn.Module):

    def __init__(self,
                 inplanes,
                 planes,
                 reduction=0.5,
                 shortcut=True,
                 act_type='silu'):
        super(Bottleneck, self).__init__()
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
            ConvBnActBlock(squeezed_planes,
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


class CSPBottleneck(nn.Module):
    '''
    CSP Bottleneck with 3 convolution layers
    CSPBottleneck:https://github.com/WongKinYiu/CrossStagePartialNetworks
    '''

    def __init__(self,
                 inplanes,
                 planes,
                 bottleneck_nums=1,
                 reduction=0.5,
                 shortcut=True,
                 act_type='silu'):
        super(CSPBottleneck, self).__init__()
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
            Bottleneck(squeezed_planes,
                       squeezed_planes,
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





class YOLOV5FPN(nn.Module):

    def __init__(self,
                 inplanes,
                 csp_nums=3,
                 csp_shortcut=False,
                 per_level_num_anchors=3,
                 num_classes=80,
                 act_type='silu'):
        super(YOLOV5FPN, self).__init__()
        # inplanes:[C3_inplanes,C4_inplanes,C5_inplanes]
        self.per_level_num_anchors = per_level_num_anchors

        self.P5_fpn_1 = CSPBottleneck(inplanes[2],
                                      inplanes[2],
                                      bottleneck_nums=csp_nums,
                                      reduction=0.5,
                                      shortcut=csp_shortcut,
                                      act_type=act_type)
        self.P5_fpn_2 = ConvBnActBlock(inplanes[2],
                                       inplanes[1],
                                       kernel_size=1,
                                       stride=1,
                                       padding=0,
                                       groups=1,
                                       has_bn=True,
                                       has_act=True,
                                       act_type=act_type)

        self.P4_fpn_1 = CSPBottleneck(int(inplanes[1] * 2),
                                      inplanes[1],
                                      bottleneck_nums=csp_nums,
                                      reduction=0.5,
                                      shortcut=csp_shortcut,
                                      act_type=act_type)
        self.P4_fpn_2 = ConvBnActBlock(inplanes[1],
                                       inplanes[0],
                                       kernel_size=1,
                                       stride=1,
                                       padding=0,
                                       groups=1,
                                       has_bn=True,
                                       has_act=True,
                                       act_type=act_type)

        self.P3_out = CSPBottleneck(int(inplanes[0] * 2),
                                    inplanes[0],
                                    bottleneck_nums=csp_nums,
                                    reduction=0.5,
                                    shortcut=csp_shortcut,
                                    act_type=act_type)
        self.P3_pred_conv = nn.Conv2d(inplanes[0],
                                      per_level_num_anchors *
                                      (1 + 4 + num_classes),
                                      kernel_size=1,
                                      stride=1,
                                      padding=0,
                                      bias=True)
        self.P3_pan_1 = ConvBnActBlock(inplanes[0],
                                       inplanes[0],
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       groups=1,
                                       has_bn=True,
                                       has_act=True,
                                       act_type=act_type)

        self.P4_out = CSPBottleneck(inplanes[1],
                                    inplanes[1],
                                    bottleneck_nums=csp_nums,
                                    reduction=0.5,
                                    shortcut=csp_shortcut,
                                    act_type=act_type)
        self.P4_pred_conv = nn.Conv2d(inplanes[1],
                                      per_level_num_anchors *
                                      (1 + 4 + num_classes),
                                      kernel_size=1,
                                      stride=1,
                                      padding=0,
                                      bias=True)
        self.P4_pan_1 = ConvBnActBlock(inplanes[1],
                                       inplanes[1],
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       groups=1,
                                       has_bn=True,
                                       has_act=True,
                                       act_type=act_type)

        self.P5_out = CSPBottleneck(inplanes[2],
                                    inplanes[2],
                                    bottleneck_nums=csp_nums,
                                    reduction=0.5,
                                    shortcut=csp_shortcut,
                                    act_type=act_type)
        self.P5_pred_conv = nn.Conv2d(inplanes[2],
                                      per_level_num_anchors *
                                      (1 + 4 + num_classes),
                                      kernel_size=1,
                                      stride=1,
                                      padding=0,
                                      bias=True)
        self.sigmoid = nn.Sigmoid()

        # https://arxiv.org/abs/1708.02002 section 3.3
        p5_bias = self.P5_pred_conv.bias.view(per_level_num_anchors, -1)
        # init obj pred value,per image(640 resolution) has 8 objects,stride=32
        p5_bias.data[:, 0] += math.log(8 / (640 / 32)**2)
        # init cls pred value
        p5_bias.data[:, 5:] += math.log(0.6 / (num_classes - 0.99))
        self.P5_pred_conv.bias = torch.nn.Parameter(p5_bias.view(-1),
                                                    requires_grad=True)

        p4_bias = self.P4_pred_conv.bias.view(per_level_num_anchors, -1)
        # init obj pred value,per image(640 resolution) has 8 objects,stride=16
        p4_bias.data[:, 0] += math.log(8 / (640 / 16)**2)
        # init cls pred value
        p4_bias.data[:, 5:] += math.log(0.6 / (num_classes - 0.99))
        self.P4_pred_conv.bias = torch.nn.Parameter(p4_bias.view(-1),
                                                    requires_grad=True)

        p3_bias = self.P3_pred_conv.bias.view(per_level_num_anchors, -1)
        # init obj pred value,per image(640 resolution) has 8 objects,stride=8
        p3_bias.data[:, 0] += math.log(8 / (640 / 8)**2)
        # init cls pred value
        p3_bias.data[:, 5:] += math.log(0.6 / (num_classes - 0.99))
        self.P3_pred_conv.bias = torch.nn.Parameter(p3_bias.view(-1),
                                                    requires_grad=True)

    def forward(self, inputs):
        [C3, C4, C5] = inputs

        P5 = self.P5_fpn_1(C5)
        P5 = self.P5_fpn_2(P5)

        del C5

        P5_upsample = F.interpolate(P5,
                                    size=(C4.shape[2], C4.shape[3]),
                                    mode='bilinear',
                                    align_corners=True)
        P4 = torch.cat([C4, P5_upsample], axis=1)

        del C4, P5_upsample

        P4 = self.P4_fpn_1(P4)
        P4 = self.P4_fpn_2(P4)

        P4_upsample = F.interpolate(P4,
                                    size=(C3.shape[2], C3.shape[3]),
                                    mode='bilinear',
                                    align_corners=True)
        P3 = torch.cat([C3, P4_upsample], axis=1)

        del C3, P4_upsample

        P3 = self.P3_out(P3)
        P3_result = P3
        P3_out = self.P3_pred_conv(P3)

        P3 = self.P3_pan_1(P3)
        P4 = torch.cat([P3, P4], axis=1)

        del P3

        P4 = self.P4_out(P4)
        P4_result = P4
        P4_out = self.P4_pred_conv(P4)

        P4 = self.P4_pan_1(P4)
        P5 = torch.cat([P4, P5], axis=1)

        del P4

        P5 = self.P5_out(P5)
        
        P5_result = P5
    

        return [P3_result, P4_result, P5_result]



# class YOLOV5FPNHead(nn.Module):

#     def __init__(self,
#                  inplanes,
#                  csp_nums=3,
#                  csp_shortcut=False,
#                  per_level_num_anchors=3,
#                  num_classes=80,
#                  act_type='silu'):
#         super(YOLOV5FPNHead, self).__init__()
#         # inplanes:[C3_inplanes,C4_inplanes,C5_inplanes]
#         self.per_level_num_anchors = per_level_num_anchors

#         self.P5_fpn_1 = CSPBottleneck(inplanes[2],
#                                       inplanes[2],
#                                       bottleneck_nums=csp_nums,
#                                       reduction=0.5,
#                                       shortcut=csp_shortcut,
#                                       act_type=act_type)
#         self.P5_fpn_2 = ConvBnActBlock(inplanes[2],
#                                        inplanes[1],
#                                        kernel_size=1,
#                                        stride=1,
#                                        padding=0,
#                                        groups=1,
#                                        has_bn=True,
#                                        has_act=True,
#                                        act_type=act_type)

#         self.P4_fpn_1 = CSPBottleneck(int(inplanes[1] * 2),
#                                       inplanes[1],
#                                       bottleneck_nums=csp_nums,
#                                       reduction=0.5,
#                                       shortcut=csp_shortcut,
#                                       act_type=act_type)
#         self.P4_fpn_2 = ConvBnActBlock(inplanes[1],
#                                        inplanes[0],
#                                        kernel_size=1,
#                                        stride=1,
#                                        padding=0,
#                                        groups=1,
#                                        has_bn=True,
#                                        has_act=True,
#                                        act_type=act_type)

#         self.P3_out = CSPBottleneck(int(inplanes[0] * 2),
#                                     inplanes[0],
#                                     bottleneck_nums=csp_nums,
#                                     reduction=0.5,
#                                     shortcut=csp_shortcut,
#                                     act_type=act_type)
#         self.P3_pred_conv = nn.Conv2d(inplanes[0],
#                                       per_level_num_anchors *
#                                       (1 + 4 + num_classes),
#                                       kernel_size=1,
#                                       stride=1,
#                                       padding=0,
#                                       bias=True)
#         self.P3_pan_1 = ConvBnActBlock(inplanes[0],
#                                        inplanes[0],
#                                        kernel_size=3,
#                                        stride=2,
#                                        padding=1,
#                                        groups=1,
#                                        has_bn=True,
#                                        has_act=True,
#                                        act_type=act_type)

#         self.P4_out = CSPBottleneck(inplanes[1],
#                                     inplanes[1],
#                                     bottleneck_nums=csp_nums,
#                                     reduction=0.5,
#                                     shortcut=csp_shortcut,
#                                     act_type=act_type)
#         self.P4_pred_conv = nn.Conv2d(inplanes[1],
#                                       per_level_num_anchors *
#                                       (1 + 4 + num_classes),
#                                       kernel_size=1,
#                                       stride=1,
#                                       padding=0,
#                                       bias=True)
#         self.P4_pan_1 = ConvBnActBlock(inplanes[1],
#                                        inplanes[1],
#                                        kernel_size=3,
#                                        stride=2,
#                                        padding=1,
#                                        groups=1,
#                                        has_bn=True,
#                                        has_act=True,
#                                        act_type=act_type)

#         self.P5_out = CSPBottleneck(inplanes[2],
#                                     inplanes[2],
#                                     bottleneck_nums=csp_nums,
#                                     reduction=0.5,
#                                     shortcut=csp_shortcut,
#                                     act_type=act_type)
#         self.P5_pred_conv = nn.Conv2d(inplanes[2],
#                                       per_level_num_anchors *
#                                       (1 + 4 + num_classes),
#                                       kernel_size=1,
#                                       stride=1,
#                                       padding=0,
#                                       bias=True)
#         self.sigmoid = nn.Sigmoid()

#         # https://arxiv.org/abs/1708.02002 section 3.3
#         p5_bias = self.P5_pred_conv.bias.view(per_level_num_anchors, -1)
#         # init obj pred value,per image(640 resolution) has 8 objects,stride=32
#         p5_bias.data[:, 0] += math.log(8 / (640 / 32)**2)
#         # init cls pred value
#         p5_bias.data[:, 5:] += math.log(0.6 / (num_classes - 0.99))
#         self.P5_pred_conv.bias = torch.nn.Parameter(p5_bias.view(-1),
#                                                     requires_grad=True)

#         p4_bias = self.P4_pred_conv.bias.view(per_level_num_anchors, -1)
#         # init obj pred value,per image(640 resolution) has 8 objects,stride=16
#         p4_bias.data[:, 0] += math.log(8 / (640 / 16)**2)
#         # init cls pred value
#         p4_bias.data[:, 5:] += math.log(0.6 / (num_classes - 0.99))
#         self.P4_pred_conv.bias = torch.nn.Parameter(p4_bias.view(-1),
#                                                     requires_grad=True)

#         p3_bias = self.P3_pred_conv.bias.view(per_level_num_anchors, -1)
#         # init obj pred value,per image(640 resolution) has 8 objects,stride=8
#         p3_bias.data[:, 0] += math.log(8 / (640 / 8)**2)
#         # init cls pred value
#         p3_bias.data[:, 5:] += math.log(0.6 / (num_classes - 0.99))
#         self.P3_pred_conv.bias = torch.nn.Parameter(p3_bias.view(-1),
#                                                     requires_grad=True)

#     def forward(self, inputs):
#         [C3, C4, C5] = inputs

#         P5 = self.P5_fpn_1(C5)
#         P5 = self.P5_fpn_2(P5)

#         del C5

#         P5_upsample = F.interpolate(P5,
#                                     size=(C4.shape[2], C4.shape[3]),
#                                     mode='bilinear',
#                                     align_corners=True)
#         P4 = torch.cat([C4, P5_upsample], axis=1)

#         del C4, P5_upsample

#         P4 = self.P4_fpn_1(P4)
#         P4 = self.P4_fpn_2(P4)

#         P4_upsample = F.interpolate(P4,
#                                     size=(C3.shape[2], C3.shape[3]),
#                                     mode='bilinear',
#                                     align_corners=True)
#         P3 = torch.cat([C3, P4_upsample], axis=1)

#         del C3, P4_upsample

#         P3 = self.P3_out(P3)
#         P3_out = self.P3_pred_conv(P3)

#         P3 = self.P3_pan_1(P3)
#         P4 = torch.cat([P3, P4], axis=1)

#         del P3

#         P4 = self.P4_out(P4)
#         P4_out = self.P4_pred_conv(P4)

#         P4 = self.P4_pan_1(P4)
#         P5 = torch.cat([P4, P5], axis=1)

#         del P4

#         P5 = self.P5_out(P5)
#         P5_out = self.P5_pred_conv(P5)

#         del P5

#         # P3_out shape:[B,255,H,W]->[B,H,W,255]->[B,H,W,3,85]
#         P3_out = P3_out.permute(0, 2, 3, 1).contiguous()
#         P3_out = P3_out.view(P3_out.shape[0], P3_out.shape[1], P3_out.shape[2],
#                              self.per_level_num_anchors, -1).contiguous()
#         # P4_out shape:[B,255,H,W]->[B,H,W,255]->[B,H,W,3,85]
#         P4_out = P4_out.permute(0, 2, 3, 1).contiguous()
#         P4_out = P4_out.view(P4_out.shape[0], P4_out.shape[1], P4_out.shape[2],
#                              self.per_level_num_anchors, -1).contiguous()
#         # P5_out shape:[B,255,H,W]->[B,H,W,255]->[B,H,W,3,85]
#         P5_out = P5_out.permute(0, 2, 3, 1).contiguous()
#         P5_out = P5_out.view(P5_out.shape[0], P5_out.shape[1], P5_out.shape[2],
#                              self.per_level_num_anchors, -1).contiguous()

#         P3_out = self.sigmoid(P3_out)
#         P4_out = self.sigmoid(P4_out)
#         P5_out = self.sigmoid(P5_out)

#         return [P3_out, P4_out, P5_out]