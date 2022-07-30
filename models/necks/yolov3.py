import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ActivationBlock(nn.Module):

    def __init__(self, act_type='leakyrelu', inplace=True):
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
                 act_type='leakyrelu'):
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




# class Yolov3TinyFPNHead(nn.Module):

#     def __init__(self,
#                  inplanes,
#                  per_level_num_anchors=3,
#                  num_classes=80,
#                  act_type='leakyrelu'):
#         super(Yolov3TinyFPNHead, self).__init__()
#         # inplanes:[C4_inplanes,C5_inplanes]
#         self.per_level_num_anchors = per_level_num_anchors

#         self.conv1 = ConvBnActBlock(inplanes[1],
#                                     1024,
#                                     kernel_size=3,
#                                     stride=1,
#                                     padding=1,
#                                     groups=1,
#                                     has_bn=True,
#                                     has_act=True,
#                                     act_type=act_type)
#         self.conv2 = ConvBnActBlock(1024,
#                                     256,
#                                     kernel_size=1,
#                                     stride=1,
#                                     padding=0,
#                                     groups=1,
#                                     has_bn=True,
#                                     has_act=True,
#                                     act_type=act_type)
#         self.P5_conv = ConvBnActBlock(256,
#                                       512,
#                                       kernel_size=3,
#                                       stride=1,
#                                       padding=1,
#                                       groups=1,
#                                       has_bn=True,
#                                       has_act=True,
#                                       act_type=act_type)
#         self.P5_pred_conv = nn.Conv2d(512,
#                                       per_level_num_anchors *
#                                       (1 + 4 + num_classes),
#                                       kernel_size=1,
#                                       stride=1,
#                                       padding=0,
#                                       groups=1,
#                                       bias=True)
#         self.conv3 = ConvBnActBlock(256,
#                                     128,
#                                     kernel_size=1,
#                                     stride=1,
#                                     padding=0,
#                                     groups=1,
#                                     has_bn=True,
#                                     has_act=True,
#                                     act_type=act_type)
#         self.P4_conv = ConvBnActBlock(int(128 + inplanes[0]),
#                                       256,
#                                       kernel_size=3,
#                                       stride=1,
#                                       padding=1,
#                                       groups=1,
#                                       has_bn=True,
#                                       has_act=True,
#                                       act_type=act_type)
#         self.P4_pred_conv = nn.Conv2d(256,
#                                       per_level_num_anchors *
#                                       (1 + 4 + num_classes),
#                                       kernel_size=1,
#                                       stride=1,
#                                       padding=0,
#                                       groups=1,
#                                       bias=True)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, inputs):
#         [C4, C5] = inputs

#         C5 = self.conv1(C5)
#         C5 = self.conv2(C5)

#         P5 = self.P5_conv(C5)
#         P5 = self.P5_pred_conv(P5)

#         C5_upsample = F.interpolate(self.conv3(C5),
#                                     size=(C4.shape[2], C4.shape[3]),
#                                     mode='bilinear',
#                                     align_corners=True)
#         del C5
#         C4 = torch.cat([C4, C5_upsample], dim=1)

#         P4 = self.P4_conv(C4)
#         P4 = self.P4_pred_conv(P4)
#         del C4

#         # P4 shape:[B,255,H,W]->[B,H,W,255]->[B,H,W,3,85]
#         P4 = P4.permute(0, 2, 3, 1).contiguous()
#         P4 = P4.view(P4.shape[0], P4.shape[1], P4.shape[2],
#                      self.per_level_num_anchors, -1)
#         # P5 shape:[B,255,H,W]->[B,H,W,255]->[B,H,W,3,85]
#         P5 = P5.permute(0, 2, 3, 1).contiguous()
#         P5 = P5.view(P5.shape[0], P5.shape[1], P5.shape[2],
#                      self.per_level_num_anchors, -1)

#         P4[:, :, :, :, 0:3] = torch.sigmoid(P4[:, :, :, :, 0:3])
#         P4[:, :, :, :, 5:] = torch.sigmoid(P4[..., 5:])
#         P5[:, :, :, :, 0:3] = torch.sigmoid(P5[:, :, :, :, 0:3])
#         P5[:, :, :, :, 5:] = torch.sigmoid(P5[..., 5:])

#         return [P4, P5]


class Yolov3TinyFPN(nn.Module):

    def __init__(self,
                 inplanes,
                 per_level_num_anchors=3,
                 num_classes=80,
                 act_type='leakyrelu'):
        super(Yolov3TinyFPNHead, self).__init__()
        # inplanes:[C4_inplanes,C5_inplanes]
        self.per_level_num_anchors = per_level_num_anchors

        self.conv1 = ConvBnActBlock(inplanes[1],
                                    1024,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1,
                                    groups=1,
                                    has_bn=True,
                                    has_act=True,
                                    act_type=act_type)
        self.conv2 = ConvBnActBlock(1024,
                                    256,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    groups=1,
                                    has_bn=True,
                                    has_act=True,
                                    act_type=act_type)
        self.P5_conv = ConvBnActBlock(256,
                                      512,
                                      kernel_size=3,
                                      stride=1,
                                      padding=1,
                                      groups=1,
                                      has_bn=True,
                                      has_act=True,
                                      act_type=act_type)
 
        self.conv3 = ConvBnActBlock(256,
                                    128,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    groups=1,
                                    has_bn=True,
                                    has_act=True,
                                    act_type=act_type)
        self.P4_conv = ConvBnActBlock(int(128 + inplanes[0]),
                                      256,
                                      kernel_size=3,
                                      stride=1,
                                      padding=1,
                                      groups=1,
                                      has_bn=True,
                                      has_act=True,
                                      act_type=act_type)


    def forward(self, inputs):
        [C4, C5] = inputs

        C5 = self.conv1(C5)
        C5 = self.conv2(C5)

        P5 = self.P5_conv(C5)

        C5_upsample = F.interpolate(self.conv3(C5),
                                    size=(C4.shape[2], C4.shape[3]),
                                    mode='bilinear',
                                    align_corners=True)
        del C5
        C4 = torch.cat([C4, C5_upsample], dim=1)

        P4 = self.P4_conv(C4)
        del C4

        return [P4, P5]


class Yolov3FPN(nn.Module):

    def __init__(self,
                 inplanes,
                 per_level_num_anchors=3,
                 num_classes=80,
                 act_type='leakyrelu'):
        super(Yolov3FPN, self).__init__()
        # inplanes:[C3_inplanes,C4_inplanes,C5_inplanes]
        self.per_level_num_anchors = per_level_num_anchors

        P5_1_layers = []
        for i in range(5):
            P5_1_layers.append(
                ConvBnActBlock(inplanes[2],
                               inplanes[2] // 2,
                               kernel_size=1,
                               stride=1,
                               padding=0,
                               groups=1,
                               has_bn=True,
                               has_act=True,
                               act_type=act_type) if i %
                2 == 0 else ConvBnActBlock(inplanes[2] // 2,
                                           inplanes[2],
                                           kernel_size=3,
                                           stride=1,
                                           padding=1,
                                           groups=1,
                                           has_bn=True,
                                           has_act=True,
                                           act_type=act_type))
        self.P5_1 = nn.Sequential(*P5_1_layers)
        self.P5_2 = ConvBnActBlock(inplanes[2] // 2,
                                   inplanes[2],
                                   kernel_size=3,
                                   stride=1,
                                   padding=1,
                                   groups=1,
                                   has_bn=True,
                                   has_act=True,
                                   act_type=act_type)
        

        self.P5_up_conv = ConvBnActBlock(inplanes[2] // 2,
                                         inplanes[1] // 2,
                                         kernel_size=1,
                                         stride=1,
                                         padding=0,
                                         groups=1,
                                         has_bn=True,
                                         has_act=True,
                                         act_type=act_type)

        P4_1_layers = []
        for i in range(5):
            if i % 2 == 0:
                P4_1_layers.append(
                    ConvBnActBlock((inplanes[1] // 2) + inplanes[1],
                                   inplanes[1] // 2,
                                   kernel_size=1,
                                   stride=1,
                                   padding=0,
                                   groups=1,
                                   has_bn=True,
                                   has_act=True,
                                   act_type=act_type) if i ==
                    0 else ConvBnActBlock(inplanes[1],
                                          inplanes[1] // 2,
                                          kernel_size=1,
                                          stride=1,
                                          padding=0,
                                          groups=1,
                                          has_bn=True,
                                          has_act=True,
                                          act_type=act_type))
            else:
                P4_1_layers.append(
                    ConvBnActBlock(inplanes[1] // 2,
                                   inplanes[1],
                                   kernel_size=3,
                                   stride=1,
                                   padding=1,
                                   groups=1,
                                   has_bn=True,
                                   has_act=True,
                                   act_type=act_type))
        self.P4_1 = nn.Sequential(*P4_1_layers)
        self.P4_2 = ConvBnActBlock(inplanes[1] // 2,
                                   inplanes[1],
                                   kernel_size=3,
                                   stride=1,
                                   padding=1,
                                   groups=1,
                                   has_bn=True,
                                   has_act=True,
                                   act_type=act_type)

        self.P4_up_conv = ConvBnActBlock(inplanes[1] // 2,
                                         inplanes[0] // 2,
                                         kernel_size=1,
                                         stride=1,
                                         padding=0,
                                         groups=1,
                                         has_bn=True,
                                         has_act=True,
                                         act_type=act_type)

        P3_1_layers = []
        for i in range(5):
            if i % 2 == 0:
                P3_1_layers.append(
                    ConvBnActBlock((inplanes[0] // 2) + inplanes[0],
                                   inplanes[0] // 2,
                                   kernel_size=1,
                                   stride=1,
                                   padding=0,
                                   groups=1,
                                   has_bn=True,
                                   has_act=True,
                                   act_type=act_type) if i ==
                    0 else ConvBnActBlock(inplanes[0],
                                          inplanes[0] // 2,
                                          kernel_size=1,
                                          stride=1,
                                          padding=0,
                                          groups=1,
                                          has_bn=True,
                                          has_act=True,
                                          act_type=act_type))
            else:
                P3_1_layers.append(
                    ConvBnActBlock(inplanes[0] // 2,
                                   inplanes[0],
                                   kernel_size=3,
                                   stride=1,
                                   padding=1,
                                   groups=1,
                                   has_bn=True,
                                   has_act=True,
                                   act_type=act_type))
        self.P3_1 = nn.Sequential(*P3_1_layers)
        self.P3_2 = ConvBnActBlock(inplanes[0] // 2,
                                   inplanes[0],
                                   kernel_size=3,
                                   stride=1,
                                   padding=1,
                                   groups=1,
                                   has_bn=True,
                                   has_act=True,
                                   act_type=act_type)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        [C3, C4, C5] = inputs

        P5 = self.P5_1(C5)
        del C5

        C5_upsample = F.interpolate(self.P5_up_conv(P5),
                                    size=(C4.shape[2], C4.shape[3]),
                                    mode='bilinear',
                                    align_corners=True)
        C4 = torch.cat([C4, C5_upsample], axis=1)
        del C5_upsample

        P4 = self.P4_1(C4)
        del C4

        C4_upsample = F.interpolate(self.P4_up_conv(P4),
                                    size=(C3.shape[2], C3.shape[3]),
                                    mode='bilinear',
                                    align_corners=True)
        C3 = torch.cat([C3, C4_upsample], axis=1)
        del C4_upsample

        P3 = self.P3_1(C3)
        del C3

        P5 = self.P5_2(P5)

        P4 = self.P4_2(P4)

        P3 = self.P3_2(P3)

        return [P3, P4, P5]



# class Yolov3FPNHead(nn.Module):

#     def __init__(self,
#                  inplanes,
#                  per_level_num_anchors=3,
#                  num_classes=80,
#                  act_type='leakyrelu'):
#         super(Yolov3FPNHead, self).__init__()
#         # inplanes:[C3_inplanes,C4_inplanes,C5_inplanes]
#         self.per_level_num_anchors = per_level_num_anchors

#         P5_1_layers = []
#         for i in range(5):
#             P5_1_layers.append(
#                 ConvBnActBlock(inplanes[2],
#                                inplanes[2] // 2,
#                                kernel_size=1,
#                                stride=1,
#                                padding=0,
#                                groups=1,
#                                has_bn=True,
#                                has_act=True,
#                                act_type=act_type) if i %
#                 2 == 0 else ConvBnActBlock(inplanes[2] // 2,
#                                            inplanes[2],
#                                            kernel_size=3,
#                                            stride=1,
#                                            padding=1,
#                                            groups=1,
#                                            has_bn=True,
#                                            has_act=True,
#                                            act_type=act_type))
#         self.P5_1 = nn.Sequential(*P5_1_layers)
#         self.P5_2 = ConvBnActBlock(inplanes[2] // 2,
#                                    inplanes[2],
#                                    kernel_size=3,
#                                    stride=1,
#                                    padding=1,
#                                    groups=1,
#                                    has_bn=True,
#                                    has_act=True,
#                                    act_type=act_type)
#         self.P5_pred_conv = nn.Conv2d(inplanes[2],
#                                       per_level_num_anchors *
#                                       (1 + 4 + num_classes),
#                                       kernel_size=1,
#                                       stride=1,
#                                       padding=0,
#                                       groups=1,
#                                       bias=True)

#         self.P5_up_conv = ConvBnActBlock(inplanes[2] // 2,
#                                          inplanes[1] // 2,
#                                          kernel_size=1,
#                                          stride=1,
#                                          padding=0,
#                                          groups=1,
#                                          has_bn=True,
#                                          has_act=True,
#                                          act_type=act_type)

#         P4_1_layers = []
#         for i in range(5):
#             if i % 2 == 0:
#                 P4_1_layers.append(
#                     ConvBnActBlock((inplanes[1] // 2) + inplanes[1],
#                                    inplanes[1] // 2,
#                                    kernel_size=1,
#                                    stride=1,
#                                    padding=0,
#                                    groups=1,
#                                    has_bn=True,
#                                    has_act=True,
#                                    act_type=act_type) if i ==
#                     0 else ConvBnActBlock(inplanes[1],
#                                           inplanes[1] // 2,
#                                           kernel_size=1,
#                                           stride=1,
#                                           padding=0,
#                                           groups=1,
#                                           has_bn=True,
#                                           has_act=True,
#                                           act_type=act_type))
#             else:
#                 P4_1_layers.append(
#                     ConvBnActBlock(inplanes[1] // 2,
#                                    inplanes[1],
#                                    kernel_size=3,
#                                    stride=1,
#                                    padding=1,
#                                    groups=1,
#                                    has_bn=True,
#                                    has_act=True,
#                                    act_type=act_type))
#         self.P4_1 = nn.Sequential(*P4_1_layers)
#         self.P4_2 = ConvBnActBlock(inplanes[1] // 2,
#                                    inplanes[1],
#                                    kernel_size=3,
#                                    stride=1,
#                                    padding=1,
#                                    groups=1,
#                                    has_bn=True,
#                                    has_act=True,
#                                    act_type=act_type)
#         self.P4_pred_conv = nn.Conv2d(inplanes[1],
#                                       per_level_num_anchors *
#                                       (1 + 4 + num_classes),
#                                       kernel_size=1,
#                                       stride=1,
#                                       padding=0,
#                                       groups=1,
#                                       bias=True)

#         self.P4_up_conv = ConvBnActBlock(inplanes[1] // 2,
#                                          inplanes[0] // 2,
#                                          kernel_size=1,
#                                          stride=1,
#                                          padding=0,
#                                          groups=1,
#                                          has_bn=True,
#                                          has_act=True,
#                                          act_type=act_type)

#         P3_1_layers = []
#         for i in range(5):
#             if i % 2 == 0:
#                 P3_1_layers.append(
#                     ConvBnActBlock((inplanes[0] // 2) + inplanes[0],
#                                    inplanes[0] // 2,
#                                    kernel_size=1,
#                                    stride=1,
#                                    padding=0,
#                                    groups=1,
#                                    has_bn=True,
#                                    has_act=True,
#                                    act_type=act_type) if i ==
#                     0 else ConvBnActBlock(inplanes[0],
#                                           inplanes[0] // 2,
#                                           kernel_size=1,
#                                           stride=1,
#                                           padding=0,
#                                           groups=1,
#                                           has_bn=True,
#                                           has_act=True,
#                                           act_type=act_type))
#             else:
#                 P3_1_layers.append(
#                     ConvBnActBlock(inplanes[0] // 2,
#                                    inplanes[0],
#                                    kernel_size=3,
#                                    stride=1,
#                                    padding=1,
#                                    groups=1,
#                                    has_bn=True,
#                                    has_act=True,
#                                    act_type=act_type))
#         self.P3_1 = nn.Sequential(*P3_1_layers)
#         self.P3_2 = ConvBnActBlock(inplanes[0] // 2,
#                                    inplanes[0],
#                                    kernel_size=3,
#                                    stride=1,
#                                    padding=1,
#                                    groups=1,
#                                    has_bn=True,
#                                    has_act=True,
#                                    act_type=act_type)
#         self.P3_pred_conv = nn.Conv2d(inplanes[0],
#                                       per_level_num_anchors *
#                                       (1 + 4 + num_classes),
#                                       kernel_size=1,
#                                       stride=1,
#                                       padding=0,
#                                       groups=1,
#                                       bias=True)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, inputs):
#         [C3, C4, C5] = inputs

#         P5 = self.P5_1(C5)
#         del C5

#         C5_upsample = F.interpolate(self.P5_up_conv(P5),
#                                     size=(C4.shape[2], C4.shape[3]),
#                                     mode='bilinear',
#                                     align_corners=True)
#         C4 = torch.cat([C4, C5_upsample], axis=1)
#         del C5_upsample

#         P4 = self.P4_1(C4)
#         del C4

#         C4_upsample = F.interpolate(self.P4_up_conv(P4),
#                                     size=(C3.shape[2], C3.shape[3]),
#                                     mode='bilinear',
#                                     align_corners=True)
#         C3 = torch.cat([C3, C4_upsample], axis=1)
#         del C4_upsample

#         P3 = self.P3_1(C3)
#         del C3

#         P5 = self.P5_2(P5)
#         P5 = self.P5_pred_conv(P5)

#         P4 = self.P4_2(P4)
#         P4 = self.P4_pred_conv(P4)

#         P3 = self.P3_2(P3)
#         P3 = self.P3_pred_conv(P3)

#         # P3 shape:[B,255,H,W]->[B,H,W,255]->[B,H,W,3,85]
#         P3 = P3.permute(0, 2, 3, 1).contiguous()
#         P3 = P3.view(P3.shape[0], P3.shape[1], P3.shape[2],
#                      self.per_level_num_anchors, -1)
#         # P4 shape:[B,255,H,W]->[B,H,W,255]->[B,H,W,3,85]
#         P4 = P4.permute(0, 2, 3, 1).contiguous()
#         P4 = P4.view(P4.shape[0], P4.shape[1], P4.shape[2],
#                      self.per_level_num_anchors, -1)
#         # P5 shape:[B,255,H,W]->[B,H,W,255]->[B,H,W,3,85]
#         P5 = P5.permute(0, 2, 3, 1).contiguous()
#         P5 = P5.view(P5.shape[0], P5.shape[1], P5.shape[2],
#                      self.per_level_num_anchors, -1)

#         P3[:, :, :, :, 0:3] = torch.sigmoid(P3[:, :, :, :, 0:3])
#         P3[:, :, :, :, 5:] = torch.sigmoid(P3[..., 5:])
#         P4[:, :, :, :, 0:3] = torch.sigmoid(P4[:, :, :, :, 0:3])
#         P4[:, :, :, :, 5:] = torch.sigmoid(P4[..., 5:])
#         P5[:, :, :, :, 0:3] = torch.sigmoid(P5[:, :, :, :, 0:3])
#         P5[:, :, :, :, 5:] = torch.sigmoid(P5[..., 5:])

#         return [P3, P4, P5]

