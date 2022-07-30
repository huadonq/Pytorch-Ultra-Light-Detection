import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SPP(nn.Module):
    '''
    Spatial pyramid pooling layer used in YOLOv3-SPP
    '''

    def __init__(self, kernels=[5, 9, 13]):
        super(SPP, self).__init__()
        self.maxpool_layers = nn.ModuleList([
            nn.MaxPool2d(kernel_size=kernel, stride=1, padding=kernel // 2)
            for kernel in kernels
        ])

    def forward(self, x):
        out = torch.cat([x] + [layer(x) for layer in self.maxpool_layers],
                        dim=1)

        return out


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




class Yolov4TinyFPN(nn.Module):

    def __init__(self,
                 inplanes,
                 per_level_num_anchors=3,
                 num_classes=80,
                 act_type='leakyrelu'):
        super(Yolov4TinyFPN, self).__init__()
        # inplanes:[C4_inplanes,C5_inplanes]
        self.per_level_num_anchors = per_level_num_anchors

        self.P5_1 = ConvBnActBlock(inplanes[1],
                                   inplanes[1] // 2,
                                   kernel_size=1,
                                   stride=1,
                                   padding=0,
                                   groups=1,
                                   has_bn=True,
                                   has_act=True,
                                   act_type=act_type)
        self.P5_up_conv = ConvBnActBlock(inplanes[1] // 2,
                                         inplanes[0] // 2,
                                         kernel_size=1,
                                         stride=1,
                                         padding=0,
                                         groups=1,
                                         has_bn=True,
                                         has_act=True,
                                         act_type=act_type)

        self.P5_2 = ConvBnActBlock(inplanes[1] // 2,
                                   inplanes[1],
                                   kernel_size=3,
                                   stride=1,
                                   padding=1,
                                   groups=1,
                                   has_bn=True,
                                   has_act=True,
                                   act_type=act_type)
        

        self.P4_1 = ConvBnActBlock(int(inplanes[0] + inplanes[0] // 2),
                                   inplanes[0],
                                   kernel_size=3,
                                   stride=1,
                                   padding=1,
                                   groups=1,
                                   has_bn=True,
                                   has_act=True,
                                   act_type=act_type)
        

    def forward(self, inputs):
        [C4, C5] = inputs

        P5 = self.P5_1(C5)

        del C5

        P5_out = self.P5_2(P5)

        P5_upsample = F.interpolate(self.P5_up_conv(P5),
                                    size=(C4.shape[2], C4.shape[3]),
                                    mode='bilinear',
                                    align_corners=True)
        P4 = torch.cat([C4, P5_upsample], dim=1)

        del C4, P5, P5_upsample

        P4 = self.P4_1(P4)



        return [P4, P5_out]

class Yolov4FPN(nn.Module):

    def __init__(self,
                 inplanes,
                 per_level_num_anchors=3,
                 num_classes=80,
                 act_type='leakyrelu'):
        super(Yolov4FPN, self).__init__()
        # inplanes:[C3_inplanes,C4_inplanes,C5_inplanes]
        self.per_level_num_anchors = per_level_num_anchors

        p5_block1 = nn.Sequential(*[
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
                                       act_type=act_type) for i in range(3)
        ])
        p5_spp_block = SPP(kernels=(5, 9, 13))
        p5_block2 = nn.Sequential(
            ConvBnActBlock(inplanes[2] * 2,
                           inplanes[2] // 2,
                           kernel_size=1,
                           stride=1,
                           padding=0,
                           groups=1,
                           has_bn=True,
                           has_act=True,
                           act_type=act_type),
            ConvBnActBlock(inplanes[2] // 2,
                           inplanes[2],
                           kernel_size=3,
                           stride=1,
                           padding=1,
                           groups=1,
                           has_bn=True,
                           has_act=True,
                           act_type=act_type),
            ConvBnActBlock(inplanes[2],
                           inplanes[2] // 2,
                           kernel_size=1,
                           stride=1,
                           padding=0,
                           groups=1,
                           has_bn=True,
                           has_act=True,
                           act_type=act_type))
        self.P5_1 = nn.Sequential(p5_block1, p5_spp_block, p5_block2)
        self.P5_up_conv = ConvBnActBlock(inplanes[2] // 2,
                                         inplanes[1] // 2,
                                         kernel_size=1,
                                         stride=1,
                                         padding=0,
                                         groups=1,
                                         has_bn=True,
                                         has_act=True,
                                         act_type=act_type)
        self.P4_cat_conv = ConvBnActBlock(inplanes[1],
                                          inplanes[1] // 2,
                                          kernel_size=1,
                                          stride=1,
                                          padding=0,
                                          groups=1,
                                          has_bn=True,
                                          has_act=True,
                                          act_type=act_type)
        self.P4_1 = nn.Sequential(*[
            ConvBnActBlock(inplanes[1],
                           inplanes[1] // 2,
                           kernel_size=1,
                           stride=1,
                           padding=0,
                           groups=1,
                           has_bn=True,
                           has_act=True,
                           act_type=act_type) if i %
            2 == 0 else ConvBnActBlock(inplanes[1] // 2,
                                       inplanes[1],
                                       kernel_size=3,
                                       stride=1,
                                       padding=1,
                                       groups=1,
                                       has_bn=True,
                                       has_act=True,
                                       act_type=act_type) for i in range(5)
        ])
        self.P4_up_conv = ConvBnActBlock(inplanes[1] // 2,
                                         inplanes[0] // 2,
                                         kernel_size=1,
                                         stride=1,
                                         padding=0,
                                         groups=1,
                                         has_bn=True,
                                         has_act=True,
                                         act_type=act_type)
        self.P3_cat_conv = ConvBnActBlock(inplanes[0],
                                          inplanes[0] // 2,
                                          kernel_size=1,
                                          stride=1,
                                          padding=0,
                                          groups=1,
                                          has_bn=True,
                                          has_act=True,
                                          act_type=act_type)
        self.P3_1 = nn.Sequential(*[
            ConvBnActBlock(inplanes[0],
                           inplanes[0] // 2,
                           kernel_size=1,
                           stride=1,
                           padding=0,
                           groups=1,
                           has_bn=True,
                           has_act=True,
                           act_type=act_type) if i %
            2 == 0 else ConvBnActBlock(inplanes[0] // 2,
                                       inplanes[0],
                                       kernel_size=3,
                                       stride=1,
                                       padding=1,
                                       groups=1,
                                       has_bn=True,
                                       has_act=True,
                                       act_type=act_type) for i in range(5)
        ])
        self.P3_out_conv = ConvBnActBlock(inplanes[0] // 2,
                                          inplanes[0],
                                          kernel_size=3,
                                          stride=1,
                                          padding=1,
                                          groups=1,
                                          has_bn=True,
                                          has_act=True,
                                          act_type=act_type)
        self.P3_down_conv = ConvBnActBlock(inplanes[0] // 2,
                                           inplanes[1] // 2,
                                           kernel_size=3,
                                           stride=2,
                                           padding=1,
                                           groups=1,
                                           has_bn=True,
                                           has_act=True,
                                           act_type=act_type)
        self.P4_2 = nn.Sequential(*[
            ConvBnActBlock(inplanes[1],
                           inplanes[1] // 2,
                           kernel_size=1,
                           stride=1,
                           padding=0,
                           groups=1,
                           has_bn=True,
                           has_act=True,
                           act_type=act_type) if i %
            2 == 0 else ConvBnActBlock(inplanes[1] // 2,
                                       inplanes[1],
                                       kernel_size=3,
                                       stride=1,
                                       padding=1,
                                       groups=1,
                                       has_bn=True,
                                       has_act=True,
                                       act_type=act_type) for i in range(5)
        ])
        self.P4_out_conv = ConvBnActBlock(inplanes[1] // 2,
                                          inplanes[1],
                                          kernel_size=3,
                                          stride=1,
                                          padding=1,
                                          groups=1,
                                          has_bn=True,
                                          has_act=True,
                                          act_type=act_type)
        self.P4_down_conv = ConvBnActBlock(inplanes[1] // 2,
                                           inplanes[2] // 2,
                                           kernel_size=3,
                                           stride=2,
                                           padding=1,
                                           groups=1,
                                           has_bn=True,
                                           has_act=True,
                                           act_type=act_type)
        self.P5_2 = nn.Sequential(*[
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
                                       act_type=act_type) for i in range(5)
        ])
        self.P5_out_conv = ConvBnActBlock(inplanes[2] // 2,
                                          inplanes[2],
                                          kernel_size=3,
                                          stride=1,
                                          padding=1,
                                          groups=1,
                                          has_bn=True,
                                          has_act=True,
                                          act_type=act_type)


    def forward(self, inputs):
        [C3, C4, C5] = inputs

        P5 = self.P5_1(C5)
        del C5

        P5_upsample = F.interpolate(self.P5_up_conv(P5),
                                    size=(C4.shape[2], C4.shape[3]),
                                    mode='bilinear',
                                    align_corners=True)
        C4 = torch.cat([self.P4_cat_conv(C4), P5_upsample], dim=1)
        del P5_upsample

        P4 = self.P4_1(C4)
        del C4

        P4_upsample = F.interpolate(self.P4_up_conv(P4),
                                    size=(C3.shape[2], C3.shape[3]),
                                    mode='bilinear',
                                    align_corners=True)
        C3 = torch.cat([self.P3_cat_conv(C3), P4_upsample], dim=1)
        del P4_upsample

        P3 = self.P3_1(C3)
        del C3

        P3_out = self.P3_out_conv(P3)

        P4 = torch.cat([P4, self.P3_down_conv(P3)], dim=1)
        del P3
        P4 = self.P4_2(P4)

        P4_out = self.P4_out_conv(P4)

        P5 = torch.cat([P5, self.P4_down_conv(P4)], dim=1)
        del P4
        P5 = self.P5_2(P5)

        P5_out = self.P5_out_conv(P5)
        del P5



        return [P3_out, P4_out, P5_out]

# class Yolov4TinyFPNHead(nn.Module):

#     def __init__(self,
#                  inplanes,
#                  per_level_num_anchors=3,
#                  num_classes=80,
#                  act_type='leakyrelu'):
#         super(Yolov4TinyFPNHead, self).__init__()
#         # inplanes:[C4_inplanes,C5_inplanes]
#         self.per_level_num_anchors = per_level_num_anchors

#         self.P5_1 = ConvBnActBlock(inplanes[1],
#                                    inplanes[1] // 2,
#                                    kernel_size=1,
#                                    stride=1,
#                                    padding=0,
#                                    groups=1,
#                                    has_bn=True,
#                                    has_act=True,
#                                    act_type=act_type)
#         self.P5_up_conv = ConvBnActBlock(inplanes[1] // 2,
#                                          inplanes[0] // 2,
#                                          kernel_size=1,
#                                          stride=1,
#                                          padding=0,
#                                          groups=1,
#                                          has_bn=True,
#                                          has_act=True,
#                                          act_type=act_type)

#         self.P5_2 = ConvBnActBlock(inplanes[1] // 2,
#                                    inplanes[1],
#                                    kernel_size=3,
#                                    stride=1,
#                                    padding=1,
#                                    groups=1,
#                                    has_bn=True,
#                                    has_act=True,
#                                    act_type=act_type)
#         self.P5_pred_conv = nn.Conv2d(inplanes[1],
#                                       per_level_num_anchors *
#                                       (1 + 4 + num_classes),
#                                       kernel_size=1,
#                                       stride=1,
#                                       padding=0,
#                                       bias=True)

#         self.P4_1 = ConvBnActBlock(int(inplanes[0] + inplanes[0] // 2),
#                                    inplanes[0],
#                                    kernel_size=3,
#                                    stride=1,
#                                    padding=1,
#                                    groups=1,
#                                    has_bn=True,
#                                    has_act=True,
#                                    act_type=act_type)
#         self.P4_pred_conv = nn.Conv2d(inplanes[0],
#                                       per_level_num_anchors *
#                                       (1 + 4 + num_classes),
#                                       kernel_size=1,
#                                       stride=1,
#                                       padding=0,
#                                       bias=True)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, inputs):
#         [C4, C5] = inputs

#         P5 = self.P5_1(C5)

#         del C5

#         P5_out = self.P5_2(P5)
#         P5_out = self.P5_pred_conv(P5_out)

#         P5_upsample = F.interpolate(self.P5_up_conv(P5),
#                                     size=(C4.shape[2], C4.shape[3]),
#                                     mode='bilinear',
#                                     align_corners=True)
#         P4 = torch.cat([C4, P5_upsample], dim=1)

#         del C4, P5, P5_upsample

#         P4 = self.P4_1(P4)
#         P4_out = self.P4_pred_conv(P4)

#         del P4

#         # P4_out shape:[B,255,H,W]->[B,H,W,255]->[B,H,W,3,85]
#         P4_out = P4_out.permute(0, 2, 3, 1).contiguous()
#         P4_out = P4_out.view(P4_out.shape[0], P4_out.shape[1], P4_out.shape[2],
#                              self.per_level_num_anchors, -1)
#         # P5_out shape:[B,255,H,W]->[B,H,W,255]->[B,H,W,3,85]
#         P5_out = P5_out.permute(0, 2, 3, 1).contiguous()
#         P5_out = P5_out.view(P5_out.shape[0], P5_out.shape[1], P5_out.shape[2],
#                              self.per_level_num_anchors, -1)

#         P4_out[:, :, :, :, 0:3] = torch.sigmoid(P4_out[:, :, :, :, 0:3])
#         P4_out[:, :, :, :, 5:] = torch.sigmoid(P4_out[..., 5:])
#         P5_out[:, :, :, :, 0:3] = torch.sigmoid(P5_out[:, :, :, :, 0:3])
#         P5_out[:, :, :, :, 5:] = torch.sigmoid(P5_out[..., 5:])

#         return P4_out, P5_out





# class Yolov4FPNHead(nn.Module):

#     def __init__(self,
#                  inplanes,
#                  per_level_num_anchors=3,
#                  num_classes=80,
#                  act_type='leakyrelu'):
#         super(Yolov4FPNHead, self).__init__()
#         # inplanes:[C3_inplanes,C4_inplanes,C5_inplanes]
#         self.per_level_num_anchors = per_level_num_anchors

#         p5_block1 = nn.Sequential(*[
#             ConvBnActBlock(inplanes[2],
#                            inplanes[2] // 2,
#                            kernel_size=1,
#                            stride=1,
#                            padding=0,
#                            groups=1,
#                            has_bn=True,
#                            has_act=True,
#                            act_type=act_type) if i %
#             2 == 0 else ConvBnActBlock(inplanes[2] // 2,
#                                        inplanes[2],
#                                        kernel_size=3,
#                                        stride=1,
#                                        padding=1,
#                                        groups=1,
#                                        has_bn=True,
#                                        has_act=True,
#                                        act_type=act_type) for i in range(3)
#         ])
#         p5_spp_block = SPP(kernels=(5, 9, 13))
#         p5_block2 = nn.Sequential(
#             ConvBnActBlock(inplanes[2] * 2,
#                            inplanes[2] // 2,
#                            kernel_size=1,
#                            stride=1,
#                            padding=0,
#                            groups=1,
#                            has_bn=True,
#                            has_act=True,
#                            act_type=act_type),
#             ConvBnActBlock(inplanes[2] // 2,
#                            inplanes[2],
#                            kernel_size=3,
#                            stride=1,
#                            padding=1,
#                            groups=1,
#                            has_bn=True,
#                            has_act=True,
#                            act_type=act_type),
#             ConvBnActBlock(inplanes[2],
#                            inplanes[2] // 2,
#                            kernel_size=1,
#                            stride=1,
#                            padding=0,
#                            groups=1,
#                            has_bn=True,
#                            has_act=True,
#                            act_type=act_type))
#         self.P5_1 = nn.Sequential(p5_block1, p5_spp_block, p5_block2)
#         self.P5_up_conv = ConvBnActBlock(inplanes[2] // 2,
#                                          inplanes[1] // 2,
#                                          kernel_size=1,
#                                          stride=1,
#                                          padding=0,
#                                          groups=1,
#                                          has_bn=True,
#                                          has_act=True,
#                                          act_type=act_type)
#         self.P4_cat_conv = ConvBnActBlock(inplanes[1],
#                                           inplanes[1] // 2,
#                                           kernel_size=1,
#                                           stride=1,
#                                           padding=0,
#                                           groups=1,
#                                           has_bn=True,
#                                           has_act=True,
#                                           act_type=act_type)
#         self.P4_1 = nn.Sequential(*[
#             ConvBnActBlock(inplanes[1],
#                            inplanes[1] // 2,
#                            kernel_size=1,
#                            stride=1,
#                            padding=0,
#                            groups=1,
#                            has_bn=True,
#                            has_act=True,
#                            act_type=act_type) if i %
#             2 == 0 else ConvBnActBlock(inplanes[1] // 2,
#                                        inplanes[1],
#                                        kernel_size=3,
#                                        stride=1,
#                                        padding=1,
#                                        groups=1,
#                                        has_bn=True,
#                                        has_act=True,
#                                        act_type=act_type) for i in range(5)
#         ])
#         self.P4_up_conv = ConvBnActBlock(inplanes[1] // 2,
#                                          inplanes[0] // 2,
#                                          kernel_size=1,
#                                          stride=1,
#                                          padding=0,
#                                          groups=1,
#                                          has_bn=True,
#                                          has_act=True,
#                                          act_type=act_type)
#         self.P3_cat_conv = ConvBnActBlock(inplanes[0],
#                                           inplanes[0] // 2,
#                                           kernel_size=1,
#                                           stride=1,
#                                           padding=0,
#                                           groups=1,
#                                           has_bn=True,
#                                           has_act=True,
#                                           act_type=act_type)
#         self.P3_1 = nn.Sequential(*[
#             ConvBnActBlock(inplanes[0],
#                            inplanes[0] // 2,
#                            kernel_size=1,
#                            stride=1,
#                            padding=0,
#                            groups=1,
#                            has_bn=True,
#                            has_act=True,
#                            act_type=act_type) if i %
#             2 == 0 else ConvBnActBlock(inplanes[0] // 2,
#                                        inplanes[0],
#                                        kernel_size=3,
#                                        stride=1,
#                                        padding=1,
#                                        groups=1,
#                                        has_bn=True,
#                                        has_act=True,
#                                        act_type=act_type) for i in range(5)
#         ])
#         self.P3_out_conv = ConvBnActBlock(inplanes[0] // 2,
#                                           inplanes[0],
#                                           kernel_size=3,
#                                           stride=1,
#                                           padding=1,
#                                           groups=1,
#                                           has_bn=True,
#                                           has_act=True,
#                                           act_type=act_type)
#         self.P3_down_conv = ConvBnActBlock(inplanes[0] // 2,
#                                            inplanes[1] // 2,
#                                            kernel_size=3,
#                                            stride=2,
#                                            padding=1,
#                                            groups=1,
#                                            has_bn=True,
#                                            has_act=True,
#                                            act_type=act_type)
#         self.P4_2 = nn.Sequential(*[
#             ConvBnActBlock(inplanes[1],
#                            inplanes[1] // 2,
#                            kernel_size=1,
#                            stride=1,
#                            padding=0,
#                            groups=1,
#                            has_bn=True,
#                            has_act=True,
#                            act_type=act_type) if i %
#             2 == 0 else ConvBnActBlock(inplanes[1] // 2,
#                                        inplanes[1],
#                                        kernel_size=3,
#                                        stride=1,
#                                        padding=1,
#                                        groups=1,
#                                        has_bn=True,
#                                        has_act=True,
#                                        act_type=act_type) for i in range(5)
#         ])
#         self.P4_out_conv = ConvBnActBlock(inplanes[1] // 2,
#                                           inplanes[1],
#                                           kernel_size=3,
#                                           stride=1,
#                                           padding=1,
#                                           groups=1,
#                                           has_bn=True,
#                                           has_act=True,
#                                           act_type=act_type)
#         self.P4_down_conv = ConvBnActBlock(inplanes[1] // 2,
#                                            inplanes[2] // 2,
#                                            kernel_size=3,
#                                            stride=2,
#                                            padding=1,
#                                            groups=1,
#                                            has_bn=True,
#                                            has_act=True,
#                                            act_type=act_type)
#         self.P5_2 = nn.Sequential(*[
#             ConvBnActBlock(inplanes[2],
#                            inplanes[2] // 2,
#                            kernel_size=1,
#                            stride=1,
#                            padding=0,
#                            groups=1,
#                            has_bn=True,
#                            has_act=True,
#                            act_type=act_type) if i %
#             2 == 0 else ConvBnActBlock(inplanes[2] // 2,
#                                        inplanes[2],
#                                        kernel_size=3,
#                                        stride=1,
#                                        padding=1,
#                                        groups=1,
#                                        has_bn=True,
#                                        has_act=True,
#                                        act_type=act_type) for i in range(5)
#         ])
#         self.P5_out_conv = ConvBnActBlock(inplanes[2] // 2,
#                                           inplanes[2],
#                                           kernel_size=3,
#                                           stride=1,
#                                           padding=1,
#                                           groups=1,
#                                           has_bn=True,
#                                           has_act=True,
#                                           act_type=act_type)
#         self.P5_pred_conv = nn.Conv2d(inplanes[2],
#                                       per_level_num_anchors *
#                                       (1 + 4 + num_classes),
#                                       kernel_size=1,
#                                       stride=1,
#                                       padding=0,
#                                       bias=True)
#         self.P4_pred_conv = nn.Conv2d(inplanes[1],
#                                       per_level_num_anchors *
#                                       (1 + 4 + num_classes),
#                                       kernel_size=1,
#                                       stride=1,
#                                       padding=0,
#                                       bias=True)
#         self.P3_pred_conv = nn.Conv2d(inplanes[0],
#                                       per_level_num_anchors *
#                                       (1 + 4 + num_classes),
#                                       kernel_size=1,
#                                       stride=1,
#                                       padding=0,
#                                       bias=True)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, inputs):
#         [C3, C4, C5] = inputs

#         P5 = self.P5_1(C5)
#         del C5

#         P5_upsample = F.interpolate(self.P5_up_conv(P5),
#                                     size=(C4.shape[2], C4.shape[3]),
#                                     mode='bilinear',
#                                     align_corners=True)
#         C4 = torch.cat([self.P4_cat_conv(C4), P5_upsample], dim=1)
#         del P5_upsample

#         P4 = self.P4_1(C4)
#         del C4

#         P4_upsample = F.interpolate(self.P4_up_conv(P4),
#                                     size=(C3.shape[2], C3.shape[3]),
#                                     mode='bilinear',
#                                     align_corners=True)
#         C3 = torch.cat([self.P3_cat_conv(C3), P4_upsample], dim=1)
#         del P4_upsample

#         P3 = self.P3_1(C3)
#         del C3

#         P3_out = self.P3_out_conv(P3)
#         P3_out = self.P3_pred_conv(P3_out)

#         P4 = torch.cat([P4, self.P3_down_conv(P3)], dim=1)
#         del P3
#         P4 = self.P4_2(P4)

#         P4_out = self.P4_out_conv(P4)
#         P4_out = self.P4_pred_conv(P4_out)

#         P5 = torch.cat([P5, self.P4_down_conv(P4)], dim=1)
#         del P4
#         P5 = self.P5_2(P5)

#         P5_out = self.P5_out_conv(P5)
#         P5_out = self.P5_pred_conv(P5_out)
#         del P5

#         # P3_out shape:[B,255,H,W]->[B,H,W,255]->[B,H,W,3,85]
#         P3_out = P3_out.permute(0, 2, 3, 1).contiguous()
#         P3_out = P3_out.view(P3_out.shape[0], P3_out.shape[1], P3_out.shape[2],
#                              self.per_level_num_anchors, -1)
#         # P4_out shape:[B,255,H,W]->[B,H,W,255]->[B,H,W,3,85]
#         P4_out = P4_out.permute(0, 2, 3, 1).contiguous()
#         P4_out = P4_out.view(P4_out.shape[0], P4_out.shape[1], P4_out.shape[2],
#                              self.per_level_num_anchors, -1)
#         # P5_out shape:[B,255,H,W]->[B,H,W,255]->[B,H,W,3,85]
#         P5_out = P5_out.permute(0, 2, 3, 1).contiguous()
#         P5_out = P5_out.view(P5_out.shape[0], P5_out.shape[1], P5_out.shape[2],
#                              self.per_level_num_anchors, -1)

#         P3_out[:, :, :, :, 0:3] = torch.sigmoid(P3_out[:, :, :, :, 0:3])
#         P3_out[:, :, :, :, 5:] = torch.sigmoid(P3_out[..., 5:])
#         P4_out[:, :, :, :, 0:3] = torch.sigmoid(P4_out[:, :, :, :, 0:3])
#         P4_out[:, :, :, :, 5:] = torch.sigmoid(P4_out[..., 5:])
#         P5_out[:, :, :, :, 0:3] = torch.sigmoid(P5_out[:, :, :, :, 0:3])
#         P5_out[:, :, :, :, 5:] = torch.sigmoid(P5_out[..., 5:])

#         return [P3_out, P4_out, P5_out]

