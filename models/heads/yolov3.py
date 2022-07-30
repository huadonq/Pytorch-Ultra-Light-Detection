import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Yolov3TinyHead(nn.Module):

    def __init__(self,
                 inplanes,
                 per_level_num_anchors=3,
                 num_classes=80,
                 act_type='leakyrelu'):
        super(Yolov3TinyHead, self).__init__()
        # inplanes:[C4_inplanes,C5_inplanes]
        self.per_level_num_anchors = per_level_num_anchors

        self.P5_pred_conv = nn.Conv2d(512,
                                      per_level_num_anchors *
                                      (1 + 4 + num_classes),
                                      kernel_size=1,
                                      stride=1,
                                      padding=0,
                                      groups=1,
                                      bias=True)
                            
        self.P4_pred_conv = nn.Conv2d(256,
                                      per_level_num_anchors *
                                      (1 + 4 + num_classes),
                                      kernel_size=1,
                                      stride=1,
                                      padding=0,
                                      groups=1,
                                      bias=True)

        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):

        [P4, P5] = inputs
        P5 = self.P5_pred_conv(P5)
        P4 = self.P4_pred_conv(P4)

        P4 = P4.permute(0, 2, 3, 1).contiguous()
        P4 = P4.view(P4.shape[0], P4.shape[1], P4.shape[2],
                     self.per_level_num_anchors, -1)
        # P5 shape:[B,255,H,W]->[B,H,W,255]->[B,H,W,3,85]

        P5 = P5.permute(0, 2, 3, 1).contiguous()
        P5 = P5.view(P5.shape[0], P5.shape[1], P5.shape[2],
                     self.per_level_num_anchors, -1)

        P4[:, :, :, :, 0:3] = torch.sigmoid(P4[:, :, :, :, 0:3])
        P4[:, :, :, :, 5:] = torch.sigmoid(P4[..., 5:])
        P5[:, :, :, :, 0:3] = torch.sigmoid(P5[:, :, :, :, 0:3])
        P5[:, :, :, :, 5:] = torch.sigmoid(P5[..., 5:])

        return [P4, P5]


class Yolov3Head(nn.Module):

    def __init__(self,
                 inplanes,
                 per_level_num_anchors=3,
                 num_classes=80,
                 act_type='leakyrelu'):
        super(Yolov3Head, self).__init__()
        # inplanes:[C4_inplanes,C5_inplanes]
        self.per_level_num_anchors = per_level_num_anchors

        self.P5_pred_conv = nn.Conv2d(inplanes[2],
                                      per_level_num_anchors *
                                      (1 + 4 + num_classes),
                                      kernel_size=1,
                                      stride=1,
                                      padding=0,
                                      groups=1,
                                      bias=True)
                            
        self.P4_pred_conv = nn.Conv2d(inplanes[1],
                                      per_level_num_anchors *
                                      (1 + 4 + num_classes),
                                      kernel_size=1,
                                      stride=1,
                                      padding=0,
                                      groups=1,
                                      bias=True)

        self.P3_pred_conv = nn.Conv2d(inplanes[0],
                                      per_level_num_anchors *
                                      (1 + 4 + num_classes),
                                      kernel_size=1,
                                      stride=1,
                                      padding=0,
                                      groups=1,
                                      bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):

        [P3, P4, P5] = inputs

        P5 = self.P5_pred_conv(P5)
        P4 = self.P4_pred_conv(P4)
        P3 = self.P3_pred_conv(P3)

        # P3 shape:[B,255,H,W]->[B,H,W,255]->[B,H,W,3,85]
        P3 = P3.permute(0, 2, 3, 1).contiguous()
        P3 = P3.view(P3.shape[0], P3.shape[1], P3.shape[2],
                     self.per_level_num_anchors, -1)
        # P4 shape:[B,255,H,W]->[B,H,W,255]->[B,H,W,3,85]
        P4 = P4.permute(0, 2, 3, 1).contiguous()
        P4 = P4.view(P4.shape[0], P4.shape[1], P4.shape[2],
                     self.per_level_num_anchors, -1)
        # P5 shape:[B,255,H,W]->[B,H,W,255]->[B,H,W,3,85]
        P5 = P5.permute(0, 2, 3, 1).contiguous()
        P5 = P5.view(P5.shape[0], P5.shape[1], P5.shape[2],
                     self.per_level_num_anchors, -1)

        P3[:, :, :, :, 0:3] = torch.sigmoid(P3[:, :, :, :, 0:3])
        P3[:, :, :, :, 5:] = torch.sigmoid(P3[..., 5:])
        P4[:, :, :, :, 0:3] = torch.sigmoid(P4[:, :, :, :, 0:3])
        P4[:, :, :, :, 5:] = torch.sigmoid(P4[..., 5:])
        P5[:, :, :, :, 0:3] = torch.sigmoid(P5[:, :, :, :, 0:3])
        P5[:, :, :, :, 5:] = torch.sigmoid(P5[..., 5:])

        return [P3, P4, P5]