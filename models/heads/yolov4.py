import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Yolov4Head(nn.Module):

    def __init__(self,
                 inplanes,
                 per_level_num_anchors=3,
                 num_classes=80,
                 act_type='leakyrelu'):
        super(Yolov4Head, self).__init__()
        # inplanes:[C4_inplanes,C5_inplanes]
        self.per_level_num_anchors = per_level_num_anchors

        self.P5_pred_conv = nn.Conv2d(inplanes[2],
                                      per_level_num_anchors *
                                      (1 + 4 + num_classes),
                                      kernel_size=1,
                                      stride=1,
                                      padding=0,
                                      bias=True)
        self.P4_pred_conv = nn.Conv2d(inplanes[1],
                                      per_level_num_anchors *
                                      (1 + 4 + num_classes),
                                      kernel_size=1,
                                      stride=1,
                                      padding=0,
                                      bias=True)
        self.P3_pred_conv = nn.Conv2d(inplanes[0],
                                      per_level_num_anchors *
                                      (1 + 4 + num_classes),
                                      kernel_size=1,
                                      stride=1,
                                      padding=0,
                                      bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        [P3, P4, P5] = inputs

        P5_out = self.P5_pred_conv(P5)
        P4_out = self.P4_pred_conv(P4)
        P3_out = self.P3_pred_conv(P3)

        # P3_out shape:[B,255,H,W]->[B,H,W,255]->[B,H,W,3,85]
        P3_out = P3_out.permute(0, 2, 3, 1).contiguous()
        P3_out = P3_out.view(P3_out.shape[0], P3_out.shape[1], P3_out.shape[2],
                             self.per_level_num_anchors, -1)
        # P4_out shape:[B,255,H,W]->[B,H,W,255]->[B,H,W,3,85]
        P4_out = P4_out.permute(0, 2, 3, 1).contiguous()
        P4_out = P4_out.view(P4_out.shape[0], P4_out.shape[1], P4_out.shape[2],
                             self.per_level_num_anchors, -1)
        # P5_out shape:[B,255,H,W]->[B,H,W,255]->[B,H,W,3,85]
        P5_out = P5_out.permute(0, 2, 3, 1).contiguous()
        P5_out = P5_out.view(P5_out.shape[0], P5_out.shape[1], P5_out.shape[2],
                             self.per_level_num_anchors, -1)

        P3_out[:, :, :, :, 0:3] = torch.sigmoid(P3_out[:, :, :, :, 0:3])
        P3_out[:, :, :, :, 5:] = torch.sigmoid(P3_out[..., 5:])
        P4_out[:, :, :, :, 0:3] = torch.sigmoid(P4_out[:, :, :, :, 0:3])
        P4_out[:, :, :, :, 5:] = torch.sigmoid(P4_out[..., 5:])
        P5_out[:, :, :, :, 0:3] = torch.sigmoid(P5_out[:, :, :, :, 0:3])
        P5_out[:, :, :, :, 5:] = torch.sigmoid(P5_out[..., 5:])


        return [P3_out, P4_out, P5_out]

class Yolov4TinyHead(nn.Module):

    def __init__(self,
                 inplanes,
                 per_level_num_anchors=3,
                 num_classes=80,
                 act_type='leakyrelu'):
        super(Yolov4TinyHead, self).__init__()
        # inplanes:[C4_inplanes,C5_inplanes]
        self.per_level_num_anchors = per_level_num_anchors

        self.P5_pred_conv = nn.Conv2d(inplanes[1],
                                      per_level_num_anchors *
                                      (1 + 4 + num_classes),
                                      kernel_size=1,
                                      stride=1,
                                      padding=0,
                                      bias=True)

        self.P4_pred_conv = nn.Conv2d(inplanes[0],
                                      per_level_num_anchors *
                                      (1 + 4 + num_classes),
                                      kernel_size=1,
                                      stride=1,
                                      padding=0,
                                      bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        
        [P4, P5] = inputs
        P5_out = self.P5_pred_conv(P5)
        P4_out = self.P4_pred_conv(P4)

        P4_out = P4_out.permute(0, 2, 3, 1).contiguous()
        P4_out = P4_out.view(P4_out.shape[0], P4_out.shape[1], P4_out.shape[2],
                             self.per_level_num_anchors, -1)
        # P5_out shape:[B,255,H,W]->[B,H,W,255]->[B,H,W,3,85]
        P5_out = P5_out.permute(0, 2, 3, 1).contiguous()
        P5_out = P5_out.view(P5_out.shape[0], P5_out.shape[1], P5_out.shape[2],
                             self.per_level_num_anchors, -1)

        P4_out[:, :, :, :, 0:3] = torch.sigmoid(P4_out[:, :, :, :, 0:3])
        P4_out[:, :, :, :, 5:] = torch.sigmoid(P4_out[..., 5:])
        P5_out[:, :, :, :, 0:3] = torch.sigmoid(P5_out[:, :, :, :, 0:3])
        P5_out[:, :, :, :, 5:] = torch.sigmoid(P5_out[..., 5:])

        return [P4_out, P5_out]