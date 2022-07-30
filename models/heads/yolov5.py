import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Yolov5Head(nn.Module):

    def __init__(self,
                 inplanes,
                 per_level_num_anchors=3,
                 num_classes=80,
                 act_type='leakyrelu'):
        super(Yolov5Head, self).__init__()
        # inplanes:[C4_inplanes,C5_inplanes]
        self.per_level_num_anchors = per_level_num_anchors

        self.P3_pred_conv = nn.Conv2d(inplanes[0],
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
        [P3, P4, P5] = inputs

        P3_out = self.P3_pred_conv(P3)
        del P3
        P4_out = self.P4_pred_conv(P4)
        del P4
        P5_out = self.P5_pred_conv(P5)
        del P5

        # P3_out shape:[B,255,H,W]->[B,H,W,255]->[B,H,W,3,85]
        P3_out = P3_out.permute(0, 2, 3, 1).contiguous()
        P3_out = P3_out.view(P3_out.shape[0], P3_out.shape[1], P3_out.shape[2],
                             self.per_level_num_anchors, -1).contiguous()
        # P4_out shape:[B,255,H,W]->[B,H,W,255]->[B,H,W,3,85]
        P4_out = P4_out.permute(0, 2, 3, 1).contiguous()
        P4_out = P4_out.view(P4_out.shape[0], P4_out.shape[1], P4_out.shape[2],
                             self.per_level_num_anchors, -1).contiguous()
        # P5_out shape:[B,255,H,W]->[B,H,W,255]->[B,H,W,3,85]
        P5_out = P5_out.permute(0, 2, 3, 1).contiguous()
        P5_out = P5_out.view(P5_out.shape[0], P5_out.shape[1], P5_out.shape[2],
                             self.per_level_num_anchors, -1).contiguous()

        P3_out = self.sigmoid(P3_out)
        P4_out = self.sigmoid(P4_out)
        P5_out = self.sigmoid(P5_out)

        return [P3_out, P4_out, P5_out]