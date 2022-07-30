import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# class RetinaFPN(nn.Module):

#     def __init__(self, inplanes, planes, use_p5=False):
#         super(RetinaFPN, self).__init__()
#         # inplanes:[C3_inplanes,C4_inplanes,C5_inplanes]
#         self.use_p5 = use_p5
#         self.P3_1 = nn.Conv2d(inplanes[0],
#                               planes,
#                               kernel_size=1,
#                               stride=1,
#                               padding=0)
#         self.P3_2 = nn.Conv2d(planes,
#                               planes,
#                               kernel_size=3,
#                               stride=1,
#                               padding=1)
#         self.P4_1 = nn.Conv2d(inplanes[1],
#                               planes,
#                               kernel_size=1,
#                               stride=1,
#                               padding=0)
#         self.P4_2 = nn.Conv2d(planes,
#                               planes,
#                               kernel_size=3,
#                               stride=1,
#                               padding=1)
#         self.P5_1 = nn.Conv2d(inplanes[2],
#                               planes,
#                               kernel_size=1,
#                               stride=1,
#                               padding=0)
#         self.P5_2 = nn.Conv2d(planes,
#                               planes,
#                               kernel_size=3,
#                               stride=1,
#                               padding=1)

#         self.P6 = nn.Conv2d(
#             planes, planes, kernel_size=3, stride=2,
#             padding=1) if self.use_p5 else nn.Conv2d(
#                 inplanes[2], planes, kernel_size=3, stride=2, padding=1)

#         self.P7 = nn.Sequential(
#             nn.ReLU(),
#             nn.Conv2d(planes, planes, kernel_size=3, stride=2, padding=1))

#     def forward(self, inputs):
#         [C3, C4, C5] = inputs

#         P5 = self.P5_1(C5)
#         P4 = self.P4_1(C4)
#         P4 = F.interpolate(P5,
#                            size=(P4.shape[2], P4.shape[3]),
#                            mode='bilinear',
#                            align_corners=True) + P4
#         P3 = self.P3_1(C3)
#         P3 = F.interpolate(P4,
#                            size=(P3.shape[2], P3.shape[3]),
#                            mode='bilinear',
#                            align_corners=True) + P3

#         del C3, C4

#         P5 = self.P5_2(P5)
#         P4 = self.P4_2(P4)
#         P3 = self.P3_2(P3)

#         P6 = self.P6(P5) if self.use_p5 else self.P6(C5)

#         del C5

#         P7 = self.P7(P6)

#         return [P3, P4, P5, P6, P7]


class RetinaFPN(nn.Module):

    def __init__(self, inplanes, planes, use_p5=False):
        super(RetinaFPN, self).__init__()
        # inplanes:[C3_inplanes,C4_inplanes,C5_inplanes]
        self.use_p5 = use_p5
        self.P3_1 = nn.Conv2d(inplanes[0],
                              planes,
                              kernel_size=1,
                              stride=1,
                              padding=0)
        self.P3_2 = nn.Conv2d(planes,
                              planes,
                              kernel_size=3,
                              stride=1,
                              padding=1)
        self.P4_1 = nn.Conv2d(inplanes[1],
                              planes,
                              kernel_size=1,
                              stride=1,
                              padding=0)
        self.P4_2 = nn.Conv2d(planes,
                              planes,
                              kernel_size=3,
                              stride=1,
                              padding=1)
        self.P5_1 = nn.Conv2d(inplanes[2],
                              planes,
                              kernel_size=1,
                              stride=1,
                              padding=0)
        self.P5_2 = nn.Conv2d(planes,
                              planes,
                              kernel_size=3,
                              stride=1,
                              padding=1)

        # self.P6 = nn.Conv2d(
        #     planes, planes, kernel_size=3, stride=2,
        #     padding=1) if self.use_p5 else nn.Conv2d(
        #         inplanes[2], planes, kernel_size=3, stride=2, padding=1)

        # self.P7 = nn.Sequential(
        #     nn.ReLU(),
        #     nn.Conv2d(planes, planes, kernel_size=3, stride=2, padding=1))

    def forward(self, inputs):
        [C3, C4, C5] = inputs

        P5 = self.P5_1(C5)
        P4 = self.P4_1(C4)
        P4 = F.interpolate(P5,
                           size=(P4.shape[2], P4.shape[3]),
                           mode='bilinear',
                           align_corners=True) + P4
        P3 = self.P3_1(C3)
        P3 = F.interpolate(P4,
                           size=(P3.shape[2], P3.shape[3]),
                           mode='bilinear',
                           align_corners=True) + P3

        del C3, C4

        P5 = self.P5_2(P5)
        P4 = self.P4_2(P4)
        P3 = self.P3_2(P3)

        # P6 = self.P6(P5) if self.use_p5 else self.P6(C5)

        del C5

        # P7 = self.P7(P6)

        return [P3, P4, P5]
