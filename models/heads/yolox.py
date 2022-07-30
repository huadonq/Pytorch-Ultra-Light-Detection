import math
import torch
import torch.nn as nn
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


class YOLOXHead(nn.Module):

    def __init__(self,
                 inplanes_list,
                 planes,
                 num_classes,
                 block=ConvBnActBlock,
                 act_type='silu'):
        super(YOLOXHead, self).__init__()

        self.stem_conv_list = nn.ModuleList()
        self.cls_conv_list = nn.ModuleList()
        self.reg_conv_list = nn.ModuleList()
        self.cls_pred_list = nn.ModuleList()
        self.reg_pred_list = nn.ModuleList()
        self.obj_pred_list = nn.ModuleList()

        for i in range(len(inplanes_list)):
            self.stem_conv_list.append(
                ConvBnActBlock(inplanes_list[i],
                               planes,
                               kernel_size=1,
                               stride=1,
                               padding=0,
                               groups=1,
                               has_bn=True,
                               has_act=True,
                               act_type=act_type))
            self.cls_conv_list.append(
                nn.Sequential(
                    block(planes,
                          planes,
                          kernel_size=3,
                          stride=1,
                          padding=1,
                          groups=1,
                          has_bn=True,
                          has_act=True,
                          act_type=act_type),
                    block(planes,
                          planes,
                          kernel_size=3,
                          stride=1,
                          padding=1,
                          groups=1,
                          has_bn=True,
                          has_act=True,
                          act_type=act_type)))
            self.reg_conv_list.append(
                nn.Sequential(
                    block(planes,
                          planes,
                          kernel_size=3,
                          stride=1,
                          padding=1,
                          groups=1,
                          has_bn=True,
                          has_act=True,
                          act_type=act_type),
                    block(planes,
                          planes,
                          kernel_size=3,
                          stride=1,
                          padding=1,
                          groups=1,
                          has_bn=True,
                          has_act=True,
                          act_type=act_type)))
            self.cls_pred_list.append(
                nn.Conv2d(planes,
                          num_classes,
                          kernel_size=1,
                          stride=1,
                          padding=0,
                          groups=1,
                          bias=True))
            self.reg_pred_list.append(
                nn.Conv2d(planes,
                          4,
                          kernel_size=1,
                          stride=1,
                          padding=0,
                          groups=1,
                          bias=True))
            self.obj_pred_list.append(
                nn.Conv2d(planes,
                          1,
                          kernel_size=1,
                          stride=1,
                          padding=0,
                          groups=1,
                          bias=True))
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        obj_outputs, cls_outputs, reg_outputs = [], [], []
        for i, x in enumerate(inputs):
            x = self.stem_conv_list[i](x)

            cls_out = self.cls_conv_list[i](x)
            reg_out = self.reg_conv_list[i](x)

            obj_out = self.obj_pred_list[i](reg_out)
            reg_out = self.reg_pred_list[i](reg_out)
            cls_out = self.cls_pred_list[i](cls_out)

            obj_out = self.sigmoid(obj_out)
            cls_out = self.sigmoid(cls_out)

            obj_outputs.append(obj_out)
            cls_outputs.append(cls_out)
            reg_outputs.append(reg_out)

        return obj_outputs, cls_outputs, reg_outputs
