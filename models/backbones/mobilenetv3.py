'''
https://arxiv.org/pdf/1905.02244.pdf
'''
import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

import torch
import torch.nn as nn
import torch.nn.functional as F
__all__ = [
    'MobileNetV3_small_x1_00',
]

mobilenetv3_config_dict = {
    'MobileNetV3_small_x0_35': {
        'type': 'small',
        'scale': 0.35,
    },
    'MobileNetV3_small_x0_50': {
        'type': 'small',
        'scale': 0.5,
    },
    'MobileNetV3_small_x0_75': {
        'type': 'small',
        'scale': 0.75,
    },
    'MobileNetV3_small_x1_00': {
        'type': 'small',
        'scale': 1.0,
    },
    'MobileNetV3_small_x1_25': {
        'type': 'small',
        'scale': 1.25,
    },
    'MobileNetV3_large_x0_35': {
        'type': 'large',
        'scale': 0.35,
    },
    'MobileNetV3_large_x0_50': {
        'type': 'large',
        'scale': 0.5,
    },
    'MobileNetV3_large_x0_75': {
        'type': 'large',
        'scale': 0.75,
    },
    'MobileNetV3_large_x1_00': {
        'type': 'large',
        'scale': 1.0,
    },
    'MobileNetV3_large_x1_25': {
        'type': 'large',
        'scale': 1.25,
    },
}


class HardSwish(nn.Module):
    def __init__(self):
        super(HardSwish, self).__init__()

    def forward(self, x):
        x = x * (F.relu6(x + 3, inplace=True) / 6.0)

        return x


class Hardsigmoid(nn.Module):
    def __init__(self):
        super(Hardsigmoid, self).__init__()

    def forward(self, x):
        x = F.relu6(x + 3, inplace=True) / 6.0

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
                 act_type='relu'):
        super(ConvBnActBlock, self).__init__()
        assert act_type in ['relu', 'hard_swish']
        self.has_act = has_act
        self.layer = nn.Sequential(
            nn.Conv2d(inplanes,
                      planes,
                      kernel_size,
                      stride=stride,
                      padding=padding,
                      groups=groups,
                      bias=False),
            nn.BatchNorm2d(planes) if has_bn else nn.Sequential())
        if self.has_act:
            if act_type == 'relu':
                self.act = nn.ReLU(inplace=True)
            elif act_type == 'hard_swish':
                self.act = HardSwish()

    def forward(self, x):
        x = self.layer(x)

        if self.has_act:
            x = self.act(x)

        return x


class SeBlock(nn.Module):
    def __init__(self, inplanes, se_ratio=0.25):
        super(SeBlock, self).__init__()
        squeezed_planes = max(1, int(inplanes * se_ratio))
        self.layers = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(inplanes,
                      squeezed_planes,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=True), nn.ReLU(inplace=True),
            nn.Conv2d(squeezed_planes,
                      inplanes,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=True), Hardsigmoid())

    def forward(self, x):
        x=x*self.layers(x)

        return x


class MBConvBlock(nn.Module):
    def __init__(self,
                 inplanes,
                 expand_planes,
                 planes,
                 kernel_size,
                 stride,
                 act=None,
                 use_se=False):
        super(MBConvBlock, self).__init__()
        self.inplanes = inplanes
        self.planes = planes
        self.stride = stride
        self.use_se = use_se
        self.expand_conv = ConvBnActBlock(inplanes,
                                          expand_planes,
                                          kernel_size=1,
                                          stride=1,
                                          padding=0,
                                          groups=1,
                                          has_bn=True,
                                          has_act=True,
                                          act_type=act)
        self.depthwise_conv = ConvBnActBlock(expand_planes,
                                             expand_planes,
                                             kernel_size=kernel_size,
                                             stride=stride,
                                             padding=int(
                                                 (kernel_size - 1) // 2),
                                             groups=expand_planes,
                                             has_bn=True,
                                             has_act=True,
                                             act_type=act)

        if self.use_se:
            self.se = SeBlock(expand_planes)

        self.pointwise_conv = ConvBnActBlock(expand_planes,
                                             planes,
                                             kernel_size=1,
                                             stride=1,
                                             padding=0,
                                             groups=1,
                                             has_bn=True,
                                             has_act=False,
                                             act_type=act)

    def forward(self, inputs):
        x = self.expand_conv(inputs)
        x = self.depthwise_conv(x)
        if self.use_se:
            x = self.se(x)
        x = self.pointwise_conv(x)

        if self.stride == 1 and self.inplanes == self.planes:
            x = x + inputs

        del inputs

        return x


class MobileNetV3(nn.Module):
    def __init__(self, arch):
        super(MobileNetV3, self).__init__()
        self.config = mobilenetv3_config_dict[arch]
        self.type = self.config['type']
        self.scale = self.config['scale']

        assert self.type in ['small', 'large'], 'wrong type!'
        assert self.scale in [0.35, 0.5, 0.75, 1.0, 1.25], 'wrong scale!'

        layers_setting_dict = {
            'large': {
                'layer': [
                    # k, exp, c,  se,     nl,  s,
                    [3, 16, 16, False, 'relu', 1],
                    [3, 64, 24, False, 'relu', 2],
                    [3, 72, 24, False, 'relu', 1],  #2
                    [5, 72, 40, True, 'relu', 2],
                    [5, 120, 40, True, 'relu', 1],
                    [5, 120, 40, True, 'relu', 1],  #5
                    [3, 240, 80, False, 'hard_swish', 2],
                    [3, 200, 80, False, 'hard_swish', 1],
                    [3, 184, 80, False, 'hard_swish', 1],
                    [3, 184, 80, False, 'hard_swish', 1],
                    [3, 480, 112, True, 'hard_swish', 1],
                    [3, 672, 112, True, 'hard_swish', 1],  #11
                    [5, 672, 160, True, 'hard_swish', 2],
                    [5, 960, 160, True, 'hard_swish', 1],
                    [5, 960, 160, True, 'hard_swish', 1],  #14
                ],
                'squeeze':
                960,
                'out_idx': [2, 5, 11, 14],
            },
            'small': {
                # 'layer': [
                #     # k, exp, c,  se,     nl,  s,
                #     [3, 16, 16, True, 'relu', 2],  #0
                #     [3, 72, 24, False, 'relu', 2],
                #     [3, 88, 24, False, 'relu', 1],  #2
                #     [5, 96, 40, True, 'hard_swish', 2],
                #     [5, 240, 40, True, 'hard_swish', 1],
                #     [5, 240, 40, True, 'hard_swish', 1],
                #     [5, 120, 48, True, 'hard_swish', 1],
                #     [5, 144, 48, True, 'hard_swish', 1],  #7
                #     [5, 288, 96, True, 'hard_swish', 2],
                #     [5, 576, 96, True, 'hard_swish', 1],
                #     [5, 576, 96, True, 'hard_swish', 1],  #10
                # ],

                'layer': [
                    # k, exp, c,  se,     nl,  s,
                    [3, 16, 16, True, 'relu', 2],  #0
                    [3, 72, 24, False, 'relu', 2],
                    [3, 88, 24, False, 'relu', 1],  #2
                    [5, 96, 40, True, 'hard_swish', 2],
                    [5, 240, 40, True, 'hard_swish', 1],
                    [5, 240, 40, True, 'hard_swish', 1],
                    [5, 120, 48, True, 'hard_swish', 1],
                    [5, 144, 48, True, 'hard_swish', 1],  #7
                    [5, 288, 96, True, 'hard_swish', 2],
                    [5, 576, 96, True, 'hard_swish', 1],
                    [5, 576, 96, True, 'hard_swish', 1],  #10
                ],
                

               

                'squeeze':
                576,
                # 'out_idx': [0, 2, 7, 10],
                'out_idx': [2, 7, 10],
            },
        }

        self.layers_config = layers_setting_dict[self.type]
        self.layers_setting = self.layers_config['layer']
        self.squeeze = self.layers_config['squeeze']
        self.out_idx = self.layers_config['out_idx']

        self.mid_planes = self.make_divisible(16 * self.scale)
        self.first_conv = ConvBnActBlock(3,
                                         self.mid_planes,
                                         kernel_size=3,
                                         stride=2,
                                         padding=1,
                                         groups=1,
                                         has_bn=True,
                                         has_act=True,
                                         act_type='hard_swish')  # h/2,w/2

        self.out_channels = []

        mid_layers = []
        for idx, per_layer_setting in enumerate(self.layers_setting):
            if idx in self.out_idx:
                self.out_channels.append(
                    self.make_divisible(self.scale * per_layer_setting[2]))

            mid_layers.append(
                MBConvBlock(
                    self.mid_planes,
                    self.make_divisible(self.scale * per_layer_setting[1]),
                    self.make_divisible(self.scale * per_layer_setting[2]),
                    kernel_size=per_layer_setting[0],
                    stride=per_layer_setting[5],
                    act=per_layer_setting[4],
                    use_se=per_layer_setting[3]))

            self.mid_planes = self.make_divisible(self.scale *
                                                  per_layer_setting[2])

        self.mid_layers = nn.Sequential(*mid_layers)

        self.last_conv = ConvBnActBlock(self.mid_planes,
                                        self.make_divisible(self.scale *
                                                            self.squeeze),
                                        kernel_size=1,
                                        stride=1,
                                        padding=0,
                                        groups=1,
                                        has_bn=True,
                                        has_act=True,
                                        act_type='hard_swish')

        #ori sync with forward "out[-1] = self.last_conv(out[-1])"
        # self.out_planes[-1] = self.make_divisible(self.scale * self.squeeze)

    def make_divisible(self, v, divisor=4, min_value=None):
        if min_value is None:
            min_value = divisor
        new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
        if new_v < 0.9 * v:
            new_v += divisor

        return new_v

    def forward(self, x):
        x = self.first_conv(x)

        out = []
        for idx, per_layer in enumerate(self.mid_layers):
            x = per_layer(x)
            if idx in self.out_idx:
                out.append(x)
        
        #ori
        # out[-1] = self.last_conv(out[-1])

        return out

def MobileNetV3_small_x1_00(pretrained_path=''):
    model = MobileNetV3('MobileNetV3_small_x1_00')

    return model


if __name__ == '__main__':
    import os
    import random
    import numpy as np
    import torch
    seed = 0
    # for hash
    os.environ['PYTHONHASHSEED'] = str(seed)
    # for python and numpy
    random.seed(seed)
    np.random.seed(seed)
    # for cpu gpu
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    net = MobileNetV3('MobileNetV3_small_x1_00')
    image_h, image_w = 640, 640
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                            inputs=(torch.randn(1, 3, image_h, image_w), ),
                            verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    out = net(torch.autograd.Variable(torch.randn(6, 3, image_h, image_w)))
    print(f'MobileNetV3_small_x1_00 --> macs: {macs}, params: {params}')
    print(net.out_channels)

    for per_feature in out:
        print('MobileNetV3_small_x1_00 feature -->', per_feature.shape)

    net = MobileNetV3('MobileNetV3_large_x0_50')
    image_h, image_w = 640, 640
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                            inputs=(torch.randn(1, 3, image_h, image_w), ),
                            verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    out = net(torch.autograd.Variable(torch.randn(6, 3, image_h, image_w)))
    print(f'MobileNetV3_large_x0_50 --> macs: {macs}, params: {params}')
    print(net.out_channels)
    for per_feature in out:
        print('MobileNetV3_large_x0_50 --> feature', per_feature.shape)