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
from tools.common import load_state_dict


__all__ = [
    'RepMobileNetV3_small_x1_00',
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
                 deploy=False,
                 use_se=False):
        super(MBConvBlock, self).__init__()
        self.deploy = deploy
        self.inplanes = inplanes
        self.expand_planes = expand_planes
        self.planes = planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.use_se = use_se
        self.padding = int(kernel_size - 1) // 2

        self.nonlinearity = nn.ReLU()


        if self.deploy:
            self.expand_reparam = nn.Conv2d(in_channels=inplanes, out_channels=expand_planes, kernel_size=1, stride=1, bias=True)
            self.dw_reparam = nn.Conv2d(in_channels=expand_planes, out_channels=expand_planes, kernel_size=kernel_size, stride=stride,
                                      padding=self.padding, dilation=1, groups=expand_planes, bias=True, padding_mode='zeros')
            self.pw_reparam = nn.Conv2d(in_channels=expand_planes, out_channels=planes, kernel_size=1, stride=1, bias=True)
        else:

            self.expand_conv = ConvBnActBlock(inplanes,
                                            expand_planes,
                                            kernel_size=1,
                                            stride=1,
                                            padding=0,
                                            groups=1,
                                            has_bn=True,
                                            has_act=False,
                                            act_type=act)
            self.expand_bn_layer = nn.BatchNorm2d(inplanes) if expand_planes == inplanes and self.stride == 1 else None

            self.depthwise_conv = ConvBnActBlock(expand_planes,
                                                expand_planes,
                                                kernel_size=kernel_size,
                                                stride=stride,
                                                padding=self.padding,
                                                groups=expand_planes,
                                                has_bn=True,
                                                has_act=False,
                                                act_type=act)
            self.dw_1x1 = ConvBnActBlock(expand_planes,
                                        expand_planes,
                                        kernel_size=1,
                                        stride=stride,
                                        padding=0,
                                        groups=expand_planes,
                                        has_bn=True,
                                        has_act=False,
                                        act_type=act)
            self.dw_bn_layer = nn.BatchNorm2d(expand_planes) if self.stride == 1 else None


            if self.use_se:
                # ori
                # self.se = SeBlock(expand_planes)
                self.se = nn.Identity()

            self.pointwise_conv = ConvBnActBlock(expand_planes,
                                                planes,
                                                kernel_size=1,
                                                stride=1,
                                                padding=0,
                                                groups=1,
                                                has_bn=True,
                                                has_act=False,
                                                act_type=act)
            self.pw_bn_layer = nn.BatchNorm2d(expand_planes) if expand_planes == planes and self.stride == 1 else None

    def forward(self, inputs):
        if self.deploy:
            x = self.expand_reparam(inputs)
            x = self.nonlinearity(x)
            x = self.dw_reparam(x)
            x = self.nonlinearity(x)
            x = self.pw_reparam(x)
            x = self.nonlinearity(x)

            return x



        # 1x1 expand conv
        if self.expand_bn_layer is None:
            id_out = 0
        else:
            id_out = self.expand_bn_layer(inputs)

        x = self.expand_conv(inputs)
        x = id_out + x
        x = self.nonlinearity(x)

        # 3x3 or 5x5 dw conv

        if self.dw_bn_layer is None:
            id_out = 0
        else:
            id_out = self.dw_bn_layer(x)

        x_conv_1x1 = self.dw_1x1(x)
        x = self.depthwise_conv(x)
        x = x + id_out + x_conv_1x1
        x = self.nonlinearity(x)


        if self.use_se:
            x = self.se(x)

        # 1x1 pw conv
        if self.pw_bn_layer is None:
            id_out = 0
        else:
            id_out = self.pw_bn_layer(x)

        x = self.pointwise_conv(x)
        x = x + id_out
        x = self.nonlinearity(x)

        # if self.stride == 1 and self.inplanes == self.planes:
        #     x = x + inputs

        del inputs

        return x

    
    def get_equivalent_kernel_bias(self):
        
        # expand
        expand_kernel, expand_bias = self._fuse_bn_tensor(self.expand_conv[0])
        expand_kernel_id, expand_bias_id = self._fuse_bn_tensor(self.expand_bn_layer, 1)
        expand_kernel_1x1 = expand_kernel + expand_kernel_id
        expand_bias_1x1 = expand_bias + expand_bias_id


        # dw
        dw_kernel_3x3_5x5, dw_bias_3x3_5x5 = self._fuse_bn_tensor(self.depthwise_conv[0])
        dw_kernel_1x1, dw_bias_1x1 = self._fuse_bn_tensor(self.dw_1x1[0])
        dw_kernel_id, dw_bias_id = self._fuse_bn_tensor(self.dw_bn_layer, self.expand_planes)

        if self.kernel_size == 5:
            dw_1x1_convert = self._pad_1x1_to_5x5_tensor(dw_kernel_1x1)
        else:
            dw_1x1_convert = self._pad_1x1_to_3x3_tensor(dw_kernel_1x1)

        dw_kernel = dw_kernel_3x3_5x5 + dw_1x1_convert + dw_kernel_id
        dw_bias = dw_bias_3x3_5x5 + dw_bias_1x1 + dw_bias_id

        # pw
        pw_kernel, pw_bias = self._fuse_bn_tensor(self.pointwise_conv[0])
        pw_kernel_id, pw_bias_id = self._fuse_bn_tensor(self.pw_bn_layer, 1)
        pw_kernel_1x1 = pw_kernel + pw_kernel_id
        pw_bias_1x1 = pw_bias + pw_bias_id



        return expand_kernel, expand_bias, dw_kernel, dw_bias, pw_kernel_1x1, pw_bias_1x1

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1,1,1,1])
    
    def _pad_1x1_to_5x5_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [2,2,2,2])

    def _fuse_bn_tensor(self, branch, groups=None):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            bias = branch.conv.bias
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            # if not hasattr(self, 'id_tensor'):
            input_dim = self.in_channels // groups # self.groups
            if groups == 1:
                ks = 1
            else:
                ks = self.kernel_size

            kernel_value = np.zeros((self.in_channels, input_dim, ks, ks), dtype=np.float32)
            for i in range(self.in_channels):
                if ks == 1:
                    kernel_value[i, i % input_dim, 0, 0] = 1
                else:
                    kernel_value[i, i % input_dim, 1, 1] = 1
            self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)

            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std


    def switch_to_deploy(self):
        expand_kernel, expand_bias, dw_kernel, dw_bias, pw_kernel, pw_bias = self.get_equivalent_kernel_bias()

        
        self.expand_reparam = nn.Conv2d(
            in_channels=self.expand_conv[0].in_channels,
            out_channels=self.expand_conv[0].out_channels, 
            kernel_size=1, 
            stride=1, 
            bias=True
            )
        self.dw_reparam = nn.Conv2d(
            in_channels=self.depthwise_conv[0].in_channels, 
            out_channels=self.depthwise_conv[0].out_channels,                              
            kernel_size=self.depthwise_conv[0].kernel_size, 
            stride=self.depthwise_conv[0].stride,
            padding=self.depthwise_conv[0].padding, 
            groups=self.depthwise_conv[0].in_channels, 
            bias=True, 
            )
        self.pw_reparam = nn.Conv2d(
            in_channels=self.pointwise_conv[0].in_channels,
            out_channels=self.pointwise_conv[0].out_channels, 
            kernel_size=1, 
            stride=1, 
            bias=True
            )


        self.expand_reparam.weight.data = expand_kernel
        self.expand_reparam.bias.data = expand_bias
        self.dw_reparam.weight.data = dw_kernel
        self.dw_reparam.bias.data = dw_bias
        self.pw_reparam.weight.data = pw_kernel
        self.pw_reparam.bias.data = pw_bias

        for para in self.parameters():
            para.detach_()


        self.__delattr__('dw_1x1')
        self.__delattr__(f'expand_conv')
        self.__delattr__(f'depthwise_conv')
        self.__delattr__(f'pointwise_conv')
        
        if hasattr(self, 'expand_bn_layer'):
            self.__delattr__('expand_bn_layer')
        if hasattr(self, 'dw_bn_layer'):
            self.__delattr__('dw_bn_layer')
        if hasattr(self, 'dw_bn_layer'):
            self.__delattr__('dw_bn_layer')
        if hasattr(self, 'pw_bn_layer'):
            self.__delattr__('pw_bn_layer')
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')
        self.deploy = True
        


class RepMobileNetV3(nn.Module):
    def __init__(self, arch):
        super(RepMobileNetV3, self).__init__()
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

def RepMobileNetV3_small_x1_00(pretrained_path=''):
    model = RepMobileNetV3('MobileNetV3_small_x1_00')
    load_state_dict(pretrained_path, model)
    

    return model
def RepMobileNetV3_model_convert(model:torch.nn.Module, save_path=None, do_copy=True):
    if do_copy:
        model = copy.deepcopy(model)
    for module in model.modules():
        if hasattr(module, 'switch_to_deploy'):
            module.switch_to_deploy()
    if save_path is not None:
        torch.save(model.state_dict(), save_path)
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

    net = RepMobileNetV3('MobileNetV3_small_x1_00')
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

    net = RepMobileNetV3('MobileNetV3_large_x0_50')
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