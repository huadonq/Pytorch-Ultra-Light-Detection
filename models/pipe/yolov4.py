import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

import torch
import torch.nn as nn

from models import backbones
from models.necks.yolov4 import Yolov4FPN, Yolov4TinyFPN
from models.heads.yolov4 import Yolov4TinyHead, Yolov4Head

__all__ = [
    'cspdarknettiny_yolov4',
    'cspdarknet53_yolov4',
]


class YOLOV4(nn.Module):

    def __init__(self,
                 backbone_type,
                 backbone_pretrained_path='',
                 act_type='leakyrelu',
                 per_level_num_anchors=3,
                 num_classes=80):
        super(YOLOV4, self).__init__()
        assert backbone_type in [
            'yolov4cspdarknettinybackbone', 'yolov4cspdarknet53backbone'
        ]
        self.per_level_num_anchors = per_level_num_anchors
        self.num_classes = num_classes

        self.backbone = backbones.__dict__[backbone_type](**{
            'pretrained_path': backbone_pretrained_path,
            'act_type': act_type,
        })

        if backbone_type == 'yolov4cspdarknettinybackbone':
            self.fpn = Yolov4TinyFPN(
                self.backbone.out_channels,
                per_level_num_anchors=self.per_level_num_anchors,
                num_classes=self.num_classes,
                act_type=act_type)
            self.head = Yolov4TinyHead(
                self.backbone.out_channels,
                per_level_num_anchors=self.per_level_num_anchors,
                num_classes=self.num_classes,
                act_type=act_type)
        elif backbone_type == 'yolov4cspdarknet53backbone':
            self.fpn = Yolov4FPNHead(
                self.backbone.out_channels,
                per_level_num_anchors=self.per_level_num_anchors,
                num_classes=self.num_classes,
                act_type=act_type)
            self.head = Yolov4Head(
                self.backbone.out_channels,
                per_level_num_anchors=self.per_level_num_anchors,
                num_classes=self.num_classes,
                act_type=act_type)

    def forward(self, inputs):
        features = self.backbone(inputs)
        features = self.fpn(features)
        preds = self.head(features)

        del inputs
        obj_reg_cls_heads = []
        for pred in preds:
            # feature shape:[B,H,W,3,85]

            # obj_head:feature[:, :, :, :, 0:1], shape:[B,H,W,3,1]
            # reg_head:feature[:, :, :, :, 1:5], shape:[B,H,W,3,4]
            # cls_head:feature[:, :, :, :, 5:],  shape:[B,H,W,3,80]
            obj_reg_cls_heads.append(pred)

        del features

        # if input size:[B,3,416,416]
        # features shape:[[B, 255, 52, 52],[B, 255, 26, 26],[B, 255, 13, 13]]
        # obj_reg_cls_heads shape:[[B, 52, 52, 3, 85],[B, 26, 26, 3, 85],[B, 13, 13, 3, 85]]
        return [obj_reg_cls_heads]


def _yolov4(backbone_type, backbone_pretrained_path, **kwargs):
    model = YOLOV4(backbone_type,
                   backbone_pretrained_path=backbone_pretrained_path,
                   **kwargs)

    return model


def cspdarknettiny_yolov4(backbone_pretrained_path='', **kwargs):
    return _yolov4('yolov4cspdarknettinybackbone',
                   backbone_pretrained_path=backbone_pretrained_path,
                   **kwargs)


def cspdarknet53_yolov4(backbone_pretrained_path='', **kwargs):
    return _yolov4('yolov4cspdarknet53backbone',
                   backbone_pretrained_path=backbone_pretrained_path,
                   **kwargs)


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

    net = cspdarknettiny_yolov4()
    image_h, image_w = 640, 640
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.randn(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    print(f'1111, macs: {macs}, params: {params}')
    outs = net(torch.autograd.Variable(torch.randn(8, 3, image_h, image_w)))
    for out in outs:
        for per_level_out in out:
            print('2222', per_level_out.shape)

    net = cspdarknet53_yolov4()
    image_h, image_w = 640, 640
    from thop import profile
    from thop import clever_format
    macs, params = profile(net,
                           inputs=(torch.randn(1, 3, image_h, image_w), ),
                           verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    print(f'1111, macs: {macs}, params: {params}')
    outs = net(torch.autograd.Variable(torch.randn(8, 3, image_h, image_w)))
    for out in outs:
        for per_level_out in out:
            print('2222', per_level_out.shape)