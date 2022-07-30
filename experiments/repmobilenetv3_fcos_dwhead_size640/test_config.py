import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
sys.path.append(BASE_DIR)

from tools.path import qrcode_data_path

from models import pipe
from models import losses
from models import decode
from datasets.cocodataset import CocoDetection
from tools.common import (RetinaStyleResize, YoloStyleResize, RandomHorizontalFlip, RandomCrop, RandomTranslate, 
                        Normalize, DetectionCollater, load_state_dict)

import torch
import torchvision.transforms as transforms


class config:
    network = 'resnet50_retinanet'
    num_classes = 1
    input_image_size = [400, 667]

    model = pipe.__dict__[network](**{
        'backbone_pretrained_path': '',
        'num_classes': num_classes,
    })

    # load total pretrained model or not
    # trained_model_path = ''
    trained_model_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'checkpoints',
        [pth for pth in os.listdir(os.path.join(
            os.path.dirname(os.path.abspath(__file__)),'checkpoints')) if 'best' in pth][0]
    )


    load_state_dict(trained_model_path, model)

    criterion = losses.__dict__['FCOSLoss'](
        **{
            'strides': [8, 16, 32],
            'mi': [[-1, 64], [64, 128], [128, 256], [256, 512],
                   [512, 100000000]],
            'alpha': 0.25,
            'gamma': 2.,
            'cls_loss_weight': 1.,
            'box_loss_weight': 1.,
            'center_ness_loss_weight': 1.,
            'box_loss_iou_type': 'CIoU',
            'center_sample_radius': 1.5,
            'use_center_sample': True,
        })

    decoder = decode.__dict__['FCOSDecoder'](**{
        'strides': [8, 16, 32],
        'max_object_num': 100,
        'min_score_threshold': 0.05,
        'topn': 1000,
        'nms_type': 'python_nms',
        'nms_threshold': 0.6,
    })

    val_dataset = CocoDetection(qrcode_data_path,
                                set_name='val',
                                transform=transforms.Compose([
                                    RetinaStyleResize(
                                        resize=input_image_size[0],
                                        divisor=32,
                                        stride=32,
                                        multi_scale=False,
                                        multi_scale_range=[0.8, 1.0]),
                                    Normalize(),
                                ]))
    collater = DetectionCollater()

    # 'COCO' or 'VOC'
    eval_type = 'COCO'
    eval_voc_iou_threshold_list = [
        0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95
    ]

    seed = 0
    # batch_size is total size
    batch_size = 32
    # num_workers is total workers
    num_workers = 4