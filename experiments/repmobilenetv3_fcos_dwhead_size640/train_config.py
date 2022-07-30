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
    network = 'RepMobileNetV3_small_x1_00_fcos'
    num_classes = 1
    input_image_size = [640, 640]

    # load backbone pretrained model or not
    backbone_pretrained_path = '/home/jovyan/qrcode-single-detection/pretrain_weights/backbone/mobilenetv3/mbv3_large.pth.tar'
    # backbone_pretrained_path = os.path.join(
    #     BASE_DIR, 'pretrained_models/resnet/resnet50-acc76.322.pth')
    model = pipe.__dict__[network](**{
        'backbone_pretrained_path': backbone_pretrained_path,
        'num_classes': num_classes,
    })

    # load total pretrained model or not
    trained_model_path = ''
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

    train_dataset = CocoDetection(qrcode_data_path,
                                  set_name='train',
                                  transform=transforms.Compose([
                                      RandomHorizontalFlip(prob=0.5),
                                      RandomCrop(prob=0.5),
                                      RandomTranslate(prob=0.5),
                                      RetinaStyleResize(
                                          resize=input_image_size[0],
                                          divisor=32,
                                          stride=32,
                                          multi_scale=True,
                                          multi_scale_range=[0.8, 1.0]),
                                      Normalize(),
                                  ]))

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

    seed = 0
    # batch_size is total size
    batch_size = 32
    # num_workers is total workers
    num_workers = 4

    # choose 'SGD' or 'AdamW'
    # optimizer = (
    #     'SGD',
    #     {
    #         'lr': 0.1,
    #         'momentum': 0.9,
    #         'weight_decay': 1e-4,
    #     },
    # )

    optimizer = (
        'AdamW',
        {
            'lr': 1e-4,
            'weight_decay': 1e-3,
        },
    )

    scheduler = (
        'MultiStepLR',
        {
            'warm_up_epochs': 0,
            'gamma': 0.1,
            'milestones': [8, 12],
        },
    )

    # scheduler = (
    #     'CosineLR',
    #     {
    #         'warm_up_epochs': 0,
    #     },
    # )

    epochs = 13
    eval_epoch = [i for i in range(0, epoch, 2)]
    print_interval = 50

    # 'COCO' or 'VOC'
    eval_type = 'COCO'
    eval_voc_iou_threshold_list = [
        0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95
    ]
    save_model_metric = 'IoU=0.50:0.95,area=all,maxDets=100,mAP'

    sync_bn = False
    apex = True