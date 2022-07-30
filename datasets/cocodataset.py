import os
import cv2
import math
import numpy as np

from pycocotools.coco import COCO
from torch.utils.data import Dataset

COCO_CLASSES = [
    'person',
    'bicycle',
    'car',
    'motorcycle',
    'airplane',
    'bus',
    'train',
    'truck',
    'boat',
    'traffic light',
    'fire hydrant',
    'stop sign',
    'parking meter',
    'bench',
    'bird',
    'cat',
    'dog',
    'horse',
    'sheep',
    'cow',
    'elephant',
    'bear',
    'zebra',
    'giraffe',
    'backpack',
    'umbrella',
    'handbag',
    'tie',
    'suitcase',
    'frisbee',
    'skis',
    'snowboard',
    'sports ball',
    'kite',
    'baseball bat',
    'baseball glove',
    'skateboard',
    'surfboard',
    'tennis racket',
    'bottle',
    'wine glass',
    'cup',
    'fork',
    'knife',
    'spoon',
    'bowl',
    'banana',
    'apple',
    'sandwich',
    'orange',
    'broccoli',
    'carrot',
    'hot dog',
    'pizza',
    'donut',
    'cake',
    'chair',
    'couch',
    'potted plant',
    'bed',
    'dining table',
    'toilet',
    'tv',
    'laptop',
    'mouse',
    'remote',
    'keyboard',
    'cell phone',
    'microwave',
    'oven',
    'toaster',
    'sink',
    'refrigerator',
    'book',
    'clock',
    'vase',
    'scissors',
    'teddy bear',
    'hair drier',
    'toothbrush',
]

COCO_CLASSES_COLOR = [(241, 23, 78), (63, 71, 49), (67, 79, 143),
                      (32, 250, 205), (136, 228, 157), (135, 125, 104),
                      (151, 46, 171), (129, 37, 28), (3, 248, 159),
                      (154, 129, 58), (93, 155, 200), (201, 98, 152),
                      (187, 194, 70), (122, 144, 121), (168, 31, 32),
                      (168, 68, 189), (173, 68, 45), (200, 81, 154),
                      (171, 114, 139), (216, 211, 39), (187, 119, 238),
                      (201, 120, 112), (129, 16, 164), (211, 3, 208),
                      (169, 41, 248), (100, 77, 159), (140, 104, 243),
                      (26, 165, 41), (225, 176, 197), (35, 212, 67),
                      (160, 245, 68), (7, 87, 70), (52, 107, 85),
                      (103, 64, 188), (245, 76, 17), (248, 154, 59),
                      (77, 45, 123), (210, 95, 230), (172, 188, 171),
                      (250, 44, 233), (161, 71, 46), (144, 14, 134),
                      (231, 142, 186), (34, 1, 200), (144, 42, 108),
                      (222, 70, 139), (138, 62, 77), (178, 99, 61),
                      (17, 94, 132), (93, 248, 254), (244, 116, 204),
                      (138, 165, 238), (44, 216, 225), (224, 164, 12),
                      (91, 126, 184), (116, 254, 49), (70, 250, 105),
                      (252, 237, 54), (196, 136, 21), (234, 13, 149),
                      (66, 43, 47), (2, 73, 234), (118, 181, 5),
                      (105, 99, 225), (150, 253, 92), (59, 2, 121),
                      (176, 190, 223), (91, 62, 47), (198, 124, 140),
                      (100, 135, 185), (20, 207, 98), (216, 38, 133),
                      (17, 202, 208), (216, 135, 81), (212, 203, 33),
                      (108, 135, 76), (28, 47, 170), (142, 128, 121),
                      (23, 161, 179), (33, 183, 224)]


class CocoDetection(Dataset):

    def __init__(self, root_dir, set_name='train', transform=None):
        assert set_name in ['train', 'val'], 'Wrong set name!'

        self.image_dir = os.path.join(root_dir, set_name, 'images')
        self.annot_dir = os.path.join(root_dir, set_name, 'annotations',
                                      f'{set_name}.json')
        self.coco = COCO(self.annot_dir)

        self.image_ids = self.coco.getImgIds()

        if 'train' in set_name:
            # filter image id without annotation,from 118287 ids to 117266 ids
            ids = []
            for image_id in self.image_ids:
                annot_ids = self.coco.getAnnIds(imgIds=image_id)
                annots = self.coco.loadAnns(annot_ids)
                if len(annots) == 0:
                    continue
                ids.append(image_id)
            self.image_ids = ids

        self.cat_ids = self.coco.getCatIds()
        self.cats = sorted(self.coco.loadCats(self.cat_ids),
                           key=lambda x: x['id'])
        self.num_classes = len(self.cats)

        # cat_id is an original cat id,coco_label is set from 0 to 79
        self.cat_id_to_cat_name = {cat['id']: cat['name'] for cat in self.cats}
        self.cat_id_to_coco_label = {
            cat['id']: i
            for i, cat in enumerate(self.cats)
        }
        self.coco_label_to_cat_id = {
            i: cat['id']
            for i, cat in enumerate(self.cats)
        }
        self.coco_label_to_cat_name = {
            coco_label: self.cat_id_to_cat_name[cat_id]
            for coco_label, cat_id in self.coco_label_to_cat_id.items()
        }

        self.transform = transform

        print(f'Dataset Size:{len(self.image_ids)}')
        print(f'Dataset Class Num:{self.num_classes}')

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image = self.load_image(idx)
        annots = self.load_annots(idx)

        scale = np.array(1.).astype(np.float32)
        size = np.array([image.shape[0], image.shape[1]]).astype(np.float32)

        sample = {
            'image': image,
            'annots': annots,
            'scale': scale,
            'size': size,
        }

        if self.transform:
            sample = self.transform(sample)

        return sample

    def load_image(self, idx):
        file_name = self.coco.loadImgs(self.image_ids[idx])[0]['file_name']
        image = cv2.imdecode(
            np.fromfile(os.path.join(self.image_dir, file_name),
                        dtype=np.uint8), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image.astype(np.float32)

    def load_annots(self, idx):
        annot_ids = self.coco.getAnnIds(imgIds=self.image_ids[idx])
        annots = self.coco.loadAnns(annot_ids)

        image_info = self.coco.loadImgs(self.image_ids[idx])[0]
        image_h, image_w = image_info['height'], image_info['width']

        targets = np.zeros((0, 5))
        if len(annots) == 0:
            return targets.astype(np.float32)

        # filter annots
        for annot in annots:
            if 'ignore' in annot.keys():
                continue
            # bbox format:[x_min, y_min, w, h]
            bbox = annot['bbox']

            inter_w = max(0, min(bbox[0] + bbox[2], image_w) - max(bbox[0], 0))
            inter_h = max(0, min(bbox[1] + bbox[3], image_h) - max(bbox[1], 0))
            if inter_w * inter_h == 0:
                continue
            if bbox[2] * bbox[3] < 1 or bbox[2] < 1 or bbox[3] < 1:
                continue
            if annot['category_id'] not in self.cat_ids:
                continue

            target = np.zeros((1, 5))
            target[0, :4] = bbox
            target[0, 4] = self.cat_id_to_coco_label[annot['category_id']]
            targets = np.append(targets, target, axis=0)

        # transform bbox targets from [x_min, y_min, w, h] to [x_min, y_min, x_max, y_max]
        targets[:, 2] = targets[:, 0] + targets[:, 2]
        targets[:, 3] = targets[:, 1] + targets[:, 3]

        return targets.astype(np.float32)


class MosaicResizeCocoDetection(CocoDetection):
    '''
    When using MosaicResizeCocoDetection class, don't use YoloStyleResize/RetinaStyleResize data augment.
    '''

    def __init__(self,
                 root_dir,
                 set_name='train2017',
                 resize=640,
                 stride=32,
                 use_multi_scale=False,
                 multi_scale_range=[0.5, 1.0],
                 transform=None):
        assert set_name in ['train2017', 'val2017'], 'Wrong set name!'

        self.image_dir = os.path.join(root_dir, 'images', set_name)
        self.annot_dir = os.path.join(root_dir, 'annotations',
                                      f'instances_{set_name}.json')
        self.coco = COCO(self.annot_dir)

        self.resize = resize
        self.stride = stride
        self.use_multi_scale = use_multi_scale
        self.multi_scale_range = multi_scale_range
        self.transform = transform

        self.image_ids = self.coco.getImgIds()

        if 'train' in set_name:
            # filter image id without annotation,from 118287 ids to 117266 ids
            ids = []
            for image_id in self.image_ids:
                annot_ids = self.coco.getAnnIds(imgIds=image_id)
                annots = self.coco.loadAnns(annot_ids)
                if len(annots) == 0:
                    continue
                ids.append(image_id)
            self.image_ids = ids

        self.cat_ids = self.coco.getCatIds()
        self.cats = sorted(self.coco.loadCats(self.cat_ids),
                           key=lambda x: x['id'])
        self.num_classes = len(self.cats)

        # cat_id is an original cat id,coco_label is set from 0 to 79
        self.cat_id_to_cat_name = {cat['id']: cat['name'] for cat in self.cats}
        self.cat_id_to_coco_label = {
            cat['id']: i
            for i, cat in enumerate(self.cats)
        }
        self.coco_label_to_cat_id = {
            i: cat['id']
            for i, cat in enumerate(self.cats)
        }
        self.coco_label_to_cat_name = {
            coco_label: self.cat_id_to_cat_name[cat_id]
            for coco_label, cat_id in self.coco_label_to_cat_id.items()
        }

        print(f'Dataset Size:{len(self.image_ids)}')
        print(f'Dataset Class Num:{self.num_classes}')

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        # mosaic center x, y
        x_ctr, y_ctr = [int(self.resize), int(self.resize)]
        # 4 images ids
        image_ids = [idx] + [
            np.random.randint(0, len(self.image_ids)) for _ in range(3)
        ]
        # 4 images annots
        image_annots = []
        # combined image by 4 images
        combined_img = np.zeros(
            (int(self.resize * 2), int(self.resize * 2), 3), dtype=np.float32)

        for i, idx in enumerate(image_ids):
            image = self.load_image(idx)
            annots = self.load_annots(idx)

            h, w, _ = image.shape

            factor = self.resize / max(h, w)

            resize_h, resize_w = math.ceil(h * factor), math.ceil(w * factor)
            image = cv2.resize(image, (resize_w, resize_h))
            annots[:, :4] *= factor

            # top left img
            if i == 0:
                # xmin, ymin, xmax, ymax (large image)
                x1a, y1a, x2a, y2a = max(x_ctr - resize_w,
                                         0), max(y_ctr - resize_h,
                                                 0), x_ctr, y_ctr
                # xmin, ymin, xmax, ymax (small image)
                x1b, y1b, x2b, y2b = resize_w - (x2a - x1a), resize_h - (
                    y2a - y1a), resize_w, resize_h
            # top right img
            elif i == 1:
                x1a, y1a, x2a, y2a = x_ctr, max(y_ctr - resize_h, 0), min(
                    x_ctr + resize_w, int(self.resize * 2)), y_ctr
                x1b, y1b, x2b, y2b = 0, resize_h - (y2a - y1a), min(
                    resize_w, x2a - x1a), resize_h
            # bottom left img
            elif i == 2:
                x1a, y1a, x2a, y2a = max(x_ctr - resize_w,
                                         0), y_ctr, x_ctr, min(
                                             int(self.resize * 2),
                                             y_ctr + resize_h)
                x1b, y1b, x2b, y2b = resize_w - (x2a - x1a), 0, max(
                    x_ctr, resize_w), min(y2a - y1a, resize_h)
            # bottom right img
            elif i == 3:
                x1a, y1a, x2a, y2a = x_ctr, y_ctr, min(
                    x_ctr + resize_w,
                    int(self.resize * 2)), min(int(self.resize * 2),
                                               y_ctr + resize_h)
                x1b, y1b, x2b, y2b = 0, 0, min(resize_w, x2a - x1a), min(
                    y2a - y1a, resize_h)

            # combined_img[ymin:ymax, xmin:xmax]
            combined_img[y1a:y2a, x1a:x2a] = image[y1b:y2b, x1b:x2b]
            padw, padh = x1a - x1b, y1a - y1b

            # annot coordinates transform
            if annots.shape[0] > 0:
                annots[:, 0] = annots[:, 0] + padw
                annots[:, 1] = annots[:, 1] + padh
                annots[:, 2] = annots[:, 2] + padw
                annots[:, 3] = annots[:, 3] + padh

            image_annots.append(annots)

        image_annots = np.concatenate(image_annots, axis=0)
        image_annots[:, 0:4] = np.clip(image_annots[:, 0:4], 0,
                                       int(self.resize * 2))

        image_annots = image_annots[image_annots[:, 2] -
                                    image_annots[:, 0] > 1]
        image_annots = image_annots[image_annots[:, 3] -
                                    image_annots[:, 1] > 1]

        scale = np.array(1.).astype(np.float32)
        size = np.array([int(self.resize * 2),
                         int(self.resize * 2)]).astype(np.float32)

        if self.use_multi_scale:
            combine_h, combine_w, _ = combined_img.shape
            scale_range = [
                int(self.multi_scale_range[0] * int(self.resize * 2)),
                int(self.multi_scale_range[1] * int(self.resize * 2))
            ]
            resize_list = [
                i // self.stride * self.stride
                for i in range(scale_range[0], scale_range[1] + self.stride)
            ]
            resize_list = list(set(resize_list))

            random_idx = np.random.randint(0, len(resize_list))
            final_resize = resize_list[random_idx]

            scale_factor = final_resize / max(combine_h, combine_w)
            resize_h, resize_w = math.ceil(
                combine_h * scale_factor), math.ceil(combine_w * scale_factor)
            combined_img = cv2.resize(combined_img, (resize_w, resize_h))

            image_annots[:, 0:4] *= scale_factor
            scale *= scale_factor
            size = np.array([combined_img.shape[0],
                             combined_img.shape[1]]).astype(np.float32)

        combine_h, combine_w, _ = combined_img.shape

        pad_w = 0 if combine_w % 32 == 0 else 32 - combine_w % 32
        pad_h = 0 if combine_h % 32 == 0 else 32 - combine_h % 32

        padded_image = np.zeros((combine_h + pad_h, combine_w + pad_w, 3),
                                dtype=np.uint8)

        padded_image[:combine_h, :combine_w, :] = combined_img
        padded_image = padded_image.astype(np.float32)
        image_annots = image_annots.astype(np.float32)

        sample = {
            'image': padded_image,
            'annots': image_annots,
            'scale': scale,
            'size': size,
        }

        if self.transform:
            sample = self.transform(sample)

        return sample


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

    import os
    import sys

    BASE_DIR = os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    sys.path.append(BASE_DIR)

    from tools.path import COCO2017_path

    import torchvision.transforms as transforms
    from tqdm import tqdm

    from simpleAICV.detection.common import RandomHorizontalFlip, RandomCrop, RandomTranslate, Normalize, YoloStyleResize, RetinaStyleResize, DetectionCollater

    cocodataset = CocoDetection(
        COCO2017_path,
        set_name='train2017',
        transform=transforms.Compose([
            RandomHorizontalFlip(prob=0.5),
            RandomCrop(prob=0.5),
            RandomTranslate(prob=0.5),
            # RetinaStyleResize(resize=400,
            #                   divisor=32,
            #                   stride=32,
            #                   multi_scale=True,
            #                   multi_scale_range=[0.8, 1.0]),
            YoloStyleResize(resize=640,
                            divisor=32,
                            stride=32,
                            multi_scale=False,
                            multi_scale_range=[0.5, 1.0]),
            # Normalize(),
        ]))

    count = 0
    for per_sample in tqdm(cocodataset):
        print('1111', per_sample['image'].shape, per_sample['annots'].shape,
              per_sample['scale'], per_sample['size'])
        print('1111', per_sample['image'].dtype, per_sample['annots'].dtype,
              per_sample['scale'].dtype, per_sample['size'].dtype)

        temp_dir = './temp'
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)

        image = np.ascontiguousarray(per_sample['image'], dtype=np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        annots = per_sample['annots']

        # draw all label boxes
        for per_annot in annots:
            per_box = (per_annot[0:4]).astype(np.int32)
            per_box_class_index = per_annot[4].astype(np.int32)
            class_name, class_color = COCO_CLASSES[
                per_box_class_index], COCO_CLASSES_COLOR[per_box_class_index]
            left_top, right_bottom = (per_box[0], per_box[1]), (per_box[2],
                                                                per_box[3])
            cv2.rectangle(image,
                          left_top,
                          right_bottom,
                          color=class_color,
                          thickness=2,
                          lineType=cv2.LINE_AA)

            text = f'{class_name}'
            text_size = cv2.getTextSize(text, 0, 0.5, thickness=1)[0]
            fill_right_bottom = (max(left_top[0] + text_size[0],
                                     right_bottom[0]),
                                 left_top[1] - text_size[1] - 3)
            cv2.rectangle(image,
                          left_top,
                          fill_right_bottom,
                          color=class_color,
                          thickness=-1,
                          lineType=cv2.LINE_AA)
            cv2.putText(image,
                        text, (left_top[0], left_top[1] - 2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color=(0, 0, 0),
                        thickness=1,
                        lineType=cv2.LINE_AA)

        cv2.imencode('.jpg', image)[1].tofile(
            os.path.join(temp_dir, f'idx_{count}.jpg'))

        if count < 5:
            count += 1
        else:
            break

    from torch.utils.data import DataLoader
    collater = DetectionCollater()
    train_loader = DataLoader(cocodataset,
                              batch_size=16,
                              shuffle=True,
                              num_workers=2,
                              collate_fn=collater)

    count = 0
    for data in tqdm(train_loader):
        images, annots, scales, sizes = data['image'], data['annots'], data[
            'scale'], data['size']
        print('2222', images.shape, annots.shape, scales.shape, sizes.shape)
        print('2222', images.dtype, annots.dtype, scales.dtype, sizes.dtype)

        if count < 5:
            count += 1
        else:
            break

    mosaiccocodataset = MosaicResizeCocoDetection(
        COCO2017_path,
        set_name='train2017',
        resize=640,
        stride=32,
        use_multi_scale=True,
        multi_scale_range=[0.5, 1.0],
        transform=transforms.Compose([
            RandomHorizontalFlip(prob=0.5),
            RandomCrop(prob=0.5),
            RandomTranslate(prob=0.5),
            # Normalize(),
        ]))

    count = 0
    for per_sample in tqdm(mosaiccocodataset):
        print('3333', per_sample['image'].shape, per_sample['annots'].shape,
              per_sample['scale'], per_sample['size'])
        print('3333', per_sample['image'].dtype, per_sample['annots'].dtype,
              per_sample['scale'].dtype, per_sample['size'].dtype)
        if count < 5:
            count += 1
        else:
            break

    from torch.utils.data import DataLoader
    collater = DetectionCollater()
    mosaic_train_loader = DataLoader(mosaiccocodataset,
                                     batch_size=4,
                                     shuffle=True,
                                     num_workers=2,
                                     collate_fn=collater)

    count = 0
    for data in tqdm(mosaic_train_loader):
        images, annots, scales, sizes = data['image'], data['annots'], data[
            'scale'], data['size']
        print('4444', images.shape, annots.shape, scales.shape, sizes.shape)
        print('4444', images.dtype, annots.dtype, scales.dtype, sizes.dtype)

        temp_dir = './temp'
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)

        images = images.permute(0, 2, 3, 1).cpu().numpy()
        annots = annots.cpu().numpy()

        for i, (per_image, per_image_annot) in enumerate(zip(images, annots)):
            per_image = np.ascontiguousarray(per_image, dtype=np.uint8)
            per_image = cv2.cvtColor(per_image, cv2.COLOR_RGB2BGR)

            # draw all label boxes
            for per_annot in per_image_annot:
                per_box = (per_annot[0:4]).astype(np.int32)
                per_box_class_index = per_annot[4].astype(np.int32)

                if per_box_class_index == -1:
                    continue

                class_name, class_color = COCO_CLASSES[
                    per_box_class_index], COCO_CLASSES_COLOR[
                        per_box_class_index]
                left_top, right_bottom = (per_box[0], per_box[1]), (per_box[2],
                                                                    per_box[3])
                cv2.rectangle(per_image,
                              left_top,
                              right_bottom,
                              color=class_color,
                              thickness=2,
                              lineType=cv2.LINE_AA)

                text = f'{class_name}'
                text_size = cv2.getTextSize(text, 0, 0.5, thickness=1)[0]
                fill_right_bottom = (max(left_top[0] + text_size[0],
                                         right_bottom[0]),
                                     left_top[1] - text_size[1] - 3)
                cv2.rectangle(per_image,
                              left_top,
                              fill_right_bottom,
                              color=class_color,
                              thickness=-1,
                              lineType=cv2.LINE_AA)
                cv2.putText(per_image,
                            text, (left_top[0], left_top[1] - 2),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            color=(0, 0, 0),
                            thickness=1,
                            lineType=cv2.LINE_AA)

            cv2.imencode('.jpg', per_image)[1].tofile(
                os.path.join(temp_dir, f'idx_{count}_{i}.jpg'))

        if count < 5:
            count += 1
        else:
            break