import cv2
import numpy as np

import torch
from collections import OrderedDict

class RetinaStyleResize:

    def __init__(self,
                 resize=400,
                 divisor=32,
                 stride=32,
                 multi_scale=False,
                 multi_scale_range=[0.8, 1.0]):
        self.resize = resize
        self.divisor = divisor
        self.stride = stride
        self.multi_scale = multi_scale
        self.multi_scale_range = multi_scale_range
        self.ratio = 1333. / 800

    def __call__(self, sample):
        '''
        sample must be a dict,contains 'image'、'annots'、'scale' keys.
        '''
        image, annots, scale, size = sample['image'], sample['annots'], sample[
            'scale'], sample['size']
        h, w, _ = image.shape

        if self.multi_scale:
            scale_range = [
                int(self.multi_scale_range[0] * self.resize),
                int(self.multi_scale_range[1] * self.resize)
            ]
            resize_list = [
                i // self.stride * self.stride
                for i in range(scale_range[0], scale_range[1] + self.stride)
            ]
            resize_list = list(set(resize_list))

            random_idx = np.random.randint(0, len(resize_list))
            scales = (resize_list[random_idx],
                      int(round(self.resize * self.ratio)))
        else:
            scales = (self.resize, int(round(self.resize * self.ratio)))

        max_long_edge, max_short_edge = max(scales), min(scales)
        factor = min(max_long_edge / max(h, w), max_short_edge / min(h, w))

        resize_h, resize_w = int(round(h * factor)), int(round(w * factor))
        image = cv2.resize(image, (resize_w, resize_h))

        pad_w = 0 if resize_w % self.divisor == 0 else self.divisor - resize_w % self.divisor
        pad_h = 0 if resize_h % self.divisor == 0 else self.divisor - resize_h % self.divisor

        padded_image = np.zeros((resize_h + pad_h, resize_w + pad_w, 3),
                                dtype=np.float32)
        padded_image[:resize_h, :resize_w, :] = image

        factor = np.float32(factor)
        annots[:, :4] *= factor
        scale *= factor

        return {
            'image': padded_image,
            'annots': annots,
            'scale': scale,
            'size': size,
        }


class YoloStyleResize:

    def __init__(self,
                 resize=640,
                 divisor=32,
                 stride=32,
                 multi_scale=False,
                 multi_scale_range=[0.5, 1.0]):
        self.resize = resize
        self.divisor = divisor
        self.stride = stride
        self.multi_scale = multi_scale
        self.multi_scale_range = multi_scale_range

    def __call__(self, sample):
        '''
        sample must be a dict,contains 'image'、'annots'、'scale' keys.
        '''
        image, annots, scale, size = sample['image'], sample['annots'], sample[
            'scale'], sample['size']
        h, w, _ = image.shape

        if self.multi_scale:
            scale_range = [
                int(self.multi_scale_range[0] * self.resize),
                int(self.multi_scale_range[1] * self.resize)
            ]
            resize_list = [
                i // self.stride * self.stride
                for i in range(scale_range[0], scale_range[1] + self.stride)
            ]
            resize_list = list(set(resize_list))

            random_idx = np.random.randint(0, len(resize_list))
            final_resize = resize_list[random_idx]
        else:
            final_resize = self.resize

        factor = final_resize / max(h, w)

        resize_h, resize_w = int(round(h * factor)), int(round(w * factor))
        image = cv2.resize(image, (resize_w, resize_h))

        pad_w = 0 if resize_w % self.divisor == 0 else self.divisor - resize_w % self.divisor
        pad_h = 0 if resize_h % self.divisor == 0 else self.divisor - resize_h % self.divisor

        padded_image = np.zeros((resize_h + pad_h, resize_w + pad_w, 3),
                                dtype=np.float32)
        padded_image[:resize_h, :resize_w, :] = image

        factor = np.float32(factor)
        annots[:, :4] *= factor
        scale *= factor

        return {
            'image': padded_image,
            'annots': annots,
            'scale': scale,
            'size': size,
        }


class RandomHorizontalFlip:

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, sample):
        '''
        sample must be a dict,contains 'image'、'annots'、'scale' keys.
        '''
        image, annots, scale, size = sample['image'], sample['annots'], sample[
            'scale'], sample['size']

        if annots.shape[0] == 0:
            return sample

        if np.random.uniform(0, 1) < self.prob:
            image = image[:, ::-1, :]
            _, w, _ = image.shape

            x1 = annots[:, 0].copy()
            x2 = annots[:, 2].copy()

            annots[:, 0] = w - x2
            annots[:, 2] = w - x1

        return {
            'image': image,
            'annots': annots,
            'scale': scale,
            'size': size,
        }


class RandomCrop:

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, sample):
        '''
        sample must be a dict,contains 'image'、'annots'、'scale' keys.
        '''
        image, annots, scale, size = sample['image'], sample['annots'], sample[
            'scale'], sample['size']

        if annots.shape[0] == 0:
            return sample

        if np.random.uniform(0, 1) < self.prob:
            h, w, _ = image.shape
            max_bbox = np.concatenate([
                np.min(annots[:, 0:2], axis=0),
                np.max(annots[:, 2:4], axis=0)
            ],
                                      axis=-1)
            max_left_trans, max_up_trans = max_bbox[0], max_bbox[1]
            max_right_trans, max_down_trans = w - max_bbox[2], h - max_bbox[3]
            crop_xmin = max(
                0, int(max_bbox[0] - np.random.uniform(0, max_left_trans)))
            crop_ymin = max(
                0, int(max_bbox[1] - np.random.uniform(0, max_up_trans)))
            crop_xmax = min(
                w, int(max_bbox[2] + np.random.uniform(0, max_right_trans)))
            crop_ymax = min(
                h, int(max_bbox[3] + np.random.uniform(0, max_down_trans)))

            image = image[crop_ymin:crop_ymax, crop_xmin:crop_xmax]
            annots[:, [0, 2]] = annots[:, [0, 2]] - crop_xmin
            annots[:, [1, 3]] = annots[:, [1, 3]] - crop_ymin

        return {
            'image': image,
            'annots': annots,
            'scale': scale,
            'size': size,
        }


class RandomTranslate:

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, sample):
        '''
        sample must be a dict,contains 'image'、'annots'、'scale' keys.
        '''
        image, annots, scale, size = sample['image'], sample['annots'], sample[
            'scale'], sample['size']

        if annots.shape[0] == 0:
            return sample

        if np.random.uniform(0, 1) < self.prob:
            h, w, _ = image.shape
            max_bbox = np.concatenate([
                np.min(annots[:, 0:2], axis=0),
                np.max(annots[:, 2:4], axis=0)
            ],
                                      axis=-1)
            max_left_trans, max_up_trans = max_bbox[0], max_bbox[1]
            max_right_trans, max_down_trans = w - max_bbox[2], h - max_bbox[3]
            tx = np.random.uniform(-(max_left_trans - 1),
                                   (max_right_trans - 1))
            ty = np.random.uniform(-(max_up_trans - 1), (max_down_trans - 1))
            M = np.array([[1, 0, tx], [0, 1, ty]])
            image = cv2.warpAffine(image, M, (w, h))
            annots[:, [0, 2]] = annots[:, [0, 2]] + tx
            annots[:, [1, 3]] = annots[:, [1, 3]] + ty

        return {
            'image': image,
            'annots': annots,
            'scale': scale,
            'size': size,
        }


class Normalize:

    def __init__(self):
        pass

    def __call__(self, sample):
        '''
        sample must be a dict,contains 'image'、'annots'、'scale' keys.
        '''
        image, annots, scale, size = sample['image'], sample['annots'], sample[
            'scale'], sample['size']

        image = image / 255.

        return {
            'image': image,
            'annots': annots,
            'scale': scale,
            'size': size,
        }


class DetectionCollater:

    def __init__(self, divisor=32):
        self.divisor = divisor

    def __call__(self, data):
        images = [s['image'] for s in data]
        annots = [s['annots'] for s in data]
        scales = [s['scale'] for s in data]
        sizes = [s['size'] for s in data]

        max_h = max(image.shape[0] for image in images)
        max_w = max(image.shape[1] for image in images)

        pad_h = 0 if max_h % self.divisor == 0 else self.divisor - max_h % self.divisor
        pad_w = 0 if max_w % self.divisor == 0 else self.divisor - max_w % self.divisor

        input_images = np.zeros((len(images), max_h + pad_h, max_w + pad_w, 3),
                                dtype=np.float32)
        for i, image in enumerate(images):
            input_images[i, 0:image.shape[0], 0:image.shape[1], :] = image
        input_images = torch.from_numpy(input_images)
        # B H W 3 ->B 3 H W
        input_images = input_images.permute(0, 3, 1, 2)

        max_annots_num = max(annot.shape[0] for annot in annots)
        if max_annots_num > 0:
            input_annots = np.ones(
                (len(annots), max_annots_num, 5), dtype=np.float32) * (-1)
            for i, annot in enumerate(annots):
                if annot.shape[0] > 0:
                    input_annots[i, :annot.shape[0], :] = annot
        else:
            input_annots = np.ones(
                (len(annots), 1, 5), dtype=np.float32) * (-1)

        input_annots = torch.from_numpy(input_annots)

        scales = np.array(scales, dtype=np.float32)
        sizes = np.array(sizes, dtype=np.float32)

        return {
            'image': input_images,
            'annots': input_annots,
            'scale': scales,
            'size': sizes,
        }


class DetectionDataPrefetcher:

    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            sample = next(self.loader)
            self.next_input, self.next_annot = sample['image'], sample[
                'annots']
        except StopIteration:
            self.next_input = None
            self.next_annot = None
            return
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_annot = self.next_annot.cuda(non_blocking=True)
            self.next_input = self.next_input.float()

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        inputs = self.next_input
        annots = self.next_annot
        self.preload()

        return inputs, annots


def load_state_dict(saved_model_path, model, excluded_layer_name=()):
    '''
    saved_model_path: a saved model.state_dict() .pth file path
    model: a new defined model
    excluded_layer_name: layer names that doesn't want to load parameters
    only load layer parameters which has same layer name and same layer weight shape

    '''

    if not saved_model_path:
        print('💥💥💥 No pretrained model file! 💥💥💥')
        return
    
    saved_state_dict = torch.load(saved_model_path,
                                    map_location=torch.device('cpu'))

    if not isinstance(saved_state_dict, OrderedDict):
        for key, value in saved_state_dict.items():
            if isinstance(value, OrderedDict):
                saved_state_dict = value
                break
        
        filtered_state_dict = {
            name.split('module.')[-1]: weight
            for name, weight in saved_state_dict.items()
            if name.split('module.')[-1] in model.state_dict() and not any(
                excluded_name in name.split('module.')[-1] for excluded_name in excluded_layer_name)
            and weight.shape == model.state_dict()[name.split('module.')[-1]].shape
        }
    else:
        filtered_state_dict = {
            name: weight
            for name, weight in saved_state_dict.items()
            if name in model.state_dict() and not any(
                excluded_name in name for excluded_name in excluded_layer_name)
            and weight.shape == model.state_dict()[name].shape
        }

    
    if len(filtered_state_dict) == 0:
        print('💥💥💥  No pretrained parameters to load!  💥💥💥')
    else:
        print('😊😊😊  Weight parameters load successfully!  😊😊😊')
        model.load_state_dict(filtered_state_dict, strict=False)

    return


# def load_state_dict(saved_model_path, model, excluded_layer_name=()):
#     '''
#     saved_model_path: a saved model.state_dict() .pth file path
#     model: a new defined model
#     excluded_layer_name: layer names that doesn't want to load parameters
#     only load layer parameters which has same layer name and same layer weight shape
#     '''
#     if not saved_model_path:
#         print('No pretrained model file!')
#         return

#     saved_state_dict = torch.load(saved_model_path,
#                                   map_location=torch.device('cpu'))

#     filtered_state_dict = {
#         name: weight
#         for name, weight in saved_state_dict.items()
#         if name in model.state_dict() and not any(
#             excluded_name in name for excluded_name in excluded_layer_name)
#         and weight.shape == model.state_dict()[name].shape
#     }

#     if len(filtered_state_dict) == 0:
#         print('No pretrained parameters to load!')
#     else:
#         model.load_state_dict(filtered_state_dict, strict=False)

#     return