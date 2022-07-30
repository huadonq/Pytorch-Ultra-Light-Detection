import json
import os
import argparse
import funcy
import shutil
from sklearn.model_selection import train_test_split


def save_images(images, save_path, img_dict):
    save_path = os.path.join(save_path, 'images')
    # mksave_dir(save_path)
    os.makedirs(
        save_path, exist_ok=True
    )
    for img in images:
        img_name = img['file_name']
        source_path = img_dict[img_name]
        shutil.copy(os.path.join(source_path, img_name), os.path.join(save_path, img_name))

def save_coco(file, info, licenses, images, annotations, categories, img_dict):
    # mksave_dir(file)
    os.makedirs(
        os.path.dirname(file), exist_ok=True
    )
    with open(file, 'wt', encoding='UTF-8') as coco:
        json.dump({ 'info': info, 'licenses': licenses, 'images': images, 
            'annotations': annotations, 'categories': categories}, coco, indent=2, sort_keys=True)
    save_images(images, file.replace(file.split('/')[-1], '/').replace('/annotations/', '/'), img_dict)


def filter_annotations(annotations, images):
    image_ids = funcy.lmap(lambda i: int(i['id']), images)
    return funcy.lfilter(lambda a: int(a['image_id']) in image_ids, annotations)

def split_train_val(args, img_dict):
    ratio_train = args.ratio_train
    ratio_valid = args.ratio_valid
    with open(args.save_annotations, 'rt', encoding='UTF-8') as annotations:
        coco = json.load(annotations)
        info = None
        licenses = None
        images = coco['images']
        annotations = coco['annotations']
        categories = coco['categories']

        number_of_images = len(images)

        images_with_annotations = funcy.lmap(lambda a: int(a['image_id']), annotations)

        if args.save_annotations:
            images = funcy.lremove(lambda i: i['id'] not in images_with_annotations, images)

        train_before, valid = train_test_split(
            images, test_size=ratio_valid)

        # ratio_remaining = 1 - ratio_test
        # ratio_valid_adjusted = ratio_valid / ratio_remaining

        # train_after, valid = train_test_split(
        #     train_before, test_size=ratio_valid_adjusted)

        save_coco(args.trainJson_name, info, licenses, train_before, filter_annotations(annotations, train_before), categories, img_dict)
        # save_coco(args.testJson_name, info, licenses, test, filter_annotations(annotations, test), categories)
        save_coco(args.validJson_name, info, licenses, valid, filter_annotations(annotations, valid), categories, img_dict)



if __name__ == "__main__":
    split_train_val(args)