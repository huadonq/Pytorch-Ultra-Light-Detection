# author : zhaojihuai z50010333
import os
import argparse
from tools.subdir2rootdir import merge_json, json_list, image_dict
from tools.label2coco import labelme2coco
from tools.split_coco_tran_val import split_train_val
# import glob


parser = argparse.ArgumentParser(
        description="labelme annotation to coco data json file."
    )
parser.add_argument(
    "--root_dir", 
    type=str, 
    # default='/home/jovyan/qrcode-single-detection/0718_detection/',
    help="Directory to labelme images and annotation json files."
)
parser.add_argument(
    "--save_path",
    type=str,
    help="Directory to save images and annotation json files."
)

parser.add_argument(
    "--save_annotations", help="Output json file path.", default="trainval.json"
)
parser.add_argument('--train_ratio', type=float, dest='ratio_train', default='0.7', help='set train dataset ratio')
parser.add_argument('--valid_ratio', type=float,  dest='ratio_valid',default='0.3', help='set valid dataset ratio')
parser.add_argument('--trainJson_name', type=str, default='train/annotations/train.json', help='Where to store COCO training annotations')
parser.add_argument('--validJson_name', type=str, default='val/annotations/val.json', help='Where to store COCO valid annotations')

args = parser.parse_args()

def remove_file(path):
    for file in os.listdir(path):
        os.remove(os.path.join(path,file)) if not os.path.isdir(os.path.join(path,file)) else None

def main(args):
    args.save_annotations = os.path.join(args.save_path, args.save_annotations)
    args.trainJson_name = os.path.join(args.save_path, args.trainJson_name)
    args.validJson_name = os.path.join(args.save_path, args.validJson_name)

    os.mkdir(args.save_path) if not os.path.exists(args.save_path) else None
    print("🚀🚀🚀  merge files...  🚀🚀🚀")
    merge_json(args.root_dir)
    # merge_file(args.root_dir, args.save_path)
    print("🚀🚀🚀  convert to coco format...  🚀🚀🚀")
    # labelme_json = glob.glob(os.path.join(args.save_path, "*.json"))
    labelme2coco(json_list, args.save_annotations)
    print("🚀🚀🚀  split train val...  🚀🚀🚀")
    split_train_val(args, image_dict)
    # remove_file(args.save_path)
    print("😊😊😊  finish processing!!!  😊😊😊")

if __name__=='__main__':

    main(args)