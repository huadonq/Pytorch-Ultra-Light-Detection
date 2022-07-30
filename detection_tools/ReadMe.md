

# ReadMe
**Thanks to the author 😊ZhaoJihuai😊 for his dedication to this repository.**

**This repository only support Linux OS.**

**This repository is used for from labelme format detection or segmentation dataset convert to COCO dataset format.**

**And this repository will auto split train and val dataset.**

**environments:**
ubuntu1~18.04, Python Version:3.8.10

# Usage
You just need to change the original labelme data path and the save path in run.sh, then "sh run.sh" in the TERMINAL.

the "run.sh" contains:

```
python main.py --root_dir /home/data_source --save_path /home/data
```
the "root_dir" is your original data path, and i think you know what the "save_path", it's the save path, this folder contains COCO format data struct.

And you just "sh run.sh" in the TERMINAL.

```
sh run.sh
```
If you want to set train dataset and val dataset split ratio, please mod it in main.py, or add args in "run.sh".


# Prepare datasets
No matter how many subdir in your data root dir, you just need to make sure that the images and json files in each leaf subdir are placed together,
it's no problem even if there are subdir nested.


you need prepare labelme-format dataset and make sure the each leaf sub folder architecture as follows:
```
data
|
|----->sub1
|--------->sub2
|----------------->sub3
|---------------------------xxx.jpg
|---------------------------xxx.json
...
|---------------------------xxx.jpg
|---------------------------xxx.json
|----------------->sub4
|---------------------------xxx.jpg
|---------------------------xxx.json
...
|---------------------------xxx.jpg
|---------------------------xxx.json
|----->sub5
|--------->sub6
|----------------->xxx.jpg
|----------------->xxx.json
...
|----------------->xxx.jpg
|----------------->xxx.json
```

# Visual
If you want to visual coco-format annotations in single image, please replace visual_coco_dataset.py's relative path.

Or you want to visual detection result by coco-format, please replace detection_result_visual.py's relative path.

🙌To be honest, it is only support single-image processing, I'll code for batch-processing quickly in the future. 🙌

# Citation
**If you find my work useful in your research, please consider citing:**

```
@inproceedings{zjh,
 title={Ultra-Light-Detection-Tools},
 author={zjh},
 year={2022}
}
```