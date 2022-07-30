

# Prepare datasets

If you want to reproduce my pretrained models, you need download coco-format dataset and make sure the folder architecture as follows:
```
data
|
|----->train
|----------------->images
|---------------------------xxx.jpg
...
|---------------------------xxx.jpg
|----------------->annotations
|---------------------------train.json
...
|----->val
|----------------->images
|---------------------------xxx.jpg
...
|---------------------------xxx.jpg
|----------------->annotations
|---------------------------val.json

```
# Environments

**This repository only support DDP training.**

**environments:**
ubuntu1~18.04, 4*Tesla V100, Python Version:3.8.10, CUDA Version:11.4

Please make sure your Python version>=3.7.
**Use pip or conda to install those Packages:**
```
torch==1.10.0
torchvision==0.11.1
torchaudio==0.10.0
onnx==1.11.0
onnx-simplifier==0.3.6
numpy
Cython
pycocotools
opencv-python
tqdm
thop
yapf
tensorboard
apex

```

**How to install apex?**

apex needs to be installed separately.For torch1.10,modify apex/apex/amp/utils.py:
```
if cached_x.grad_fn.next_functions[1][0].variable is not x:
```
to
```
if cached_x.grad_fn.next_functions[0][0].variable is not x:
```

Then use the following orders to install apex:
```
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir ./
```
Using apex to train can reduce video memory usage by 25%-30%, but the training speed will be slower, the trained model has the same performance as not using apex.


# Train and test model

If you want to train or test model,you need enter a training folder directory,then run train.sh and test.sh.

For example,you can enter experiments/res50_retinanet_retinaresize400.
If you want to train this model from scratch, please delete checkpoints and log folders first,then run train.sh:
```
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --nproc_per_node=2 --master_addr 127.0.1.0 --master_port 10000 ../../tools/train.py  --work-dir ./
```

CUDA_VISIBLE_DEVICES is used to specify the gpu ids for this training.Please make sure the number of nproc_per_node equal to the number of gpu cards.
Make sure master_addr/master_port are unique for each training.

if you want to test this model,you need have a pretrained model first,modify pre_model_path in test_config.py,then run test.sh:
```
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --nproc_per_node=2 --master_addr 127.0.1.0 --master_port 10000 ../../tools/test.py --work-dir ./
```
Also, You can modify super parameters in train_config.py/test_config.py.

if you want to convert this model to jit, you need have a pretrained model first, modify pre_model_path and save_jit_path in test_config.py,then run torch2jit.sh:
```
python ../../tools/torch2jit.py --work-dir ./
```
Also, You can modify super parameters in test_config.py.

if you want to convert this model to onnx, you need have a pretrained model first, modify pre_model_path and save_onnx_path in test_config.py,then run torch2onnx.sh:
```
python ../../tools/torch2onnx.py --work-dir ./
```
Also, You can modify super parameters in test_config.py.

# Detection training results

## Training results


to be continue...

You can find more model training details in experiments.


# Distillation training results


**KD loss**
Paper:https://arxiv.org/abs/1503.02531

**DKD loss**
Paper:https://arxiv.org/abs/2203.08679

**DML loss**
Paper:https://arxiv.org/abs/1706.00384

to be continue...

# Citation
**If you find my work useful in your research, please consider citing:**

```
@inproceedings{huadonq,
 title={Pytorch-Ultra-Light-Detection},
 author={huadonq},
 year={2022}
}
```