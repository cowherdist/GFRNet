# Pytorch Code for GFRNet GFRNet
# Use archs.py to replace the network model part of your own project 
# Use losses.py to replace  your own loss.


## Using the code:

The code is stable while using Python 3.6.13, CUDA >=10.1

- Clone this repository：

To install all the dependencies using conda:

```bash
conda env create -f environment.yml
conda activate GFRNet
```

If you prefer pip, install following versions:

```bash
timm==0.3.2
mmcv-full==1.2.7
torch==1.7.1
torchvision==0.8.2
opencv-python==4.5.1.48
```

## Datasets

1) ISIC 2018 - [Link](https://challenge.isic-archive.com/data/)
2) ISIC 2016

## Data Format

Make sure to put the files as the following structure (e.g. the number of classes is 2):

```
inputs
└── <dataset name>
    ├── images
    |   ├── 001.png
    │   ├── 002.png
    │   ├── 003.png
    │   ├── ...
    |
    └── masks
        ├── 0
        |   ├── 001.png
        |   ├── 002.png
        |   ├── 003.png
        |   ├── ...
        |
        └── 1
            ├── 001.png
            ├── 002.png
            ├── 003.png
            ├── ...
```

For binary segmentation problems, just use folder 0.

## Training and Validation

1. Train the model.
```
python train.py --dataset <dataset name> --arch GFRNet --name <exp name> --img_ext .png --mask_ext .png --lr 0.0001 --epochs 400 --input_w 224 --input_h 224 --b 16
```
2. Evaluate.
```
python val.py --name <exp name>
```

### Acknowledgements:

This code-base uses certain code-blocks and helper functions from [UNet++](https://github.com/4uiiurz1/pytorch-nested-unet), [Segformer](https://github.com/NVlabs/SegFormer), and [AS-MLP](https://github.com/svip-lab/AS-MLP). Naming credits to [Poojan](https://scholar.google.co.in/citations?user=9dhBHuAAAAAJ&hl=en).


# for more details or other relevant information such as datasets form, Please refer to the paper.
# You can also contact us by email "shanglaimail@foxmail.com"
