# append root
import os
import sys
a = (os.path.abspath(sys.argv[0])[0: os.path.abspath(sys.argv[0]).find('sparse_sharing')])
b = 'sparse_sharing'
root = os.path.join(a, b)
sys.path.append(root)
# import pytorch / fastai
import torch
from fastai.vision.all import get_image_files, ImageBlock, CategoryBlock, RegressionBlock, DataBlock, Resize, aug_transforms, Rotate
# misc
from os import path
import pandas as pd
import matplotlib.pyplot as plt


img_location = path.join(root, 'data', '2classification', 'imgs')  # /Users/mazeyu/newEraFrom2020.8/sparse_sharing/data/2classification/imgs
label_csv_path = path.join(root, 'data', '2classification', 'label.csv')  # /Users/mazeyu/newEraFrom2020.8/sparse_sharing/data/2classification/label.csv
df = pd.read_csv(label_csv_path)


def get_y_from_name(fname):
    """
    根据名字获取label
    :param fname: 某张图像的绝对路径
    :return: 类别的参数
    """
    fname = path.basename(fname)
    return fname.split('_')[0]


def get_y_from_df(fname):
    """
    根据名字从df中找到label
    :param fname: 某张图像的绝对路径
    :return: csv中对称性
    """
    fname = path.basename(fname)
    return df[df.img_name==fname]['label'].values[0]

data_block_512 = DataBlock(blocks = (ImageBlock, CategoryBlock, RegressionBlock),
                       get_items  = get_image_files,
                       get_y      = [get_y_from_name, get_y_from_df],
                       n_inp      = 1,
                       item_tfms  = Resize(512),
                       batch_tfms = aug_transforms(batch=True)
)

data_block_224 = DataBlock(blocks = (ImageBlock, CategoryBlock, RegressionBlock),
                       get_items  = get_image_files,
                       get_y      = [get_y_from_name, get_y_from_df],
                       n_inp      = 1,
                       item_tfms  = Resize(224),
                       batch_tfms = aug_transforms(batch=True)
)

data_block_312 = DataBlock(blocks = (ImageBlock, CategoryBlock, RegressionBlock),
                       get_items  = get_image_files,
                       get_y      = [get_y_from_name, get_y_from_df],
                       n_inp      = 1,
                       item_tfms  = [Resize(312)],
                       batch_tfms = aug_transforms(batch=True)
)

dataloader_s512_bs4 = data_block_512.dataloaders(img_location, bs=4)  # {'melanoma': 0, 'nevus': 1} imgs, label1s, label2s

dataloader_s224_bs16 = data_block_224.dataloaders(img_location, bs=16)

dataloader_s312_bs12 = data_block_312.dataloaders(img_location, bs=12)

# dataloader_s312_bs12.show_batch()
# plt.show()