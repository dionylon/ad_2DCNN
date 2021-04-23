import os

import PIL
import nibabel
import numpy
import torch
from PIL import Image
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import cv2
from nibabel.filebasedimages import ImageFileError
from torchvision.transforms import transforms


augments = [
    # ('h_flip',  transforms.RandomHorizontalFlip(p=1)),
    # ('v_flip',  transforms.RandomVerticalFlip(p=1))
]


def slice_image(src_dir='./data/mni/ad', target_dir='./data/mni/ad'):
    """
    读取3D MRI,切片形成图像，保存图像到指定目录
    """
    for i, f in enumerate(os.listdir(src_dir)):
        name = os.path.join(src_dir, f)
        image = nibabel.load(name)
        data = image.get_data()
        print(name)
        print(data.shape)
        x, y, z = data.shape
        # ch1 = data[:, y//2, :]
        # ch2 = data[:, y//2, :]
        # ch3 = data[:, y//2, :]
        ch1 = data[x//2, :, :]
        ch2 = data[:, y//2, :]
        ch3 = data[:, :, z//2]
        ch1 = cv2.resize(ch1, (224, 224))
        ch2 = cv2.resize(ch2, (224, 224))
        ch3 = cv2.resize(ch3, (224, 224))
        img = numpy.stack((ch1, ch2, ch3))
        img = torch.from_numpy(img)
        # img = img / img.max()
        img = transforms.ToPILImage()(img)
        img.save(os.path.join(target_dir, '{}.jpg'.format(i)))
        for transform_name, transform in augments:
            print(transform_name)
            t_img = transform(img)
            t_img.save(os.path.join(target_dir, '{}_{}.jpg'.format(i, transform_name)))


if __name__ == '__main__':
    slice_image('./data/wmp1/ad', './data/img/ad')
    slice_image('./data/wmp1/cn', './data/img/cn')
