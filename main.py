import os

import nibabel as nib
import matplotlib.pyplot as plt
from nibabel.viewers import OrthoSlicer3D
import numpy as np


def test(f):
    img1 = nib.load(f)
    # 打印文件信息
    # print(img1)
    # OrthoSlicer3D(img1.dataobj).show()
    data = img1.get_data()
    print(data.shape)
    idx = data.shape[1] // 2
    clip = data[:, idx, :]
    plt.imshow(clip, cmap=plt.cm.gray)
    plt.show()


def show_all(path='./data/mni'):
    for i in os.listdir(path):
        print(i)
        name = os.path.join(path, i)
        test(name)


if __name__ == '__main__':
    show_all('./data/AD')
