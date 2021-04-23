import os

import cv2
import numpy as numpy
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader

import nibabel as nib
from torchvision.transforms import transforms

from utils import show_from_tensor, np_to_tensor


class SliceDataSet(Dataset):
    def __init__(self, root_dir='./data/train', transforms=transforms.Compose([
        transforms.ToTensor()
    ])):
        super().__init__()
        self.transforms = transforms
        self.root_dir = root_dir
        self.data = []
        self.load_data()

    def __getitem__(self, item):
        path, label = self.data[item]
        img = Image.open(path)
        img = self.transforms(img)
        return img, label

    def __len__(self):
        return len(self.data)

    def load_data(self):
        ad_dir = os.path.join(self.root_dir, 'ad')
        cn_dir = os.path.join(self.root_dir, 'cn')
        for file in os.listdir(ad_dir):
            self.data.append((os.path.join(ad_dir, file), 1))
        for file in os.listdir(cn_dir):
            self.data.append((os.path.join(cn_dir, file), 0))


if __name__ == '__main__':
    dataset = SliceDataSet()
    loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=3)
    for i, data in enumerate(loader):
        img, label = data
        print(img.shape)
        print(label.shape)
        break
