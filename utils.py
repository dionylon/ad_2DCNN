import os

import torch
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import torch
import matplotlib.pyplot as plt
from torchvision.transforms import transforms

# loader使用torchvision中自带的transforms函数

loader = transforms.Compose([
    transforms.ToTensor()])

unloader = transforms.ToPILImage()


# 输入图片地址
# 返回tensor变量
def image_loader(image_name):
    image = Image.open(image_name).convert('RGB')
    image = loader(image).unsqueeze(0)
    return image.to(torch.float)


# 输入PIL格式图片
# 返回tensor变量
def PIL_to_tensor(image):
    image = loader(image).unsqueeze(0)
    return image.to(torch.float)


# 输入tensor变量
# 输出PIL格式图片
def tensor_to_PIL(tensor):
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = unloader(image)
    return image


def imshow(tensor, title=None):
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)  # remove the fake batch dimension
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.show()  # pause a bit so that plots are updated


def save_image(tensor, **para):
    dir = 'results'
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)  # remove the fake batch dimension
    image = unloader(image)
    if not os.path.exists(dir):
        os.makedirs(dir)
    image.save('results/s{}-c{}-l{}-e{}-sl{:4f}-cl{:4f}.jpg'
               .format(para['style_weight'], para['content_weight'], para['lr'], para['epoch'],
                       para['style_loss'], para['content_loss']))


def tensor_to_np(tensor):
    img = tensor.mul(255).byte()
    img = img.cpu().numpy().squeeze().transpose((1, 2, 0))
    return img


def show_from_cv(img, title=None):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure()
    plt.imshow(img)
    if title is not None:
        plt.title(title)
    plt.show()


def show_from_tensor(tensor, title=None):
    img = tensor.clone()
    img = tensor_to_np(img)
    plt.figure()
    plt.imshow(img)
    if title is not None:
        plt.title(title)
    plt.show()


def np_to_tensor(img):
    img = torch.from_numpy(img.transpose((2, 0, 1)))
    return img.float().div(255)