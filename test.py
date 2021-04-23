import os

import numpy as np
import torch
from matplotlib import pyplot
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, auc
from dataset import SliceDataSet
from model import AlexNet

root_dir = './data/test'
model_save_path = './save/model.pkl'


def test():
    model: AlexNet = torch.load(model_save_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = SliceDataSet(root_dir=root_dir)
    loader = DataLoader(dataset, shuffle=True, batch_size=1, num_workers=3, drop_last=False)
    model = model.to(device)
    model.eval()
    score_list = []
    label_list = []
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    with torch.no_grad():
        for i, data in enumerate(loader):
            image, label = data
            image, label = image.to(device), label.to(device)
            output = model(image)
            predict = output.argmax(dim=1)

            TP += ((predict == 1) & (label == 1)).cpu().sum()
            TN += ((predict == 0) & (label == 0)).cpu().sum()
            FN += ((predict == 0) & (label == 1)).cpu().sum()
            FP += ((predict == 1) & (label == 0)).cpu().sum()
            score_list.append(output[0][label[0]].detach().cpu().numpy())
            label_list.extend(label.cpu().numpy())
    total = TP + TN + FN + FP
    assert total == len(dataset)

    fpr, tpr, _ = roc_curve(label_list, score_list)
    AUC = auc(fpr, tpr)
    print(f'准确率: {TP + TN}/{total} = {100.0 * (TP + TN) / total:.5f}\n'
          f'特异性: {TN}/{(TN + FP)} = {100.0 * TN / (TN + FP):.5f}\n'
          f'AUC: {AUC}\n')
    pyplot.plot(fpr, tpr, lw=1, label=f'AD vs CN, area={AUC:.2f})')

    pyplot.xlim([0.00, 1.0])
    pyplot.ylim([0.00, 1.0])
    pyplot.xlabel("False Positive Rate")
    pyplot.ylabel("True Positive Rate")
    pyplot.title("ROC")
    pyplot.legend(loc="lower right")
    pyplot.savefig(r"./save/ROC.png")


if __name__ == '__main__':
    test()
