import argparse
import json
import os

import torch
import torch.nn
from torch import nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, random_split

from dataset import SliceDataSet
from model import AlexNet


def train(args, model, device, loader, loss_fn, optimizer):
    model.train()
    loss_record = []
    for i, data in enumerate(loader):
        # for param in model.parameters():
        #     print(param.max())
        # print('=================')
        image, label = data
        image, label = image.to(device), label.to(device)
        output = model(image)
        loss = loss_fn(output, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_record.append(loss.item())

        if (i + 1) % args.log_interval == 0 or i == len(loader) - 1:
            print('Training {}/{} : {:.2f}% of total, Loss:{:.5f}'.format(
                i + 1, len(loader), (i + 1) * 100.0 / len(loader), loss.item()
            ))
        if args.single_run:
            break
    return loss_record


def validate(args, model, device, loader, loss_fn):
    model.eval()
    total_loss = 0.0
    correct = 0
    with torch.no_grad():
        for i, data in enumerate(loader):
            image, label = data
            image, label = image.to(device), label.to(device)
            output = model(image)

            loss = loss_fn(output, label)
            total_loss += loss.item()

            predict = output.argmax(dim=1)
            correct += predict.eq(label.view_as(predict)).sum().item()

            if (i + 1) % args.log_interval == 0 or i == len(loader) - 1:
                print('Testing {}/{} : {:.2f}% of total, Loss:{:.5f}'.format(
                    i + 1, len(loader), (i + 1) * 100.0 / len(loader), loss.detach().item()
                ))
            if args.single_run:
                break
    print('Correct : {} / {} = {:.5f}%, Avg Loss = {:.5f}.'.format(
            correct, len(loader.dataset), 100.0 * correct / len(loader.dataset), total_loss / len(loader)
        )
    )


def get_loaders(args):
    dataset = SliceDataSet()
    train_size = int(args.train_rate * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=args.batch_size,
                              num_workers=args.num_workers,
                              shuffle=args.shuffle)
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=args.batch_size,
                             num_workers=args.num_workers,
                             shuffle=args.shuffle)
    return train_loader, test_loader


def main():
    parser = argparse.ArgumentParser(description="Train Recounting Model")
    parser.add_argument("--epoch", help="Training epoch", type=int, default=20)
    parser.add_argument("--batch_size", help="Training batch size", type=int, default=4)
    parser.add_argument("--save_model", help="Whether to save the model", action="store_true", default=True)
    parser.add_argument("--no_cuda", help="train without cuda", action='store_true', default=False)
    parser.add_argument("--log_interval", help="Log interval for training", type=int, default=5)
    parser.add_argument("--num_workers", help="Num of workers", type=int, default=3)
    parser.add_argument("--shuffle", action="store_true", default=True)
    parser.add_argument("--single_run", help="Run a single pass", action="store_true", default=False)
    parser.add_argument("--lr", help="Learning rate", type=float, default=0.001)
    parser.add_argument("--gamma", help="lr scheduler gamma", default=0.9)
    parser.add_argument("--step_size", help="lr step size", type=int, default=5)
    parser.add_argument("--train_rate", help="split the train size and test size", type=float, default=0.8)
    parser.add_argument("--pre_load", help="Load pre-trained model", action='store_true', default=True)
    parser.add_argument("--save_path", help="model path", default="./save/model.pkl")
    parser.add_argument("--cv", help="K-fold cross validate", type=int, default=5)
    parser.add_argument("--record_loss", help="record loss", type=bool, default=True)
    args = parser.parse_args()

    model: AlexNet = AlexNet()
    if args.pre_load and os.path.isfile(args.save_path):
        print('Loading model')
        model = torch.load(args.save_path)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    loss_fn = nn.CrossEntropyLoss()
    use_cuda = torch.cuda.is_available() and not args.no_cuda
    device = torch.device("cuda" if use_cuda else "cpu")
    print('Device: ', device)

    model = model.to(device)
    print(args)

    total_losses = []
    for epoch in range(1, args.epoch + 1):
        print('\nTraining epoch {}'.format(epoch))
        data_loader, test_loader = get_loaders(args)
        loss_record = train(args, model, device, data_loader, loss_fn, optimizer)
        if args.record_loss:
            total_losses.extend(loss_record)
        validate(args, model, device, test_loader, loss_fn)
        scheduler.step()

    if args.record_loss:
        print('Writing file..')
        with open('./save/loss.json', 'w+') as fp:
            json.dump(total_losses, fp)

    if args.save_model:
        print('Saving model..')
        torch.save(model, args.save_path)
        print('Done')


if __name__ == '__main__':
    main()
