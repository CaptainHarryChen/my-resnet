import os
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10

import matplotlib.pyplot as plt
import numpy as np

from MiniImagenet import MiniImagenet
from resnet import Resnet18


def eval(model, data, device, batch_size=32):
    data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size)

    correct_num = 0
    loss = 0.0
    model.eval()
    pbar = tqdm(total=len(data))
    for X, label in data_loader:
        X = X.to(device)
        label = label.to(device)

        pred = model(X)
        loss += (F.nll_loss(pred, label) * batch_size).item()
        pred = torch.argmax(pred, dim=-1)
        correct_num += (pred == label).sum()

        pbar.update(X.shape[0])
    pbar.close()
    return correct_num, loss/len(data)


device = "cuda:0"
model_name = "model_60000.pth"
dataset_name = "cifar10"
split = "train"


if __name__ == "__main__":
    if dataset_name == "mini-imagenet":
        transform = transforms.Compose([transforms.Resize([224,224]), transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])])
        data = MiniImagenet('data\mini-imagenet', split=split, transform=transform)
        class_num = train_data.class_num
    if dataset_name == "cifar10":
        transform = transforms.Compose([transforms.ToTensor(), transforms.Resize([224,224]), transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])])
        data = CIFAR10('data', train=(split=="train"), transform=transform)
        class_num = len(data.classes)

    device = torch.device(device)
    model = Resnet18(class_num).to(device)
    model.load_state_dict(torch.load(os.path.join("model_files", model_name)), strict=True)

    print(f"Eval {model_name}  ...")
    correct, loss = eval(model, data, device)
    total = len(data)
    print(f"Accuracy: {correct}/{total}    {correct/total*100.0:.3f}%")
    print(f"Loss: {loss:.5f}")
