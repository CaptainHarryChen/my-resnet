import os

import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10

import matplotlib.pyplot as plt
import numpy as np

from MiniImagenet import MiniImagenet
from resnet import Resnet18
from evaluation import eval


start_lr = 0.01
train_iterations = 60000
save_per_iter = 1000
min_delta = 0.001
device = "cuda:0"
dataset_name = "cifar10"
record_name = "cifar_record_1.pth"


def save_model(model, filename:str):
    if not os.path.exists("model_files"):
        os.makedirs("model_files")
    torch.save(model.state_dict(), os.path.join("model_files", filename))
    print(f"Saved '{filename}'")


def update_learing_rate(train_losses, optimizer):
    min_loss = min(train_losses[-save_per_iter:])
    if train_losses[-1] > min_loss - min_delta:
        origin_lr = next(iter(optimizer.param_groups))['lr']
        now_lr = origin_lr / 10.0
        for param_group in optimizer.param_groups:
            param_group['lr'] = now_lr
        print(f"Change learning rate from {origin_lr} to {now_lr}")

if dataset_name == "mini-imagenet":
    transform = transforms.Compose([transforms.Resize([224,224]), transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])])
    train_data = MiniImagenet('data/mini-imagenet', split="train", transform=transform)
    val_data = MiniImagenet('data/mini-imagenet', split="val", transform=transform)
    class_num = train_data.class_num
if dataset_name == "cifar10":
    transform = transforms.Compose([transforms.ToTensor(), transforms.Resize([224,224]), transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])])
    train_data = CIFAR10('data', train=True, transform=transform)
    val_data = CIFAR10('data', train=False, transform=transform)
    class_num = len(train_data.classes)

print(f"Dataset: {dataset_name}, class number: {class_num}")
data_loader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=32)

device = torch.device(device)
model = Resnet18(class_num).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=start_lr, momentum=0.9, weight_decay=0.0001)

train_losses = []
val_losses = []
val_accuracy = []
iteration = 0
model.train()
while iteration < train_iterations:
    for X, label in data_loader:
        iteration += 1
        X = X.to(device)
        label = label.to(device)

        pred = model(X)
        loss = F.nll_loss(pred, label)
        train_losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"Iteration {iteration} loss: {loss.item():.5f}")

        if iteration % save_per_iter == 0:
            save_model(model, f"model_{iteration}.pth")
            print("Evaluating ...")
            correct, val_loss = eval(model, val_data, device)
            total = len(val_data)
            accuracy = correct / total
            val_accuracy.append(accuracy.item())
            print(f"Validation accuracy: {correct}/{total}    {accuracy*100.0:.3f}%")
            print(f"Validation loss: {val_loss:.5f}")
            val_losses.append(val_loss)

            model.train()
            #update_learing_rate(train_losses, optimizer)
            pass
        if iteration >= train_iterations:
            break

save_dict = {"start_lr": start_lr, "train_iterations": train_iterations, "save_per_iter": save_per_iter,
             "train_losses": train_losses, "val_losses": val_losses, "val_accuracy": val_accuracy}
torch.save(save_dict, os.path.join("records", record_name))
