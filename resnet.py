import torch
import torch.nn as nn
import torch.nn.functional as F

class Resnet18(nn.Module):
    def __init__(self, out_dims):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, [7,7], stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d([3,3], stride=2, padding=1)
        self.conv2 = nn.Sequential(Resblock(64), Resblock(64))
        self.conv3 = nn.Sequential(Resblock_inc(64, 128), Resblock(128))
        self.conv4 = nn.Sequential(Resblock_inc(128, 256), Resblock(256))
        self.conv5 = nn.Sequential(Resblock_inc(256, 512), Resblock(512))
        self.fc = nn.Linear(512, out_dims)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        x = F.avg_pool2d(x, kernel_size=[x.shape[-2],x.shape[-1]])
        x = x.squeeze(dim=-1)
        x = x.squeeze(dim=-1)
        x = self.fc(x)
        return F.log_softmax(x, dim=-1)


class Resblock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, [3,3], padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, [3,3], padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
    
    def forward(self, x):
        x_ = F.relu(self.bn1(self.conv1(x)))
        x_ = self.bn2(self.conv2(x_))
        return F.relu(x + x_)

    
class Resblock_inc(nn.Module):
    def __init__(self, in_channels, out_channels, use_padding=True):
        super().__init__()
        self.use_padding = use_padding
        self.conv1 = nn.Conv2d(in_channels, out_channels, [3,3], stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, [3,3], padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if not self.use_padding:
            raise NotImplementedError
    
    def forward(self, x):
        x_ = F.relu(self.bn1(self.conv1(x)))
        x_ = self.bn2(self.conv2(x_))
        if self.use_padding:
            x = F.pad(x[:,:,::2,::2], [0,0,0,0,0,x_.shape[-3]-x.shape[-3]])
        else:
            raise NotImplementedError
        return F.relu(x + x_)
