import torch
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np

from MiniImagenet import MiniImagenet
from resnet import Resnet18

def imshow(img):
    npimg = img.numpy()
    plt.figure()
    plt.imshow(np.transpose(npimg, (1,2,0)))

# transform = transforms.Compose([transforms.Resize([224,224]), transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])])
transform = transforms.Compose([transforms.Resize([224,224])])
data = MiniImagenet('data\mini-imagenet', split="val", transform=transform)
data_loader = torch.utils.data.DataLoader(data, shuffle=True, batch_size=1)

model = Resnet18(100)

model.eval()
with torch.no_grad():
    for a in data_loader:
        print(a[0].shape)
        # print(a[1].shape)
        # print(a)
        imshow(torchvision.utils.make_grid(a[0]))
        b = model(a[0])
        print(b.shape)

        '''
        b = torch.squeeze(b)
        b = torch.stack([b, b, b], dim=1)
        imshow(torchvision.utils.make_grid(b))
        '''
        break

plt.show()
