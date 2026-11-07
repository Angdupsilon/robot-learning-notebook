import numpy as np
import torch
import torchvision
import math
from torch.utils.data import Dataset, DataLoader

# class MNIST(Dataset):
#     def __init__(self, transform=None):
#         # self.mnist=torchvision.datasets.MNIST(root="./data", train=True, download=True)
#     def __getitem__(self, index):
#         return

mnist=torchvision.datasets.MNIST(root="./data", train=True, transform=torchvision.transforms.ToTensor(),download=True)
dataloader=DataLoader(dataset=mnist, batch_size=64, shuffle=True)

data=next(iter(dataloader))
image, index = data
print(image.shape, image.dtype, image.device)   # predict device as cpu
print(index.shape, index.dtype, index.device)
print()
# epochs = 10, batch_size = 64 and learning_rate = 1e-3

print(type(mnist[0]))