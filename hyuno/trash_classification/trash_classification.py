import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
import torchvision.transforms as transforms
import os

from torchvision import models
from torchvision.datasets import ImageFolder



class Resnet(nn.Module):
    def __init__(self):
        super(Resnet, self).__init__()
        self.model = models.resnet50(pretrained=True)
        num_cnl = self.model.fc.in_features # fc layer의 node수
        self.model.fc = nn.Linear(num_cnl, len(datasets.classes))

    def forward(self, x):
        # 어떻게 forward 함수를 짜야할까


dir = 'Garbage classification'

classes = ['glass', 'paper', 'cardboard', 'plastic', 'metal', 'trash']


transformations = transforms.Compose([transforms.Resize((256, 256)),
                                      transforms.ToTensor]) #image 변조. resize후 tensor로 변환

random_seed = 42
torch.manual_seed(random_seed)

datasets = ImageFolder(dir, transform=transformations)

cnt = 0
for i in os.listdir(dir):
    cnt += len(os.listdir(dir + '/' + i))
print(cnt)

train = round(cnt*0.7)
val = round((cnt-train)*0.5)
test = round((cnt-train)*0.5)# 7:1.5:1.5로 train val test 나눔

train, val, test = random_split(datasets, [train, val, test])

batch_size = 32

train_dataloader = DataLoader(
    train, batch_size, shuffle=True, num_workers=3
)

val_dataloader = DataLoader(
    val, batch_size, shuffle=True, num_workers=3
)




