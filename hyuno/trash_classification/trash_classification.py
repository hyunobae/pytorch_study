import torch
from torch import nn
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
import torchvision.transforms as transforms
import os
import torch.nn.functional as F
from torchvision.datasets import ImageFolder


class sModel(nn.Module):
    """
    model 선언하는 부분
    항상 nn.Module을 상속해야함
    nn. -> keras의 model.과 같음
    F. -> keras의 layers
    BN layer도 선언해줘야함
    """

    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.conv1 = nn.Conv2d(3, 64, (3, 3), stride=(1, 1))  # stride로 size 줄이기
        self.conv2 = nn.Conv2d(64, 128, (3, 3), stride=(1, 1))
        self.conv3 = nn.Conv2d(128, 128, (3, 3), stride=(1, 1))
        self.conv4 = nn.Conv2d(128, 256, (3, 3), stride=(1, 1))
        self.maxpool = nn.MaxPool2d((2, 2), (2, 2))

        self.fc1 = nn.Linear(14*14*256, 64)  # flatten
        self.fc2 = nn.Linear(64, 16)
        self.fc3 = nn.Linear(16, 6)

    def forward(self, x):  # 위 model사이에 어떤 관계가 있는지 선언하고 순전파 수행
        """
        모델 사이에 어떤 관계로 순전파를 진행해야하는가
        maxpool과 같이 재사용 가능한 module은 한번만 선언
        activation은 forward에서 바로 선언해서 사용함
        Flatten이 있지만 직접 명시해줘야하는 듯
        View로 input size 미리 flatten 해야함
        """
        x = self.conv1(x)
        x = F.relu(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.maxpool(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.maxpool(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.maxpool(x)
        #print(x.shape)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = F.softmax(x, dim=1)
        return x

USE_CUDA = torch.cuda.is_available()
print(USE_CUDA)


dir = 'Garbage classification'

classes = ['glass', 'paper', 'cardboard', 'plastic', 'metal', 'trash']

transformations = transforms.Compose([transforms.Resize((256, 256)),
                                      transforms.ToTensor()])  # image 변조. resize후 tensor로 변환

random_seed = 42
torch.manual_seed(random_seed)

datasets = ImageFolder(dir, transform=transformations)

cnt = 0
for i in os.listdir(dir):
    cnt += len(os.listdir(dir + '/' + i))
print(cnt)

train = round(cnt * 0.7)
val = round((cnt - train) * 0.5)
test = round((cnt - train) * 0.5)  # 7:1.5:1.5로 train val test 나눔

train, val, test = random_split(datasets, [train, val, test])

batch_size = 32

train_dataloader = DataLoader(
    train, batch_size, shuffle=True
)

val_dataloader = DataLoader(
    val, batch_size, shuffle=True
)

loss_fn = nn.CrossEntropyLoss()

model = sModel()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
scheduler = ExponentialLR(optimizer, gamma=0.9)  # learning rate 학습하면서 변경해주는 스케줄러

print(f"Model structure: {model}\n\n")


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (x, y) in enumerate(dataloader):
        optimizer.zero_grad()   # gradient 초기화->이전 loop 값들이 update에 영향주는 것을 방지
        pred = model(x) # model에 data input
        loss = loss_fn(pred, y) # loss 계산

        loss.backward() # back-propagation하며 gradient 계산
        optimizer.step() # parameter update

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(x)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batch = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for x,y in dataloader:
            pred = model(x)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) ==y).type(torch.float).sum().item()

    test_loss /= num_batch
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")



for i in range(3):
    print(f"Epoch {i + 1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(val_dataloader, model, loss_fn)
