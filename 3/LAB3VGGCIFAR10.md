```python
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,  download=True, transform=transform_train)
testset  = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
```

    100%|██████████| 170M/170M [00:03<00:00, 43.9MB/s]
    


```python
class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.cfg = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
        self.features = self._make_layers(self.cfg)# cfg不对，应当修改成self.cfg
        self.classifier = nn.Linear(512, 10)# 2048不太合适。

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)
```


```python
net = VGG().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)
```


```python
for epoch in range(10):
    for i, (inputs, labels) in enumerate(trainloader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print('Epoch: %d Minibatch: %5d loss: %.3f' %(epoch + 1, i + 1, loss.item()))

print('Finished Training')
```

    Epoch: 1 Minibatch:     1 loss: 2.330
    Epoch: 1 Minibatch:   101 loss: 1.408
    Epoch: 1 Minibatch:   201 loss: 1.162
    Epoch: 1 Minibatch:   301 loss: 1.148
    Epoch: 2 Minibatch:     1 loss: 0.932
    Epoch: 2 Minibatch:   101 loss: 1.208
    Epoch: 2 Minibatch:   201 loss: 0.868
    Epoch: 2 Minibatch:   301 loss: 0.766
    Epoch: 3 Minibatch:     1 loss: 0.788
    Epoch: 3 Minibatch:   101 loss: 0.890
    Epoch: 3 Minibatch:   201 loss: 0.775
    Epoch: 3 Minibatch:   301 loss: 0.723
    Epoch: 4 Minibatch:     1 loss: 0.731
    Epoch: 4 Minibatch:   101 loss: 0.728
    Epoch: 4 Minibatch:   201 loss: 0.824
    Epoch: 4 Minibatch:   301 loss: 0.510
    Epoch: 5 Minibatch:     1 loss: 0.633
    Epoch: 5 Minibatch:   101 loss: 0.575
    Epoch: 5 Minibatch:   201 loss: 0.593
    Epoch: 5 Minibatch:   301 loss: 0.523
    Epoch: 6 Minibatch:     1 loss: 0.578
    Epoch: 6 Minibatch:   101 loss: 0.556
    Epoch: 6 Minibatch:   201 loss: 0.530
    Epoch: 6 Minibatch:   301 loss: 0.316
    Epoch: 7 Minibatch:     1 loss: 0.594
    Epoch: 7 Minibatch:   101 loss: 0.453
    Epoch: 7 Minibatch:   201 loss: 0.552
    Epoch: 7 Minibatch:   301 loss: 0.530
    Epoch: 8 Minibatch:     1 loss: 0.438
    Epoch: 8 Minibatch:   101 loss: 0.413
    Epoch: 8 Minibatch:   201 loss: 0.395
    Epoch: 8 Minibatch:   301 loss: 0.717
    Epoch: 9 Minibatch:     1 loss: 0.548
    Epoch: 9 Minibatch:   101 loss: 0.362
    Epoch: 9 Minibatch:   201 loss: 0.480
    Epoch: 9 Minibatch:   301 loss: 0.407
    Epoch: 10 Minibatch:     1 loss: 0.331
    Epoch: 10 Minibatch:   101 loss: 0.400
    Epoch: 10 Minibatch:   201 loss: 0.544
    Epoch: 10 Minibatch:   301 loss: 0.346
    Finished Training
    


```python
correct = 0
total = 0

for data in testloader:
    images, labels = data
    images, labels = images.to(device), labels.to(device)
    outputs = net(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %.2f %%' % (
    100 * correct / total))
```

    Accuracy of the network on the 10000 test images: 84.50 %
    
