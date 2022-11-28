"""
Fixup initialisation replaces BatchNorm.

https://github.com/hongyi-zhang/Fixup/blob/master/LICENSE
"""

import numpy as np
import torch
import torch.nn as nn


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class FixupBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.bias1a = nn.Parameter(torch.zeros(1))
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bias1b = nn.Parameter(torch.zeros(1))
        self.relu = nn.ReLU(inplace=True)
        self.bias2a = nn.Parameter(torch.zeros(1))
        self.conv2 = conv3x3(planes, planes)
        self.scale = nn.Parameter(torch.ones(1))
        self.bias2b = nn.Parameter(torch.zeros(1))
        self.downsample = downsample


    def forward(self, x):
        identity = x

        out = self.conv1(x + self.bias1a)
        out = self.relu(out + self.bias1b)

        out = self.conv2(out + self.bias2a)
        out = out * self.scale + self.bias2b

        if self.downsample is not None:
            identity = self.downsample(x + self.bias1a)
            identity = torch.cat((identity, torch.zeros_like(identity)), 1)

        out += identity
        out = self.relu(out)

        return out


def init_fixup_(self, num_layers):
    for m in self.modules():
        if isinstance(m, FixupBasicBlock):
            std = np.sqrt(2 / (m.conv1.weight.shape[0] * np.prod(m.conv1.weight.shape[2:]))) * num_layers ** (-0.5)
            nn.init.normal_(m.conv1.weight, mean=0, std=std)
            nn.init.constant_(m.conv2.weight, 0)
        elif isinstance(m, nn.Linear):
            nn.init.constant_(m.weight, 0)
            nn.init.constant_(m.bias, 0)


class FixupResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super().__init__()

        self.num_layers = sum(layers)
        self.inplanes = 64
        self.conv1 = conv3x3(3, 64)
        self.bias1 = nn.Parameter(torch.zeros(1))
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        #self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.pool = GlobalPool()
        self.bias2 = nn.Parameter(torch.zeros(1))
        self.fc = nn.Linear(self.inplanes, num_classes)

        init_fixup_(self, self.num_layers)


    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1:
            downsample = nn.AvgPool2d(1, stride=stride)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(block(planes, planes))

        return nn.Sequential(*layers)


    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x + self.bias1)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.pool(x)
        #x = x.view(x.size(0), -1)
        x = self.fc(x + self.bias2)

        return x


class GlobalPool(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.amax(x, dim=(-2,-1))


def make_model():
    model = FixupResNet(FixupBasicBlock, [2,2,3], num_classes=10)
    print(model)    
    return model

