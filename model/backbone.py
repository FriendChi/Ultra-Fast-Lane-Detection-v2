import torch,pdb
import torchvision
import torch.nn.modules
from torch import nn
import numpy as np
from torch.autograd import Variable


import torch
import torch.nn as nn

class TBBasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1):
        super(TBBasicBlock, self).__init__()

        # BatchNorm에 bias가 포함되어 있으므로, conv2d는 bias=False로 설정합니다.
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=(stride, 2), padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.ConvTranspose2d(out_channels, out_channels * TBBasicBlock.expansion, kernel_size=3, stride=(1,2), padding=1, bias=False),
            nn.BatchNorm2d(out_channels * TBBasicBlock.expansion),
        )

        # identity mapping, input과 output의 feature map size, filter 수가 동일한 경우 사용.
        self.shortcut = nn.Sequential()

        self.relu = nn.ReLU()

        # projection mapping using 1x1conv
        if stride != 1 or in_channels != TBBasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * TBBasicBlock.expansion, kernel_size=1, stride=(stride, 1), bias=False),
                nn.BatchNorm2d(out_channels * TBBasicBlock.expansion)
            )

    def forward(self, x):
        try:
            x = self.residual_function(x) + self.shortcut(x)
        except:
            x = self.residual_function(x) + self.shortcut(x)[:,:,:,:-1]
        x = self.relu(x)
        return x

        
class resnet(torch.nn.Module):
    def __init__(self,layers,pretrained = False):
        super(resnet,self).__init__()
        if layers == '18':
            model = torchvision.models.resnet18(weights="IMAGENET1K_V1")
        elif layers == '34':
            model = torchvision.models.resnet34(weights="IMAGENET1K_V1")
        elif layers == '50':
            model = torchvision.models.resnet50(weights="IMAGENET1K_V1")
        elif layers == '101':
            model = torchvision.models.resnet101(weights="IMAGENET1K_V1")
        elif layers == '152':
            model = torchvision.models.resnet152(weights="IMAGENET1K_V1")
        elif layers == '50next':
            model = torchvision.models.resnext50_32x4d(weights="IMAGENET1K_V1")
        elif layers == '101next':
            model = torchvision.models.resnext101_32x8d(weights="IMAGENET1K_V1")
        elif layers == '50wide':
            model = torchvision.models.wide_resnet50_2(weights="IMAGENET1K_V1")
        elif layers == '101wide':
            model = torchvision.models.wide_resnet101_2(weights="IMAGENET1K_V1")
        elif layers == '34fca':
            model = torch.hub.load('cfzd/FcaNet', 'fca34' ,pretrained=True)
        else:
            raise NotImplementedError
        

        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        print(x.shape)
        x = self.layer1(x)
        print(x.shape)
        x2 = self.layer2(x)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        
        return x2,x3,x4
