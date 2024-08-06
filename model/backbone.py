import torch,pdb
import torchvision
import torch.nn.modules
from torch import nn
import numpy as np
from torch.autograd import Variable


        
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
        
        self.conv1_1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1_1 = nn.BatchNorm2d(64)
        self.conv1_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1_2 = nn.BatchNorm2d(64)
        self.conv1_3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1_3 = nn.BatchNorm2d(64)
        # self.conv1 = model.conv1
        self.relu = model.relu
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
        del model.conv1    
        del model.fc
        del model.avgpool

    def forward(self,x):
        x = self.conv1_1(x)
        x = self.bn1_1(x)
        x = self.relu(x)

        x = self.conv1_2(x)
        x = self.bn1_2(x)
        x = self.relu(x)

        x = self.conv1_3(x)
        x = self.bn1_3(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x2 = self.layer2(x)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        
        return x2,x3,x4
