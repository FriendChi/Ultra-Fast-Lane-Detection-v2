import torch,pdb
import torchvision
import torch.nn.modules
from torch import nn
import numpy as np
from torch.autograd import Variable
from pycls import models

        
class resnet(torch.nn.Module):
    def __init__(self,layers,pretrained = False):
        super(resnet,self).__init__()
        
        model = models.regnety("600MF", pretrained=True)

        self.stem = model.stem
        self.s1 = model.s1
        self.s2 = model.s2
        self.s3 = model.s3
        self.s4 = model.s4

    def forward(self,x):
        x = self.stem(x)
        x = self.s1(x)
        x = self.s2(x)
        x = self.s3(x)
        x = self.s4(x)

        return None,None,x
