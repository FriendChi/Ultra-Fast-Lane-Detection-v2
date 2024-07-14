import torch,pdb
import torchvision
import torch.nn.modules
from torch import nn
import numpy as np
from torch.autograd import Variable
from ..efficientvit_master.efficientvit.models.efficientvit.backbone import efficientvit_backbone_b0 

class resnet(torch.nn.Module):
    def __init__(self,layers,pretrained = False):
        super(resnet,self).__init__()
        self.model = efficientvit_backbone_b0()
        self.conv = nn.Conv2d(128, 128*4, kernel_size=1)

    def forward(self,x):
        x = self.model(x)
        x = self.conv(x)
        
        return None,None,x
