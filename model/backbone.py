import torch,pdb
import torchvision
import torch.nn.modules
from torch import nn
import numpy as np
from torch.autograd import Variable
from .eca_resnet import eca_resnet18

class resnet(torch.nn.Module):
    def __init__(self,layers,pretrained = False):
        super(resnet,self).__init__()
        self.model = eca_resnet18(pretrained=True)
    
    
    def forward(self,x):
        x = self.model(x)
        

        return None,None,x
