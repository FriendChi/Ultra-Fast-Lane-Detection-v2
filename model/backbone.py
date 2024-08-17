import torch,pdb
import torchvision
import torch.nn.modules
from torch import nn
import numpy as np
from torch.autograd import Variable
from .ca import resnet18_ca

class resnet(torch.nn.Module):
    def __init__(self,layers,pretrained = False):
        super(resnet,self).__init__()
        self.model = resnet18_ca(True)
    
    
    def forward(self,x):
        x = self.model(x)
        

        return None,None,x
