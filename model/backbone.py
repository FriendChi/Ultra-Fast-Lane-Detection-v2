import torch,pdb
import torchvision
import torch.nn.modules
from torch import nn
import numpy as np
from torch.autograd import Variable
import iresnet

        
class resnet(torch.nn.Module):
    def __init__(self,layers,pretrained = False):
        super(resnet,self).__init__()
        
        model = iresnet.iresnet18(
        pretrained=False,
        num_classes=2,
        zero_init_residual=True)
            

    def forward(self,x):
        x4 = model(x)
        
        return None,None,x4
