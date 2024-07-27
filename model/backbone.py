import torch,pdb
import torchvision
import torch.nn.modules
from torch import nn
import numpy as np
from torch.autograd import Variable
from timm import create_model

        
class resnet(torch.nn.Module):
    def __init__(self,layers,pretrained = False):
        super(resnet,self).__init__()
        model_name = "convnext_small.fb_in1k"
        model = create_model(model_name, pretrained=True)
        self.stem = model.stem
        self.stages = model.stages



    def forward(self,x):
        x=self.stem(x)
        x=self.stages(x)

        
        return None,None,x
