import torch,pdb
import torchvision
import torch.nn.modules
from torch import nn
import numpy as np
from torch.autograd import Variable
from vit_pytorch import ViT

class resnet(torch.nn.Module):
    def __init__(self,layers,pretrained = False):
        super(resnet,self).__init__()
        self.model = ViT(
            image_size = 800,
            patch_size = 40,
            num_classes = 1000,
            dim = 1024,
            depth = 6,
            heads = 16,
            mlp_dim = 2048,
            dropout = 0.1,
            emb_dropout = 0.1
        )
        self.model.mlp_head = nn.Identity()
    
    def forward(self,x):
        x = self.model(x)
        

        return None,None,x
