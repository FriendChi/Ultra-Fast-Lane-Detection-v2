import torch,pdb
import torchvision
import torch.nn.modules
from torch import nn
import numpy as np
from torch.autograd import Variable
import sys
import os
sys.path.append('/kaggle/working/Ultra-Fast-Lane-Detection-v2/model')
# 添加efficientvit文件所在的目录到Python路径
efficientvit_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'efficientvit_master', 'efficientvit', 'models', 'efficientvit'))
if efficientvit_path not in sys.path:
    sys.path.append(efficientvit_path)

# 导入函数
from backbone_ import efficientvit_backbone_b0

class resnet(torch.nn.Module):
    def __init__(self,layers,pretrained = False):
        super(resnet,self).__init__()
        self.model = efficientvit_backbone_b0()
        self.conv = nn.Conv2d(128, 128*4, kernel_size=1)

    def forward(self,x):
        x = self.model(x)
        x = self.conv(x['stage_final'])
        
        return None,None,x
