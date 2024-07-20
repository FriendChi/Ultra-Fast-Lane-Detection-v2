import torch,pdb
import torchvision
import torch.nn.modules
from torch import nn
import numpy as np
from torch.autograd import Variable

# 定义新的下采样模块，它将包含平均池化和1x1卷积
class AvgPoolDownsample(nn.Module):
    def __init__(self, in_channels, out_channels, pool_size, pool_stride):
        super(AvgPoolDownsample, self).__init__()
        self.pool = nn.AvgPool2d(pool_size, pool_stride)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        return x
    
# 修改第一个残差块的快捷连接
def change_shortcut(layer):
    first_residual_block = layer[0]

    # 检查是否需要下采样
    if first_residual_block.downsample is not None:
        # 获取输入和输出的通道数
        in_channels = first_residual_block.conv1.in_channels
        out_channels = first_residual_block.conv2.out_channels

        # 计算池化大小和步长，通常与卷积的步长相同
        pool_size = 2  # 保持特征图尺寸不变
        pool_stride = first_residual_block.downsample[0].stride[0]  # 使用卷积的步长

        # 创建新的下采样模块
        new_downsample = AvgPoolDownsample(in_channels, out_channels, pool_size, pool_stride)

        # 替换旧的下采样模块
        first_residual_block.downsample = nn.Sequential(new_downsample)




# 对其他需要修改的残差块重复上述步骤
        
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
        change_shortcut(self.layer1)
        change_shortcut(self.layer2)
        change_shortcut(self.layer3)
        change_shortcut(self.layer4)

    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x2 = self.layer2(x)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        return x2,x3,x4
