import torch,pdb
from torch import nn
import torchvision
import torch.nn.modules



import torch
import torch.nn as nn




class SiLU(nn.Module):  
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)
    
def autopad(k, p=None):
    if p is None:
        #这样p一定等于k // 2
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k] 
    return p

class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=SiLU()):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        self.conv   = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        #print(autopad(k, p))
        self.bn     = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03)
        self.act    = nn.LeakyReLU(0.1, inplace=True) if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))
    
class Multi_Concat_Block(nn.Module):
    def __init__(self, c1, c2, c3, n=4, e=1, ids=[0]):
        super(Multi_Concat_Block, self).__init__()
        c_ = int(c2 * e)
        
        self.ids = ids
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = nn.ModuleList(
            [Conv(c_ if i ==0 else c2, c2, 3, 1) for i in range(n)]
        )
        self.cv4 = Conv(c_ * 2 + c2 * (len(ids) - 2), c3, 1, 1)

    def forward(self, x):
        x_1 = self.cv1(x)
        x_2 = self.cv2(x)
        
        x_all = [x_1, x_2]
        for i in range(len(self.cv3)):
            x_2 = self.cv3[i](x_2)
            x_all.append(x_2)
            
        out = self.cv4(torch.cat([x_all[id] for id in self.ids], 1))
        return out

class MP(nn.Module):
    def __init__(self, k=2):
        super(MP, self).__init__()
        self.m = nn.MaxPool2d(kernel_size=k, stride=k)

    def forward(self, x):
        return self.m(x)
    
class Transition_Block(nn.Module):
    def __init__(self, c1, c2):
        super(Transition_Block, self).__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.cv2 = Conv(c1, c2, 1, 1)
        self.cv3 = Conv(c2, c2, 3, 2)
        
        self.mp  = MP()

    def forward(self, x):
        x_1 = self.mp(x)
        #print('x_1:',x_1.shape)
        x_1 = self.cv1(x_1)
        #print('x_1:',x_1.shape)

        x_2 = self.cv2(x)
        #print('x_2:',x_2.shape)
        x_2 = self.cv3(x_2)
        #print('x_2:',x_2.shape)
        
        return torch.cat([x_2, x_1], 1)
    
class _Transition_Block(nn.Module):
    def __init__(self, c1, c2):
        super(_Transition_Block, self).__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.cv2 = Conv(c1, c2, 1, 1)
        #self.cv3 = Conv(c2, c2, 3, 2)
        
        self.mp  = MP()

    def forward(self, x):
        x_1 = self.mp(x)
        #print('x_1:',x_1.shape)
        x_1 = self.cv1(x_1)
        #print('x_1:',x_1.shape)

        x_2 = self.cv2(x)
        #print('x_2:',x_2.shape)
        x_2 = self.mp(x_2)
        #print('x_2:',x_2.shape)
        
        return torch.cat([x_2, x_1], 1)

class resnet(nn.Module):
    def __init__(self, layers,pretrained = False,transition_channels= 32, block_channels=16, n= {'l' : 4, 'x' : 6}['l'], phi='l'):
        super().__init__()
        #-----------------------------------------------#
        #   输入图片是640, 640, 3
        #-----------------------------------------------#
        ids = {
            'l' : [-1, -3, -5, -6],
            'x' : [-1, -3, -5, -7, -8], 
        }[phi]
        #通道3->transition_channels,长宽/2
        self.conv1 = Conv(3, transition_channels//2, 3, 2)
        #通道*2，长宽/2
        self.conv2 = Conv(transition_channels//2, transition_channels, 3, 1)
        self.conv4 = Conv(transition_channels * 16, transition_channels * 16, 3, 2)
        #长宽/2
        #self.conv3 = Conv(transition_channels * 2, transition_channels * 2, 3, 2)
        # self.stem = nn.Sequential(
        #     Conv(3, transition_channels, 3, 2),
        #     Conv(transition_channels, transition_channels * 2, 3, 2),
        #     Conv(transition_channels * 2, transition_channels * 2, 3, 2),
        # )
        #通道*4，长宽/2
        self.dark2 = nn.Sequential(
            Conv(transition_channels , transition_channels  *2, 3, 2),
            Multi_Concat_Block(transition_channels * 2, block_channels * 2, transition_channels * 4, n=n, ids=ids),
        )
        #通道*2，长宽/2
        self.dark3 = nn.Sequential(
            Transition_Block(transition_channels * 4, transition_channels * 2),
            Multi_Concat_Block(transition_channels * 4, block_channels * 2, transition_channels * 8, n=n, ids=ids),
        )
        #通道*2，长宽/2
        self.dark4 = nn.Sequential(
            Transition_Block(transition_channels * 8, transition_channels * 4),
            Multi_Concat_Block(transition_channels * 8, block_channels * 4, transition_channels * 16, n=n, ids=ids),
        )
        #通道*2，长宽/2
        self.dark5 = nn.Sequential(
            Transition_Block(transition_channels * 16, transition_channels * 8),
            Multi_Concat_Block(transition_channels * 16, block_channels * 8, transition_channels * 16, n=n, ids=ids),
        )
        
        # if pretrained:
        #     url = {
        #         "l" : 'https://github.com/bubbliiiing/yolov7-pytorch/releases/download/v1.0/yolov7_backbone_weights.pth',
        #         "x" : 'https://github.com/bubbliiiing/yolov7-pytorch/releases/download/v1.0/yolov7_x_backbone_weights.pth',
        #     }[phi]
        #     # checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", model_dir="./model_data")
        #     # self.load_state_dict(checkpoint, strict=False)
        #     # print("Load weights from " + url.split('/')[-1])
        #     pretrained_dict = torch.load('/kaggle/input/yolov7-backbone-weights/yolov7_backbone_weights.pth')
        #     # 获取当前模型的参数字典
        #     model_dict = self.state_dict()
        #     selected_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        #     model_dict.update(selected_dict)
        #     #checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", model_dir="./model_data")
        #     self.load_state_dict(model_dict, strict=False)
        #     print("Load weights from " + url.split('/')[-1])
    def forward(self, x):
        x = self.conv1(x)
        print(x.shape)
        x = self.conv2(x)
        print(x.shape)
        # x = self.conv3(x)
        # print(x.shape)
        x = self.dark2(x)
        #feat1 = x
        print(x.shape)
        #-----------------------------------------------#
        #   dark3的输出为80, 80, 512，是一个有效特征层
        #-----------------------------------------------#
        x = self.dark3(x)
        #feat1 = x
        print(3,x.shape)
        
        #-----------------------------------------------#
        #   dark4的输出为40, 40, 1024，是一个有效特征层
        #-----------------------------------------------#
        x = self.dark4(x)
        x_conv4 = self.conv4(x)
        #feat2 = x
        print(4,x.shape)
        #-----------------------------------------------#
        #   dark5的输出为20, 20, 1024，是一个有效特征层
        #-----------------------------------------------#
        x = self.dark5(x)
        feat3 = torch.cat((x, x_conv4), dim=1)
        #torch.add(x1, x2)
        print(5,feat3.shape)
        return None,None,feat3  #, feat2, feat3
