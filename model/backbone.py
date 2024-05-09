import torch,pdb
import torchvision
import torch.nn.modules
import torch
import torch.nn as nn

# class vgg16bn(torch.nn.Module):
#     def __init__(self,pretrained = False):
#         super(vgg16bn,self).__init__()
#         model = list(torchvision.models.vgg16_bn(pretrained=pretrained).features.children())
#         model = model[:33]+model[34:43]
#         self.model = torch.nn.Sequential(*model)
        
#     def forward(self,x):
#         return self.model(x)




def autopad(k, p=None):
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k] 
    return p

class SiLU(nn.Module):  
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)
    
class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=SiLU()):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        self.conv   = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
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
        x_1 = self.cv1(x_1)
        
        x_2 = self.cv2(x)
        x_2 = self.cv3(x_2)
        
        return torch.cat([x_2, x_1], 1)

class resnet1(nn.Module):
    def __init__(self, layers,pretrained = False,transition_channels= {'l' : 32, 'x' : 40}['l'], block_channels=32, n= {'l' : 4, 'x' : 6}['l'], phi='l'):
        super().__init__()
        #-----------------------------------------------#
        #   输入图片是640, 640, 3
        #-----------------------------------------------#
        ids = {
            'l' : [-1, -3, -5, -6],
            'x' : [-1, -3, -5, -7, -8], 
        }[phi]
        self.stem = nn.Sequential(
            Conv(3, transition_channels, 3, 2),
            Conv(transition_channels, transition_channels * 2, 3, 2),
            Conv(transition_channels * 2, transition_channels * 2, 3, 2),
        )
        self.dark2 = nn.Sequential(
            Conv(transition_channels * 2, transition_channels * 4, 3, 2),
            Multi_Concat_Block(transition_channels * 4, block_channels * 2, transition_channels * 8, n=n, ids=ids),
        )
        self.dark3 = nn.Sequential(
            Transition_Block(transition_channels * 8, transition_channels * 4),
            Multi_Concat_Block(transition_channels * 8, block_channels * 4, transition_channels * 16, n=n, ids=ids),
        )
        # self.dark4 = nn.Sequential(
        #     Transition_Block(transition_channels * 16, transition_channels * 8),
        #     Multi_Concat_Block(transition_channels * 16, block_channels * 8, transition_channels * 32, n=n, ids=ids),
        # )
        # self.dark5 = nn.Sequential(
        #     Transition_Block(transition_channels * 32, transition_channels * 16),
        #     Multi_Concat_Block(transition_channels * 32, block_channels * 8, transition_channels * 32, n=n, ids=ids),
        # )
        
        if pretrained:
            url = {
                "l" : 'https://github.com/bubbliiiing/yolov7-pytorch/releases/download/v1.0/yolov7_backbone_weights.pth',
                "x" : 'https://github.com/bubbliiiing/yolov7-pytorch/releases/download/v1.0/yolov7_x_backbone_weights.pth',
            }[phi]
            # checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", model_dir="./model_data")
            # self.load_state_dict(checkpoint, strict=False)
            # print("Load weights from " + url.split('/')[-1])
            pretrained_dict = torch.load('./model_data/yolov7_backbone_weights.pth')
            # 获取当前模型的参数字典
            model_dict = self.state_dict()
            selected_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(selected_dict)
            #checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", model_dir="./model_data")
            self.load_state_dict(model_dict, strict=False)
            print("Load weights from " + url.split('/')[-1])
    def forward(self, x):
        x = self.stem(x)
        x = self.dark2(x)
        #-----------------------------------------------#
        #   dark3的输出为80, 80, 512，是一个有效特征层
        #-----------------------------------------------#
        x = self.dark3(x)
        feat1 = x
        #-----------------------------------------------#
        #   dark4的输出为40, 40, 1024，是一个有效特征层
        #-----------------------------------------------#
        #x = self.dark4(x)
        #feat2 = x
        #-----------------------------------------------#
        #   dark5的输出为20, 20, 1024，是一个有效特征层
        #-----------------------------------------------#
        #x = self.dark5(x)
        #feat3 = x
        return None,None,feat1  #, feat2, feat3



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
