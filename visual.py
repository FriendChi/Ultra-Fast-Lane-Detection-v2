import torch
import time
import matplotlib.pyplot as plt
from utils.common import calc_loss
from PIL import Image
import numpy as np
from utils.common import get_model, merge_config
import torchvision.transforms as transforms
image_path = '/kaggle/input/tusimple/TUSimple/test_set/clips/0530/1492626047222176976_0/1.jpg'  # 替换为你的图片路径
torch.backends.cudnn.benchmark = True
args, cfg = merge_config()
net = get_model(cfg)
net.eval()
img_transforms = transforms.Compose([
    transforms.Resize((int(cfg.train_height), cfg.train_width)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

img = Image.open(image_path)  # 打开图片
img_tensor = img_transforms(img)  # 预处理
img_tensor = img_tensor.unsqueeze(0)  # 增加 batch 维度，形状变为 (1, 3, cfg.train_height, cfg.train_width)
x = img_tensor.cuda()  # 移动到 GPU 上
print('x:',x.shape)

y,feature_map = net(x)

# 计算在通道维度上的平均值，得到形状为 (1, 25, 40)
mean_feature_map = feature_map.mean(dim=1, keepdim=True)  # 保持维度为 1
mean_feature_map = mean_feature_map.squeeze(0).detach().cpu().numpy()  # 转换为 NumPy 数组

# 假设你已经得到了 `heatmap_data`，它的形状为 (25, 40)
heatmap_data = mean_feature_map.squeeze(0)  # 如果使用平均值方法
# heatmap_data = specific_channel  # 如果选择了特定通道

plt.figure(figsize=(8, 6))
plt.imshow(heatmap_data, cmap='jet', aspect='auto')  # 选择颜色映射
plt.colorbar()  # 显示颜色条
plt.title('Heatmap')
plt.xlabel('Width')
plt.ylabel('Height')

# 保存热图
plt.savefig('heatmap.png', bbox_inches='tight')  # 可以指定路径和文件名
plt.close()  # 关闭当前图形



