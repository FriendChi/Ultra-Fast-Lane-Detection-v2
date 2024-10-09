import torch
from torch import nn
from torch.nn.parameter import Parameter
import matplotlib.pyplot as plt
import seaborn as sns
import os


class eca_layer(nn.Module):
    """Constructs a ECA module.

    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()
        self.channel = channel

    def find_next_available_index(self,save_dir="attention_weights"):
        i = 0
        while True:
            # 构造当前检查的文件名
            file_prefix = f"attention_weights_{i}"
            # 获取目录中的所有文件
            existing_files = os.listdir(save_dir)
            
            # 检查是否有以 file_prefix 开头的文件
            if not any(file.startswith(file_prefix) for file in existing_files):
                break  # 如果没有以该前缀开头的文件，则退出循环
            i += 1  # 继续检查下一个索引
    
        return i  # 返回第一个不存在的索引

    def save_attention_weights(self,attention_weights):
        # 确保保存目录存在
        save_dir = '/kaggle/working/output'
        os.makedirs(save_dir, exist_ok=True)
        batch = self.find_next_available_index(save_dir)
        
        # attention_weights 的形状通常是 (B, 1, C, 1)
        for i in range(attention_weights.size(0)):
            weights = attention_weights[i].squeeze().detach().cpu().numpy()  # 提取当前样本的权重并转为 NumPy 数组
            
            # 绘制热图
            plt.figure(figsize=(10, 1))
            sns.heatmap(weights, cmap='viridis', cbar=True, annot=True)
            
            # 保存热图为图片
            plt.savefig(os.path.join(save_dir, f"attention_weights_{batch}_{i}_{self.channel}.png"), bbox_inches='tight', dpi=300)  # 设置 dpi 以提高分辨率
            plt.close()  # 关闭当前图形，释放内存
    
    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)
        print(y.shape)
        self.save_attention_weights(y)

        return x * y.expand_as(x)
        
