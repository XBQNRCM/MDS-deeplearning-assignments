"""
最佳模型定义 - TinyImageNet版本
结合Deeper/Wider架构和Residual连接
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """
    残差块
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # 如果输入输出维度不同，需要调整维度
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(residual)
        out = F.relu(out)
        return out


class BestModel(nn.Module):
    """
    最佳组合模型：Deeper/Wider + Residual - TinyImageNet版本
    
    对齐DeeperWiderCNN的8层卷积结构，在其基础上添加残差连接。
    
    特点：
    1. 与DeeperWiderCNN相同的深度和宽度 (8层卷积)
    2. 残差连接 - 解决梯度消失问题，提升训练效果
    3. BatchNormalization - 加速训练，提高稳定性
    4. Dropout - 防止过拟合
    5. 适配64x64输入和200类输出
    
    架构对比 (与DeeperWiderCNN完全对齐):
    - 第一组: Conv(3→128) + Conv(128→128) → 替换为 ResidualBlock(3→128, stride=1)
    - 第二组: Conv(128→256) + Conv(256→256) → 替换为 ResidualBlock(128→256, stride=2)
    - 第三组: Conv(256→512) + Conv(512→512) → 替换为 ResidualBlock(256→512, stride=2)
    - 第四组: Conv(512→1024) + Conv(1024→1024) → 替换为 ResidualBlock(512→1024, stride=2)
    
    总计: 8层卷积 (4个ResidualBlock × 2层/block)
    """
    def __init__(self, num_classes=200):
        super(BestModel, self).__init__()
        
        # 第一组残差块 - 128通道
        self.res_block1 = ResidualBlock(3, 128, stride=1)
        self.dropout1 = nn.Dropout(0.2)
        
        # 第二组残差块 - 256通道
        self.res_block2 = ResidualBlock(128, 256, stride=2)
        self.dropout2 = nn.Dropout(0.2)
        
        # 第三组残差块 - 512通道
        self.res_block3 = ResidualBlock(256, 512, stride=2)
        self.dropout3 = nn.Dropout(0.2)
        
        # 第四组残差块 - 1024通道
        self.res_block4 = ResidualBlock(512, 1024, stride=2)
        
        # 全局平均池化
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(512, 256), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )
    
    def forward(self, x):
        # 第一组残差块 + Dropout
        x = self.res_block1(x)
        x = self.dropout1(x)
        
        # 第二组残差块 + Dropout
        x = self.res_block2(x)
        x = self.dropout2(x)
        
        # 第三组残差块 + Dropout
        x = self.res_block3(x)
        x = self.dropout3(x)
        
        # 第四组残差块
        x = self.res_block4(x)
        
        # 全局平均池化和分类
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        
        return x
    
    def get_num_params(self):
        """返回模型参数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

