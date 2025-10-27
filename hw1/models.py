"""
模型定义文件
包含基线模型和各种改进的模型架构
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class BaselineCNN(nn.Module):
    """
    基线CNN模型 - 基于Example.ipynb的架构
    """
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 128, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2), nn.Dropout(0.3),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2), nn.Dropout(0.3),
            nn.Conv2d(256, 512, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2), nn.Dropout(0.3),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(512, 256), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(256, 128), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class ResidualBlock(nn.Module):
    """
    残差块 - 用于改进因子(a)
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


class CNNWithResidual(nn.Module):
    """
    带残差连接的CNN - 改进因子(a)
    """
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        
        # 残差块
        self.res_block1 = ResidualBlock(64, 128, stride=2)
        self.res_block2 = ResidualBlock(128, 256, stride=2)
        self.res_block3 = ResidualBlock(256, 512, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class DeeperWiderCNN(nn.Module):
    """
    更深更宽的CNN - 改进因子(b)
    """
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            # 第一层
            nn.Conv2d(3, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2), nn.Dropout(0.2),
            
            # 第二层
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.MaxPool2d(2), nn.Dropout(0.2),
            
            # 第三层
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.MaxPool2d(2), nn.Dropout(0.2),
            
            # 第四层
            nn.Conv2d(512, 1024, 3, padding=1), nn.BatchNorm2d(1024), nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, 3, padding=1), nn.BatchNorm2d(1024), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(512, 256), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(256, 128), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block - 通道注意力机制

    1. Squeeze: 全局平均池化，将空间信息压缩为通道描述符
    2. Excitation: 两层全连接网络，学习通道间的依赖关系
    3. Scale: 用学到的权重重新加权各通道
    
    参数量：2 * (C^2 / r) + 2C，其中r是reduction比例
    """
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        # Squeeze: 全局平均池化
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        
        # Excitation: 两层全连接
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        
        # Squeeze: 全局平均池化 (b, c, h, w) -> (b, c, 1, 1) -> (b, c)
        y = self.avgpool(x).view(b, c)
        
        # Excitation: 学习通道权重 (b, c) -> (b, c)
        y = self.fc(y).view(b, c, 1, 1)
        
        # Scale: 重新加权特征图
        return x * y.expand_as(x)


class CNNWithSEAttention(nn.Module):
    """
    带SE注意力的CNN - 改进因子(e)
    
    设计思路：
    - 在每个卷积块后添加SE注意力模块
    - 通过SE模块动态调整通道权重，强化重要特征
    """
    def __init__(self, num_classes=10):
        super().__init__()
        
        # 第一个卷积块: 3 -> 128
        self.conv1 = nn.Conv2d(3, 128, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.se1 = SEBlock(128, reduction=16)
        
        # 第二个卷积块: 128 -> 256
        self.conv2 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.se2 = SEBlock(256, reduction=16)
        
        # 第三个卷积块: 256 -> 512
        self.conv3 = nn.Conv2d(256, 512, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(512)
        self.se3 = SEBlock(512, reduction=16)
        
        # 第四个卷积块: 512 -> 512
        self.conv4 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.se4 = SEBlock(512, reduction=16)
        
        # 第五个卷积块: 512 -> 256
        self.conv5 = nn.Conv2d(512, 256, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.se5 = SEBlock(256, reduction=16)
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(512, 256), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(256, 128), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )
    
    def forward(self, x):
        # 第一个块
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.se1(x)
        x = F.max_pool2d(x, 2)
        x = F.dropout(x, p=0.3, training=self.training)
        
        # 第二个块
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.se2(x)
        x = F.max_pool2d(x, 2)
        x = F.dropout(x, p=0.3, training=self.training)
        
        # 第三个块
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.se3(x)
        
        # 第四个块
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.se4(x)
        
        # 第五个块
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.se5(x)
        x = F.max_pool2d(x, 2)
        x = F.dropout(x, p=0.3, training=self.training)
        
        # 分类
        x = self.classifier(x)
        return x


def get_model(model_name, num_classes=10):
    """
    获取指定模型
    
    可用模型：
    - baseline: 基线CNN模型
    - residual: 带残差连接的CNN
    - deeper_wider: 更深更宽的CNN
    - se_attention: 带SE注意力的CNN
    """
    models = {
        'baseline': BaselineCNN,
        'residual': CNNWithResidual,
        'deeper_wider': DeeperWiderCNN,
        'se_attention': CNNWithSEAttention,
    }
    
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Available models: {list(models.keys())}")
    
    return models[model_name](num_classes)
