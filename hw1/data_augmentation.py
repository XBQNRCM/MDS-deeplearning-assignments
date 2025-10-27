"""
数据增强模块
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
from PIL import Image


class MixUp:
    """
    MixUp数据增强 - 改进因子(d)
    将两张图像按比例混合，标签也相应混合
    """
    def __init__(self, alpha=1.0):
        self.alpha = alpha
    
    def __call__(self, batch, targets):
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1
        
        batch_size = batch.size(0)
        index = torch.randperm(batch_size).to(batch.device)
        
        # 创建新的 tensor 避免修改原始数据
        mixed_batch = lam * batch + (1 - lam) * batch[index]
        y_a, y_b = targets, targets[index]
        return mixed_batch, y_a, y_b, lam


class CutMix:
    """
    CutMix数据增强 - 改进因子(d)
    将一张图像的矩形区域替换为另一张图像的对应区域
    """
    def __init__(self, alpha=1.0):
        self.alpha = alpha
    
    def __call__(self, batch, targets):
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1
        
        batch_size = batch.size(0)
        index = torch.randperm(batch_size).to(batch.device)
        
        y_a, y_b = targets, targets[index]
        
        # Clone batch 避免修改原始数据
        mixed_batch = batch.clone()
        bbx1, bby1, bbx2, bby2 = self.rand_bbox(batch.size(), lam)
        mixed_batch[:, :, bbx1:bbx2, bby1:bby2] = batch[index, :, bbx1:bbx2, bby1:bby2]
        
        # 根据实际裁剪区域调整lambda
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (batch.size()[-1] * batch.size()[-2]))
        return mixed_batch, y_a, y_b, lam
    
    def rand_bbox(self, size, lam):
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        
        # 统一中心
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        return bbx1, bby1, bbx2, bby2


def get_transforms(dataset_name, augmentation_type='basic'):
    """
    获取数据变换和batch级别的增强方法
    
    参数:
        dataset_name: 数据集名称 ('cifar10' 或 'tiny_imagenet')
        augmentation_type: 增强类型 ('basic', 'mixup', 'cutmix')
    返回:
        train_transform: 训练集的图像变换
        test_transform: 测试集的图像变换
    """

    if dataset_name == 'cifar10':
        if augmentation_type == 'basic':
            # 基础增强：旋转、翻转、平移
            train_transform = transforms.Compose([
                transforms.RandomRotation(degrees=15),
                transforms.RandomHorizontalFlip(),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])
        else:
            # MixUp/CutMix: 使用基础的图像变换
            train_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
    
    elif dataset_name == 'tiny_imagenet':
        if augmentation_type == 'basic':
            # 基础增强：裁剪、翻转、颜色抖动
            train_transform = transforms.Compose([
                transforms.RandomResizedCrop(64),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            # MixUp/CutMix: 使用基础的图像变换
            train_transform = transforms.Compose([
                transforms.RandomResizedCrop(64),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

        test_transform = transforms.Compose([
            transforms.Resize(64),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return train_transform, test_transform
