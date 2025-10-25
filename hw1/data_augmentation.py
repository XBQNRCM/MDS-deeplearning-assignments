"""
数据增强模块
包含各种高级数据增强方法
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import random


class MixUp:
    """
    MixUp数据增强 - 改进因子(d)
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
        
        mixed_batch = lam * batch + (1 - lam) * batch[index, :]
        y_a, y_b = targets, targets[index]
        return mixed_batch, y_a, y_b, lam


class CutMix:
    """
    CutMix数据增强 - 改进因子(d)
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
        bbx1, bby1, bbx2, bby2 = self.rand_bbox(batch.size(), lam)
        batch[:, :, bbx1:bbx2, bby1:bby2] = batch[index, :, bbx1:bbx2, bby1:bby2]
        
        # 调整lambda
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (batch.size()[-1] * batch.size()[-2]))
        return batch, y_a, y_b, lam
    
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


class AutoAugment:
    """
    AutoAugment数据增强 - 改进因子(d)
    """
    def __init__(self):
        # CIFAR-10的AutoAugment策略
        self.policies = [
            [('Invert', 0.1, 7), ('Contrast', 0.2, 6)],
            [('Rotate', 0.7, 2), ('TranslateX', 0.3, 9)],
            [('Sharpness', 0.8, 1), ('Sharpness', 0.9, 3)],
            [('ShearY', 0.5, 8), ('TranslateY', 0.7, 9)],
            [('AutoContrast', 0.5, 8), ('Equalize', 0.9, 2)],
            [('ShearY', 0.2, 7), ('Posterize', 0.3, 7)],
            [('Color', 0.4, 3), ('Brightness', 0.6, 7)],
            [('Sharpness', 0.3, 9), ('Brightness', 0.7, 9)],
            [('Equalize', 0.6, 5), ('Equalize', 0.5, 1)],
            [('Contrast', 0.6, 7), ('Sharpness', 0.6, 5)],
            [('Color', 0.7, 7), ('Equalize', 0.5, 8)],
            [('AutoContrast', 0.4, 3), ('Brightness', 0.6, 4)],
            [('Brightness', 0.9, 6), ('Color', 0.2, 8)],
            [('Color', 0.3, 9), ('Equalize', 0.7, 3)],
            [('AutoContrast', 0.2, 5), ('Brightness', 0.4, 1)],
        ]
    
    def __call__(self, img):
        policy = random.choice(self.policies)
        for operation, probability, magnitude in policy:
            if random.random() < probability:
                img = self.apply_operation(img, operation, magnitude)
        return img
    
    def apply_operation(self, img, operation, magnitude):
        # 简化的操作实现
        if operation == 'Invert':
            return transforms.functional.invert(img)
        elif operation == 'Contrast':
            return transforms.functional.adjust_contrast(img, 1 + magnitude * 0.1)
        elif operation == 'Rotate':
            return transforms.functional.rotate(img, magnitude * 2)
        elif operation == 'TranslateX':
            return transforms.functional.affine(img, 0, (magnitude * 0.1, 0), 1, 0)
        elif operation == 'TranslateY':
            return transforms.functional.affine(img, 0, (0, magnitude * 0.1), 1, 0)
        elif operation == 'Sharpness':
            return transforms.functional.adjust_sharpness(img, 1 + magnitude * 0.1)
        elif operation == 'ShearY':
            return transforms.functional.affine(img, 0, (0, 0), 1, (0, magnitude * 0.1))
        elif operation == 'AutoContrast':
            return transforms.functional.autocontrast(img)
        elif operation == 'Equalize':
            return transforms.functional.equalize(img)
        elif operation == 'Posterize':
            return transforms.functional.posterize(img, int(8 - magnitude * 0.5))
        elif operation == 'Color':
            return transforms.functional.adjust_saturation(img, 1 + magnitude * 0.1)
        elif operation == 'Brightness':
            return transforms.functional.adjust_brightness(img, 1 + magnitude * 0.1)
        else:
            return img


def get_transforms(dataset_name, augmentation_type='basic'):
    """
    获取数据变换
    """
    if dataset_name == 'cifar10':
        if augmentation_type == 'basic':
            train_transform = transforms.Compose([
                transforms.RandomRotation(degrees=15),
                transforms.RandomHorizontalFlip(),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])
        elif augmentation_type == 'autoaugment':
            train_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                AutoAugment(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])
        else:
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
            train_transform = transforms.Compose([
                transforms.RandomResizedCrop(64),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        elif augmentation_type == 'autoaugment':
            train_transform = transforms.Compose([
                transforms.RandomResizedCrop(64),
                transforms.RandomHorizontalFlip(),
                AutoAugment(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
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
