"""
数据集工具函数
包含CIFAR-10和tiny-imagenet数据集的加载和预处理
"""

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
import requests
import zipfile
import tarfile
from PIL import Image
import numpy as np


def download_tiny_imagenet(data_dir='./data'):
    """
    下载并解压tiny-imagenet数据集
    """
    url = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'
    zip_path = os.path.join(data_dir, 'tiny-imagenet-200.zip')
    extract_path = os.path.join(data_dir, 'tiny-imagenet-200')
    
    if not os.path.exists(extract_path):
        print("下载tiny-imagenet数据集...")
        if not os.path.exists(zip_path):
            response = requests.get(url, stream=True)
            with open(zip_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
        
        print("解压tiny-imagenet数据集...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
    
    return extract_path


class TinyImageNetDataset(torch.utils.data.Dataset):
    """
    Tiny-ImageNet数据集类
    """
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        
        # 加载类别信息
        with open(os.path.join(root_dir, 'words.txt'), 'r') as f:
            self.class_names = {}
            for line in f:
                parts = line.strip().split('\t')
                self.class_names[parts[0]] = parts[1]
        
        # 加载类别列表
        with open(os.path.join(root_dir, 'wnids.txt'), 'r') as f:
            self.classes = [line.strip() for line in f]
        
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # 加载数据
        self.samples = []
        if split == 'train':
            self._load_train_data()
        elif split == 'val':
            self._load_val_data()
        else:
            raise ValueError(f"Unknown split: {split}")
    
    def _load_train_data(self):
        """加载训练数据"""
        train_dir = os.path.join(self.root_dir, 'train')
        for class_name in self.classes:
            class_dir = os.path.join(train_dir, class_name, 'images')
            if os.path.exists(class_dir):
                for img_name in os.listdir(class_dir):
                    if img_name.endswith('.JPEG'):
                        img_path = os.path.join(class_dir, img_name)
                        self.samples.append((img_path, self.class_to_idx[class_name]))
    
    def _load_val_data(self):
        """加载验证数据"""
        val_dir = os.path.join(self.root_dir, 'val')
        val_annotations_file = os.path.join(val_dir, 'val_annotations.txt')
        
        with open(val_annotations_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                img_name = parts[0]
                class_name = parts[1]
                if class_name in self.class_to_idx:
                    img_path = os.path.join(val_dir, 'images', img_name)
                    self.samples.append((img_path, self.class_to_idx[class_name]))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_cifar10_loaders(batch_size=64, num_workers=2, augmentation_type='basic'):
    """
    获取CIFAR-10数据加载器
    """
    from data_augmentation import get_transforms
    
    train_transform, test_transform = get_transforms('cifar10', augmentation_type)
    
    train_dataset = datasets.CIFAR10(
        root='./data', train=True, download=True, transform=train_transform
    )
    test_dataset = datasets.CIFAR10(
        root='./data', train=False, download=True, transform=test_transform
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    
    return train_loader, test_loader, 10  # 10个类别


def get_tiny_imagenet_loaders(batch_size=64, num_workers=2, augmentation_type='basic'):
    """
    获取tiny-imagenet数据加载器
    """
    from data_augmentation import get_transforms
    
    # 下载数据集
    data_dir = download_tiny_imagenet()
    
    train_transform, test_transform = get_transforms('tiny_imagenet', augmentation_type)
    
    train_dataset = TinyImageNetDataset(
        root_dir=data_dir, split='train', transform=train_transform
    )
    test_dataset = TinyImageNetDataset(
        root_dir=data_dir, split='val', transform=test_transform
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    
    return train_loader, test_loader, 200  # 200个类别


def get_dataset_loaders(dataset_name, batch_size=64, num_workers=2, augmentation_type='basic'):
    """
    获取指定数据集的数据加载器
    """
    if dataset_name == 'cifar10':
        return get_cifar10_loaders(batch_size, num_workers, augmentation_type)
    elif dataset_name == 'tiny_imagenet':
        return get_tiny_imagenet_loaders(batch_size, num_workers, augmentation_type)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
