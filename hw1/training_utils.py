"""
训练工具函数
包含训练、评估、优化器配置等工具函数
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ReduceLROnPlateau
import time
import numpy as np
from collections import defaultdict


class MixUpLoss(nn.Module):
    """
    MixUp损失函数
    """
    def __init__(self, criterion):
        super(MixUpLoss, self).__init__()
        self.criterion = criterion
    
    def forward(self, pred, y_a, y_b, lam):
        return lam * self.criterion(pred, y_a) + (1 - lam) * self.criterion(pred, y_b)


class CutMixLoss(nn.Module):
    """
    CutMix损失函数
    """
    def __init__(self, criterion):
        super(CutMixLoss, self).__init__()
        self.criterion = criterion
    
    def forward(self, pred, y_a, y_b, lam):
        return lam * self.criterion(pred, y_a) + (1 - lam) * self.criterion(pred, y_b)


def get_optimizer(model, optimizer_name, lr, weight_decay=1e-4, **kwargs):
    """
    获取优化器 - 改进因子(c)
    """
    if optimizer_name == 'Adam':
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay, **kwargs)
    elif optimizer_name == 'AdamW':
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, **kwargs)
    elif optimizer_name == 'SGD':
        momentum = kwargs.get('momentum', 0.9)
        return optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum, **kwargs)
    elif optimizer_name == 'RMSprop':
        return optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay, **kwargs)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")


def get_scheduler(optimizer, scheduler_name, **kwargs):
    """
    获取学习率调度器 - 改进因子(c)
    """
    if scheduler_name == 'cosine':
        return CosineAnnealingLR(optimizer, T_max=kwargs.get('T_max', 100))
    elif scheduler_name == 'step':
        return StepLR(optimizer, step_size=kwargs.get('step_size', 30), gamma=kwargs.get('gamma', 0.1))
    elif scheduler_name == 'plateau':
        return ReduceLROnPlateau(optimizer, mode='max', factor=kwargs.get('factor', 0.5), 
                                patience=kwargs.get('patience', 10))
    else:
        return None


def train_epoch(model, train_loader, optimizer, criterion, device, augmentation=None):
    """
    训练一个epoch
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # 初始化 batch 增强方法（移到循环外部）
    mixup_fn = None
    cutmix_fn = None
    if augmentation == 'mixup':
        from data_augmentation import MixUp
        mixup_fn = MixUp(alpha=1.0)
        mixup_loss_fn = MixUpLoss(criterion)
    elif augmentation == 'cutmix':
        from data_augmentation import CutMix
        cutmix_fn = CutMix(alpha=1.0)
        cutmix_loss_fn = CutMixLoss(criterion)
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        
        if mixup_fn is not None:
            data, y_a, y_b, lam = mixup_fn(data, target)
            output = model(data)
            loss = mixup_loss_fn(output, y_a, y_b, lam)
        elif cutmix_fn is not None:
            data, y_a, y_b, lam = cutmix_fn(data, target)
            output = model(data)
            loss = cutmix_loss_fn(output, y_a, y_b, lam)
        else:
            output = model(data)
            loss = criterion(output, target)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()  # mixup和cutmix的label是混合的，不方便计算accuracy
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total if total > 0 else 0.0
    
    return epoch_loss, epoch_acc


def evaluate(model, test_loader, criterion, device):
    """
    评估模型
    """
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    test_loss /= len(test_loader)
    test_acc = 100. * correct / total
    
    return test_loss, test_acc


def train_model(model, train_loader, test_loader, num_epochs, optimizer, criterion, 
                device, scheduler=None, augmentation=None, print_every=1):
    """
    完整训练流程
    """
    history = defaultdict(list)
    best_acc = 0.0
    best_model_state = None
    
    # 获取设备信息
    device_name = str(device)
    if device.type == 'cuda':
        device_name = f"{device} ({torch.cuda.get_device_name(device)})"
    
    print(f"开始训练，使用设备: {device_name}")
    print(f"模型参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1_000_000:.2f}M")
    
    # 记录训练开始时间
    training_start_time = time.time()
    
    for epoch in range(num_epochs):
        start_time = time.time()
        
        # 训练
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device, augmentation)
        
        # 评估
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        
        # 学习率调度
        if scheduler:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(test_acc)
            else:
                scheduler.step()
        
        # 记录历史
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        history['lr'].append(optimizer.param_groups[0]['lr'])
        
        # 保存最佳模型
        if test_acc > best_acc:
            best_acc = test_acc
            best_model_state = model.state_dict().copy()
        
        epoch_time = time.time() - start_time
        
        if epoch % print_every == 0 or epoch == num_epochs - 1:
            print(f'Epoch {epoch+1:3d}/{num_epochs}: '
                  f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
                  f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%, '
                  f'LR: {optimizer.param_groups[0]["lr"]:.6f}, '
                  f'Time: {epoch_time:.2f}s')
    
    # 计算总训练时间
    total_training_time = time.time() - training_start_time
    hours = int(total_training_time // 3600)
    minutes = int((total_training_time % 3600) // 60)
    seconds = int(total_training_time % 60)
    
    # 加载最佳模型
    if best_model_state:
        model.load_state_dict(best_model_state)
        print(f"\n{'='*60}")
        print(f"训练完成!")
        print(f"最佳测试准确率: {best_acc:.2f}%")
        print(f"总训练时间: {hours:02d}:{minutes:02d}:{seconds:02d} ({total_training_time:.2f}秒)")
        print(f"{'='*60}")
    
    return history


def get_optimizer_config(optimizer_name):
    """
    获取优化器配置 - 改进因子(c)
    """
    configs = {
        'Adam': {
            'lr': 1e-3,
            'weight_decay': 1e-4,
        },
        'AdamW': {
            'lr': 1e-3,
            'weight_decay': 1e-2,
        },
        'SGD': {
            'lr': 0.1,
            'weight_decay': 1e-4,
            'momentum': 0.9,
        },
        'RMSprop': {
            'lr': 1e-3,
            'weight_decay': 1e-4,
        }
    }
    return configs.get(optimizer_name, configs['Adam'])


def get_scheduler_config(scheduler_name, num_epochs):
    """
    获取调度器配置 - 改进因子(c)
    """
    configs = {
        'cosine': {
            'T_max': num_epochs,
        },
        'step': {
            'step_size': num_epochs // 3,
            'gamma': 0.1,
        },
        'plateau': {
            'factor': 0.5,
            'patience': 10,
        }
    }
    return configs.get(scheduler_name, {})


# 注意：MixUp和CutMix类在data_augmentation.py中定义
# 在train_epoch函数中动态导入以避免循环导入
