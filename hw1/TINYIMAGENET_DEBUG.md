# TinyImageNet 训练问题诊断

## 🔍 问题描述
用户报告：除了残差连接模型外，其他模型的test accuracy一直停留在0.5%附近，不更新。

## ✅ 已排查的项目

### 1. 模型架构 ✓
- 所有模型的forward pass正常
- 输出维度正确 (batch_size, 200)
- 梯度能够正常传播

### 2. 数据加载 ✓
- TinyImageNet数据集加载正常
- 200个类别
- 训练集: 100,000 样本
- 测试集: 10,000 样本
- 数据预处理正常（归一化）

### 3. 训练函数 ✓
- `train_model` 函数正常工作
- 模型能够学习（loss下降，accuracy上升）
- 测试显示10个epoch后accuracy能从1%涨到8-10%

### 4. 优化器配置 ✓
```python
{
    'lr': 3e-4,
    'weight_decay': 1e-6
}
```
配置合理，与CIFAR-10保持一致

## 🔴 可能的原因

### 原因1: Epoch数量设置错误 ⚠️
**已发现并修复**: notebook中原本 `num_epochs=3`，已改为50

### 原因2: Batch Size过小 ⚠️
**发现**: notebook中 `batch_size=32`
- 对于TinyImageNet可能偏小
- **建议**: 改为64或128

### 原因3: 数据加载器未正确传递 ⚠️
**可能问题**: 
- 某些训练调用使用了错误的数据加载器
- 数据加载器在某些cell中被重新创建

### 原因4: 模型未进入训练模式 ⚠️
**需要检查**: 
- `model.train()` 是否被正确调用
- 是否有误用 `model.eval()`

### 原因5: Kernel重启导致变量丢失 ⚠️
**可能情况**:
- 用户在中途重启了kernel
- 某些关键变量（如train_loader, test_loader）未重新初始化

## 📋 诊断步骤

### Step 1: 检查数据加载器
```python
# 在训练前打印
print(f"Train loader batches: {len(train_loader)}")
print(f"Test loader batches: {len(test_loader)}")
print(f"Num classes: {num_classes}")

# 检查一个batch
data, target = next(iter(train_loader))
print(f"Batch shape: {data.shape}, Target shape: {target.shape}")
print(f"Target range: [{target.min()}, {target.max()}]")
```

### Step 2: 检查模型初始化
```python
# 训练前
model = get_model('deeper_wider', num_classes).to(device)
print(f"Model device: {next(model.parameters()).device}")
print(f"Model training mode: {model.training}")
```

### Step 3: 监控训练过程
```python
# 在train_model调用后立即检查
print(f"History keys: {history.keys()}")
print(f"Train acc progression: {history['train_acc'][:5]}")
print(f"Test acc progression: {history['test_acc'][:5]}")
```

### Step 4: 检查优化器状态
```python
# 训练后检查
for i, param_group in enumerate(optimizer.param_groups):
    print(f"Param group {i}: lr={param_group['lr']}")
```

## 🔧 推荐的修复方案

### 方案1: 标准化配置
```python
config = {
    'dataset': 'tiny_imagenet',
    'batch_size': 64,  # 增加到64
    'num_workers': 4,
    'num_epochs': 50,  # 确保是50
    'print_every': 1,
}
```

### 方案2: 添加诊断输出
在每个模型训练后添加：
```python
print(f"\n=== {model_name} Training Summary ===")
print(f"Best test accuracy: {max(history['test_acc']):.2f}%")
print(f"Final test accuracy: {history['test_acc'][-1]:.2f}%")
print(f"Accuracy progression:")
for i in [0, 4, 9, 19, 29, 49]:  # 关键epoch
    if i < len(history['test_acc']):
        print(f"  Epoch {i+1}: {history['test_acc'][i]:.2f}%")
```

### 方案3: 确保每次都重新创建优化器
```python
# 对每个模型
model = get_model(model_name, num_classes).to(device)
optimizer_config = get_optimizer_config('Adam')
optimizer = get_optimizer(model, 'Adam', **optimizer_config)
# 不要重用optimizer!
```

## 🎯 预期的正常训练曲线

对于TinyImageNet (50 epochs, batch_size=64):

| Model | Epoch 1 | Epoch 10 | Epoch 30 | Epoch 50 |
|-------|---------|----------|----------|----------|
| Baseline | 1-2% | 8-12% | 25-30% | 35-40% |
| Residual | 1-2% | 10-15% | 30-35% | 45-50% |
| Deeper/Wider | 1-2% | 10-15% | 30-35% | 45-50% |
| SE Attention | 1-2% | 8-12% | 25-30% | 35-40% |

**关键指标**:
- ✅ **Epoch 1**: 1-2% (正常)
- ✅ **Epoch 10**: 应该有8-15%
- ⚠️ **停留在0.5%**: 说明模型完全没有学习

## 📊 快速测试脚本

创建一个简单的测试脚本验证模型能否学习：

```python
import torch
import torch.nn as nn
from models_imagenet import get_model
from dataset_utils import get_dataset_loaders
from training_utils import train_model, get_optimizer, get_optimizer_config

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载数据
train_loader, test_loader, num_classes = get_dataset_loaders(
    'tiny_imagenet', batch_size=64, num_workers=4, augmentation_type='basic'
)

# 测试一个模型（5个epoch快速验证）
model = get_model('deeper_wider', num_classes).to(device)
optimizer_config = get_optimizer_config('Adam')
optimizer = get_optimizer(model, 'Adam', **optimizer_config)
criterion = nn.CrossEntropyLoss()

history = train_model(
    model, train_loader, test_loader,
    num_epochs=5, optimizer=optimizer, criterion=criterion, device=device,
    print_every=1
)

print(f"\nTest accuracy progression:")
for i, acc in enumerate(history['test_acc'], 1):
    print(f"Epoch {i}: {acc:.2f}%")

if history['test_acc'][-1] > 3.0:
    print("\n✅ Model is learning normally!")
else:
    print("\n⚠️ Model is NOT learning - need to investigate!")
```

## 🚨 立即检查项

1. **重启Jupyter kernel**: 清除所有变量
2. **按顺序运行所有cells**: 不要跳过任何cell
3. **检查GPU内存**: 确保没有OOM
4. **检查数据路径**: 确保TinyImageNet数据正确下载
5. **验证num_classes**: 应该是200，不是10

## 💡 临时解决方案

如果问题持续，可以尝试：

1. **减小模型**: 测试baseline模型是否能学习
2. **减少epoch**: 先用5-10个epoch快速验证
3. **增加batch_size**: 改为64或128
4. **检查数据**: 确保数据没有被破坏
5. **重新下载数据**: 删除并重新下载TinyImageNet

## 📞 需要提供的信息

如果问题仍未解决，请提供：

1. 完整的训练输出（前10个epoch）
2. `print(history)` 的输出
3. 模型参数数量
4. GPU显存使用情况
5. 是否有任何错误或警告信息

