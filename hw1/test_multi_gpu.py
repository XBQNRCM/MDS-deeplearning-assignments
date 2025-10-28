#!/usr/bin/env python3
"""
测试多GPU训练配置
"""

import torch
import torch.nn as nn
from models_imagenet import get_model

print("="*60)
print("多GPU训练配置测试")
print("="*60)

# 检测GPU
if torch.cuda.is_available():
    gpu_count = torch.cuda.device_count()
    print(f"\n🎮 检测到 {gpu_count} 个GPU:")
    for i in range(gpu_count):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"    显存: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
    device = torch.device("cuda:0")
    use_multi_gpu = gpu_count > 1
else:
    print("\n❌ 未检测到GPU，将使用CPU")
    device = torch.device("cpu")
    use_multi_gpu = False
    gpu_count = 1

print(f"\n📍 主设备: {device}")
print(f"🚀 多GPU训练: {'✅ 启用' if use_multi_gpu else '❌ 未启用 (单卡/CPU)'}")

# 配置
batch_size = 64 * gpu_count
num_workers = 4 * gpu_count
num_classes = 200

print(f"\n⚙️ 配置:")
print(f"  Batch Size: {batch_size} ({'每GPU '+str(batch_size//gpu_count) if use_multi_gpu else '单卡'})")
print(f"  Num Workers: {num_workers}")

# 创建测试模型
print(f"\n🏗️ 创建测试模型...")
model = get_model('baseline', num_classes)

# 显示原始模型参数量
param_count = sum(p.numel() for p in model.parameters()) / 1_000_000
print(f"  模型参数量: {param_count:.2f}M")

# 多GPU包装
if use_multi_gpu:
    print(f"  🔧 包装为 DataParallel (使用 {gpu_count} 个GPU)")
    model = nn.DataParallel(model)
    print(f"  ✅ DataParallel 设备: {model.device_ids}")
else:
    print(f"  📱 单设备模式")

model = model.to(device)

# 测试前向传播
print(f"\n🧪 测试前向传播...")
test_input = torch.randn(batch_size, 3, 64, 64).to(device)
print(f"  输入形状: {test_input.shape}")

try:
    with torch.no_grad():
        output = model(test_input)
    print(f"  输出形状: {output.shape}")
    print(f"  ✅ 前向传播成功！")
except Exception as e:
    print(f"  ❌ 前向传播失败: {e}")

# 测试反向传播
print(f"\n🧪 测试反向传播...")
try:
    output = model(test_input)
    loss = output.sum()
    loss.backward()
    print(f"  ✅ 反向传播成功！")
except Exception as e:
    print(f"  ❌ 反向传播失败: {e}")

print("\n" + "="*60)
print("✅ 多GPU配置测试完成！")
print("="*60)

