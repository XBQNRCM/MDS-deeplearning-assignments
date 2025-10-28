# 单机多卡训练使用指南

## 📋 概述

TinyImageNet实验notebook已经配置为**自动支持单机多卡训练**，同时完全兼容单卡和CPU训练。

## 🚀 特性

- ✅ **自动检测GPU数量**：代码会自动检测系统中的GPU数量
- ✅ **自动切换模式**：单卡时使用标准训练，多卡时自动使用DataParallel
- ✅ **动态batch size**：多GPU时自动调整batch size（每个GPU保持64）
- ✅ **完全透明**：无需手动配置，代码自动处理所有细节
- ✅ **模型保存兼容**：正确处理DataParallel包装，保存和加载无缝衔接

## 🎮 检测到的GPU配置

当前系统配置：
```
🎮 检测到 1 个GPU:
  GPU 0: NVIDIA GeForce RTX 5090 (31.37 GB)
```

## 📊 性能预期

### 单卡训练
- Batch Size: 64
- 训练速度: ~23秒/epoch（TinyImageNet，baseline模型）

### 多卡训练（理论值）
- **2卡**: Batch Size: 128，预期加速比: 1.7-1.9x
- **4卡**: Batch Size: 256，预期加速比: 3.0-3.5x
- **8卡**: Batch Size: 512，预期加速比: 5.5-6.5x

> 注意：实际加速比取决于模型大小、网络带宽等因素。DataParallel的主卡（GPU 0）负载会稍重。

## 💻 使用方法

### 方式1：直接运行notebook

打开 `TinyImageNet_Experiments.ipynb`，运行所有cell即可。代码会自动：
1. 检测GPU数量
2. 配置合适的batch size和num workers
3. 自动包装模型为DataParallel（如果有多GPU）
4. 显示详细的训练信息

### 方式2：测试多GPU配置

运行测试脚本验证配置：
```bash
cd hw1
python test_multi_gpu.py
```

## 🔧 配置说明

### Cell 1: GPU检测和配置
```python
# 自动检测GPU并配置
if torch.cuda.is_available():
    gpu_count = torch.cuda.device_count()
    print(f"🎮 检测到 {gpu_count} 个GPU:")
    for i in range(gpu_count):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    device = torch.device("cuda:0")
    use_multi_gpu = gpu_count > 1
else:
    device = torch.device("cpu")
    use_multi_gpu = False
    gpu_count = 1
```

### Cell 3: 动态Batch Size配置
```python
config = {
    'dataset': 'tiny_imagenet',
    'batch_size': 64 * gpu_count,  # 多GPU时自动扩展
    'num_workers': 4 * gpu_count,  # 多GPU时增加worker
    'num_epochs': 50,
    'print_every': 1,
}
```

### 模型包装函数
```python
def wrap_model_multi_gpu(model, use_multi_gpu=use_multi_gpu):
    """自动包装模型为DataParallel（如果需要）"""
    if use_multi_gpu:
        print(f"  🚀 使用 DataParallel 在 {gpu_count} 个GPU上训练")
        model = nn.DataParallel(model)
    else:
        print(f"  📱 使用单设备训练")
    return model
```

### 所有模型创建位置都已更新
```python
# 标准模式（已应用到所有实验）
model = get_model('baseline', num_classes)
model = wrap_model_multi_gpu(model)  # 自动处理多GPU
model = model.to(device)
```

## 📝 代码修改总结

### 1. TinyImageNet_Experiments.ipynb
- ✅ Cell 1: 添加多GPU检测和配置
- ✅ Cell 3: 动态batch size配置 + 模型包装函数
- ✅ Cell 6: Baseline模型 - 添加多GPU包装
- ✅ Cell 8: 残差模型 - 添加多GPU包装
- ✅ Cell 10: 更深更宽模型 - 添加多GPU包装
- ✅ Cell 12: 优化器对比 - 添加多GPU包装
- ✅ Cell 14: 数据增强对比 - 添加多GPU包装
- ✅ Cell 16: 注意力机制 - 添加多GPU包装
- ✅ Cell 19: 最佳组合模型 - 添加多GPU包装
- ✅ Cell 22: 模型保存 - 正确处理DataParallel

### 2. training_utils.py
- ✅ `train_model`函数: 更新设备信息显示，正确处理DataParallel

## ⚠️ 注意事项

### 1. 模型保存和加载
代码已正确处理DataParallel的保存和加载：

**保存时**（自动处理）：
```python
model_to_save = best_model.module if isinstance(best_model, nn.DataParallel) else best_model
torch.save(model_to_save.state_dict(), model_path)
```

**加载时**（无需特殊处理）：
```python
model = BestModel(num_classes=200)
model.load_state_dict(torch.load(model_path))
# 如果需要多GPU推理，再包装
if use_multi_gpu:
    model = nn.DataParallel(model)
model = model.to(device)
```

### 2. DataParallel的限制
- GPU 0会占用更多显存（需要存储模型副本和聚合梯度）
- 实际加速比通常低于GPU数量（通信开销）
- 适合模型较大、batch size较大的场景

### 3. 如果需要更高性能
如果需要更高的多GPU效率，可以考虑：
- **DistributedDataParallel (DDP)**: 更高效，但需要脚本化训练（不适合notebook）
- **增大模型**: 更大的模型能更好地利用多GPU
- **增大batch size**: 在显存允许的情况下

## 🧪 验证清单

运行以下检查确保配置正确：

- [ ] Cell 1 显示正确的GPU数量和型号
- [ ] Cell 3 显示合适的batch size（单GPU: 64，多GPU: 64×N）
- [ ] 训练开始时显示正确的设备信息（单GPU或DataParallel）
- [ ] 模型参数量显示正确
- [ ] 训练正常进行，无OOM错误
- [ ] 多GPU时GPU利用率均衡（可用nvidia-smi监控）

## 📊 监控多GPU训练

### 实时监控GPU使用
```bash
# 每2秒更新一次
watch -n 2 nvidia-smi

# 或者持续输出
nvidia-smi dmon -s u
```

### 检查训练输出
训练开始时应该看到：
```
开始训练，使用设备: DataParallel on 2 GPUs: [0, 1]
模型参数量: 13.59M
```

## 🎯 总结

现在 `TinyImageNet_Experiments.ipynb` 已经：
- ✅ 完全支持单机多卡训练
- ✅ 自动检测和配置
- ✅ 完全兼容单卡和CPU
- ✅ 无需手动修改代码
- ✅ 保存/加载正确处理

**只需运行notebook，一切都会自动工作！** 🚀

