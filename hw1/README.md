# 深度学习作业1：图像分类实验

## 项目结构

```
hw1/
├── Example.ipynb              # 原始基线代码（请勿修改）
├── CIFAR10_Experiments.ipynb  # CIFAR-10实验主文件
├── TinyImageNet_Experiments.ipynb  # Tiny-ImageNet实验主文件
├── models.py                  # 模型定义
├── dataset_utils.py           # 数据集工具
├── training_utils.py          # 训练工具
├── data_augmentation.py       # 数据增强
├── README.md                  # 项目说明
└── data/                      # 数据目录（自动创建）
```

## 实验设计

### 目标
基于Scoring Criteria 2和3，在CIFAR-10和tiny-imagenet数据集上实验5个改进因子来提升图像分类性能。

### 5个改进因子

1. **残差连接机制 (Residual Connections)**
   - 解决梯度消失问题
   - 实现：ResidualBlock类

2. **更深/更宽的网络架构**
   - 增加模型容量
   - 实现：DeeperWiderCNN类

3. **更好的优化器和学习率调度**
   - AdamW + 余弦退火调度器
   - 实现：get_optimizer, get_scheduler函数

4. **高级数据增强**
   - MixUp, CutMix, AutoAugment
   - 实现：MixUp, CutMix, AutoAugment类

5. **注意力机制**
   - SE Block, Self-Attention
   - 实现：SEBlock, SelfAttention类

### 最佳组合模型

**BestModel** - Deeper/Wider + Residual
- 结合了因子(a)残差连接和因子(b)更深更宽网络
- 与DeeperWiderCNN对齐：8层卷积，4个残差块
- 通道数：128→256→512→1024
- 参数量：~20M
- 预期性能提升：在CIFAR-10上可达92%+准确率

## 使用方法

### 环境要求
```bash
pip install torch torchvision matplotlib pandas numpy requests
```

### 运行实验

1. **CIFAR-10实验**：
   ```bash
   jupyter notebook CIFAR10_Experiments.ipynb
   ```

2. **Tiny-ImageNet实验**：
   ```bash
   jupyter notebook TinyImageNet_Experiments.ipynb
   ```

### 实验流程

1. **基线模型**：使用Example.ipynb的架构作为基线
2. **单因子实验**：逐个测试每个改进因子
3. **组合实验**：组合最佳因子进行最终实验
4. **结果分析**：可视化结果，分析各因子贡献

## 预期结果

### CIFAR-10
- 基线准确率：~86.5%
- 目标准确率：>90%

### Tiny-ImageNet
- 基线准确率：~30-40%
- 目标准确率：>50%

## 主要特性

- **模块化设计**：代码结构清晰，易于扩展
- **完整实验流程**：从数据加载到结果分析
- **可视化支持**：训练曲线、结果对比图表
- **可复现性**：固定随机种子，详细注释

## 注意事项

1. 请勿修改`Example.ipynb`文件
2. 首次运行会自动下载数据集
3. 建议使用GPU加速训练
4. 实验时间较长，建议分批运行
