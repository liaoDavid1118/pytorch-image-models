# 🚀 PyTorch Image Models GPU训练完整指南

## 📋 环境确认

✅ **已完成环境配置**
- PyTorch版本: 2.7.1+cu118
- CUDA支持: 已启用
- GPU设备: NVIDIA GeForce RTX 3060
- 虚拟环境: .venv 已激活

## 🎯 训练模式选择

### 1. 单GPU训练 (推荐新手)

#### 快速开始
```bash
# 方法1: 使用批处理脚本 (最简单)
gpu_train.bat

# 方法2: 直接运行Python脚本
python gpu_train.py --data-dir ./dataset --model efficientnet_b0 --num-classes 2 --batch-size 32 --epochs 50 --amp
```

#### 详细参数配置
```bash
python gpu_train.py \
    --data-dir ./dataset \           # 数据集路径
    --model efficientnet_b0 \        # 模型架构
    --num-classes 2 \                # 分类数量 (bilei vs waiguan)
    --batch-size 32 \                # 批次大小 (RTX 3060建议32)
    --epochs 50 \                    # 训练轮数
    --lr 0.001 \                     # 学习率
    --img-size 224 \                 # 图像尺寸
    --output ./output \              # 输出目录
    --amp \                          # 混合精度训练 (节省显存)
    --label-smoothing 0.1            # 标签平滑 (提高泛化)
```

### 2. 多GPU训练 (如果有多张GPU)

```bash
# 分布式训练
python multi_gpu_train.py --distributed --batch-size 64 --epochs 50 --amp
```

## 🔧 关键参数优化

### RTX 3060 优化建议

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| batch-size | 32-64 | 根据显存调整 |
| img-size | 224 | EfficientNet标准尺寸 |
| amp | True | 混合精度，节省50%显存 |
| num-workers | 4 | 数据加载线程数 |

### 模型选择

| 模型 | 参数量 | 显存需求 | 推荐场景 |
|------|--------|----------|----------|
| efficientnet_b0 | 5.3M | 低 | 快速训练 |
| efficientnet_b1 | 7.8M | 中 | 平衡性能 |
| resnet50 | 25.6M | 中 | 经典选择 |
| vit_base_patch16_224 | 86M | 高 | 最佳性能 |

## 📊 训练监控

### 实时监控指标
- **训练损失**: 应该持续下降
- **验证准确率**: 目标 >95%
- **GPU利用率**: 应该 >80%
- **显存使用**: RTX 3060 建议 <10GB

### 日志文件
```
./output/
├── training.log          # 训练日志
├── checkpoint.pth        # 最新检查点
└── best_model.pth        # 最佳模型
```

## 🎛️ 高级训练技巧

### 1. 学习率调度
```python
# 余弦退火 (已内置)
scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

# 阶梯式衰减
scheduler = StepLR(optimizer, step_size=20, gamma=0.1)
```

### 2. 数据增强策略
```python
# 当前使用的增强
- RandomHorizontalFlip(0.5)    # 水平翻转
- RandomRotation(15°)          # 随机旋转
- ColorJitter                  # 颜色抖动
- Normalize                    # 标准化
```

### 3. 混合精度训练
```bash
# 启用AMP可以:
# - 减少50%显存使用
# - 提升20-30%训练速度
# - 保持相同精度
--amp
```

## 🚨 常见问题解决

### 1. 显存不足 (CUDA out of memory)
```bash
# 解决方案:
--batch-size 16        # 减小批次大小
--amp                  # 启用混合精度
--img-size 192         # 减小图像尺寸
```

### 2. 训练速度慢
```bash
# 优化方案:
--num-workers 4        # 增加数据加载线程
--pin-memory True      # 启用内存锁定
--amp                  # 混合精度训练
```

### 3. 过拟合
```bash
# 解决方案:
--label-smoothing 0.1  # 标签平滑
--weight-decay 0.01    # 权重衰减
# 增加数据增强强度
```

## 📈 性能基准

### RTX 3060 预期性能
- **训练速度**: ~100 samples/sec (batch_size=32)
- **单epoch时间**: ~2-3分钟 (1000张图片)
- **总训练时间**: ~2小时 (50 epochs)
- **最终准确率**: >98% (bilei vs waiguan)

## 🔄 恢复训练

```bash
# 从检查点恢复训练
python gpu_train.py --resume ./output/checkpoint.pth --epochs 100
```

## 📝 训练脚本使用示例

### 基础训练
```bash
# 激活虚拟环境
.venv\Scripts\activate

# 开始训练
python gpu_train.py --data-dir ./dataset --amp --epochs 50
```

### 高级训练
```bash
# 自定义配置训练
python gpu_train.py \
    --data-dir ./dataset \
    --model efficientnet_b1 \
    --batch-size 24 \
    --epochs 100 \
    --lr 0.0005 \
    --img-size 240 \
    --amp \
    --label-smoothing 0.15
```

## 🎯 训练完成后

### 模型评估
```bash
# 使用最佳模型进行验证
python validate.py --model ./output/best_model.pth --data-dir ./dataset/test
```

### 模型部署
```bash
# 转换为ONNX格式 (可选)
python export_onnx.py --model ./output/best_model.pth --output model.onnx
```

---

## 🚀 立即开始训练

1. **确保数据集准备完毕**: `./dataset/train/` 和 `./dataset/val/`
2. **运行快速训练**: `gpu_train.bat`
3. **监控训练进度**: 查看 `./output/training.log`
4. **等待训练完成**: 约2小时 (50 epochs)

**祝您训练顺利！** 🎉
