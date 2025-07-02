#!/usr/bin/env python3
"""
PyTorch Image Models (timm) 训练指南
"""

# 1. 基础训练命令示例
def basic_training_commands():
    """基础训练命令"""
    print("=== 基础训练命令 ===")
    
    commands = [
        # 基础ImageNet训练
        "python train.py /path/to/imagenet --model resnet50 --batch-size 256 --lr 0.1",
        
        # 使用预训练权重进行微调
        "python train.py /path/to/dataset --model resnet50 --pretrained --num-classes 10 --batch-size 128 --lr 0.01",
        
        # EfficientNet训练
        "python train.py /path/to/dataset --model efficientnet_b0 --batch-size 128 --lr 0.1 --epochs 300",
        
        # Vision Transformer训练
        "python train.py /path/to/dataset --model vit_base_patch16_224 --batch-size 64 --lr 0.001 --opt adamw",
        
        # 分布式训练
        "python -m torch.distributed.launch --nproc_per_node=4 train.py /path/to/dataset --model resnet50",
    ]
    
    for i, cmd in enumerate(commands, 1):
        print(f"{i}. {cmd}")

# 2. 训练参数配置
def training_parameters():
    """训练参数详解"""
    print("\n=== 训练参数配置 ===")
    
    params = {
        "数据集参数": {
            "--data-dir": "数据集根目录",
            "--dataset": "数据集类型 (如 torch/cifar10, hfds/imagenet-1k)",
            "--train-split": "训练集分割名称 (默认: train)",
            "--val-split": "验证集分割名称 (默认: validation)",
            "--num-classes": "分类数量",
            "--input-size": "输入图像尺寸 (如 3 224 224)",
        },
        
        "模型参数": {
            "--model": "模型架构名称",
            "--pretrained": "使用预训练权重",
            "--resume": "从检查点恢复训练",
            "--initial-checkpoint": "加载初始检查点",
        },
        
        "训练参数": {
            "--epochs": "训练轮数",
            "--batch-size": "批次大小",
            "--lr": "学习率",
            "--opt": "优化器 (sgd, adam, adamw, rmsprop等)",
            "--momentum": "动量 (SGD)",
            "--weight-decay": "权重衰减",
        },
        
        "数据增强": {
            "--aa": "AutoAugment策略",
            "--mixup": "Mixup alpha值",
            "--cutmix": "CutMix alpha值",
            "--drop-path": "DropPath比率",
            "--reprob": "Random Erasing概率",
        }
    }
    
    for category, param_dict in params.items():
        print(f"\n{category}:")
        for param, desc in param_dict.items():
            print(f"  {param:<20}: {desc}")

# 3. 不同场景的训练配置
def training_scenarios():
    """不同训练场景的配置"""
    print("\n=== 不同训练场景 ===")
    
    scenarios = {
        "从头训练ImageNet": [
            "python train.py /imagenet",
            "--model resnet50",
            "--batch-size 256",
            "--lr 0.1",
            "--epochs 90",
            "--opt sgd",
            "--momentum 0.9",
            "--weight-decay 1e-4",
            "--sched cosine",
            "--aa rand-m9-mstd0.5-inc1",
            "--mixup 0.2",
            "--cutmix 1.0",
            "--drop-path 0.1"
        ],
        
        "微调预训练模型": [
            "python train.py /custom_dataset",
            "--model resnet50",
            "--pretrained",
            "--num-classes 10",
            "--batch-size 128",
            "--lr 0.01",
            "--epochs 50",
            "--opt sgd",
            "--weight-decay 1e-4",
            "--sched step",
            "--decay-epochs 20"
        ],
        
        "Vision Transformer训练": [
            "python train.py /imagenet",
            "--model vit_base_patch16_224",
            "--batch-size 64",
            "--lr 0.001",
            "--epochs 300",
            "--opt adamw",
            "--weight-decay 0.05",
            "--sched cosine",
            "--warmup-epochs 10",
            "--mixup 0.8",
            "--cutmix 1.0",
            "--drop-path 0.1"
        ],
        
        "轻量级模型训练": [
            "python train.py /dataset",
            "--model mobilenetv3_large_100",
            "--batch-size 256",
            "--lr 0.1",
            "--epochs 200",
            "--opt rmsprop",
            "--decay-rate 0.9",
            "--weight-decay 1e-5",
            "--aa rand-m7-mstd0.5"
        ]
    }
    
    for scenario, commands in scenarios.items():
        print(f"\n{scenario}:")
        command = " \\\n    ".join(commands)
        print(f"  {command}")

# 4. 高级训练选项
def advanced_training_options():
    """高级训练选项"""
    print("\n=== 高级训练选项 ===")
    
    options = {
        "混合精度训练": "--amp",
        "梯度累积": "--grad-accum-steps 4",
        "梯度检查点": "--grad-checkpointing",
        "分布式训练": "python -m torch.distributed.launch --nproc_per_node=4",
        "模型EMA": "--model-ema --model-ema-decay 0.9999",
        "标签平滑": "--smoothing 0.1",
        "随机深度": "--drop-path 0.1",
        "通道最后内存布局": "--channels-last",
        "编译优化": "--torchcompile",
    }
    
    for option, command in options.items():
        print(f"{option:<15}: {command}")

# 5. 配置文件示例
def config_file_example():
    """配置文件示例"""
    print("\n=== YAML配置文件示例 ===")
    
    yaml_config = """
# config.yaml
model: resnet50
pretrained: true
num_classes: 1000
batch_size: 256
epochs: 90
lr: 0.1
opt: sgd
momentum: 0.9
weight_decay: 0.0001
sched: cosine
aa: rand-m9-mstd0.5-inc1
mixup: 0.2
cutmix: 1.0
drop_path: 0.1
amp: true
model_ema: true
model_ema_decay: 0.9999
"""
    
    print(yaml_config)
    print("使用配置文件:")
    print("python train.py /imagenet --config config.yaml")

# 6. 小数据集专用训练方案 (200张图片)
def small_dataset_training_guide():
    """小数据集(200张图片)专用训练指南"""
    print("\n=== 小数据集(200张图片)专用训练方案 ===")

    print("🎯 推荐方案: 轻量级模型 + 强数据增强 + 预训练微调")

    # 推荐模型选择
    recommended_models = {
        "首选模型": {
            "模型": "efficientnet_b0",
            "原因": "参数少(5.3M)、效果好、适合小数据集",
            "预期效果": "在小数据集上表现优秀"
        },
        "备选模型1": {
            "模型": "mobilenetv3_small_100",
            "原因": "极轻量(2.5M参数)、训练快",
            "预期效果": "快速收敛，适合快速验证"
        },
        "备选模型2": {
            "模型": "resnet18",
            "原因": "经典架构、稳定可靠",
            "预期效果": "稳定的基线性能"
        }
    }

    for category, info in recommended_models.items():
        print(f"\n{category}:")
        for key, value in info.items():
            print(f"  {key}: {value}")

    # 最佳训练命令
    print("\n🚀 最佳训练命令:")
    best_command = [
        "python train.py /path/to/your/dataset",
        "--model efficientnet_b0",
        "--pretrained",  # 必须使用预训练权重
        "--num-classes YOUR_CLASS_NUM",  # 替换为您的类别数
        "--batch-size 16",  # 小批次避免过拟合
        "--lr 0.001",  # 较小学习率
        "--epochs 100",  # 更多轮次
        "--opt adamw",  # AdamW优化器
        "--weight-decay 0.01",  # 权重衰减防止过拟合
        "--sched cosine",  # 余弦学习率调度
        "--warmup-epochs 5",  # 预热
        # 强数据增强 - 关键!
        "--aa rand-m15-mstd0.5-inc1",  # 强AutoAugment
        "--mixup 0.4",  # Mixup增强
        "--cutmix 1.0",  # CutMix增强
        "--reprob 0.3",  # Random Erasing
        "--drop-path 0.1",  # DropPath正则化
        # 其他重要设置
        "--amp",  # 混合精度
        "--model-ema",  # 指数移动平均
        "--model-ema-decay 0.9999",
        "--patience 15",  # 早停耐心值
        "--min-lr 1e-6"  # 最小学习率
    ]

    command = " \\\n    ".join(best_command)
    print(f"  {command}")

    # 数据集划分建议
    print("\n📊 数据集划分建议:")
    split_recommendations = {
        "训练集": "160张 (80%)",
        "验证集": "40张 (20%)",
        "建议": "确保每个类别至少有8-10张图片用于验证"
    }

    for key, value in split_recommendations.items():
        print(f"  {key}: {value}")

    # 关键技巧
    print("\n💡 小数据集训练关键技巧:")
    key_tips = [
        "1. 必须使用预训练权重 (--pretrained)",
        "2. 使用强数据增强防止过拟合",
        "3. 较小的批次大小 (16-32)",
        "4. 较小的学习率 (0.001-0.01)",
        "5. 更多的训练轮次 (100-200)",
        "6. 启用早停机制防止过拟合",
        "7. 使用模型EMA提高稳定性",
        "8. 监控训练/验证损失差异"
    ]

    for tip in key_tips:
        print(f"  {tip}")

def small_dataset_monitoring():
    """小数据集训练监控指南"""
    print("\n=== 小数据集训练监控 ===")

    monitoring_points = {
        "过拟合检测": [
            "训练损失持续下降，验证损失上升",
            "训练准确率 >> 验证准确率",
            "解决方案: 增强数据增强、减少模型复杂度"
        ],

        "欠拟合检测": [
            "训练和验证损失都很高",
            "准确率提升缓慢",
            "解决方案: 增加模型复杂度、调整学习率"
        ],

        "理想状态": [
            "训练和验证损失同步下降",
            "验证准确率稳步提升",
            "训练/验证准确率差距 < 5%"
        ]
    }

    for status, indicators in monitoring_points.items():
        print(f"\n{status}:")
        for indicator in indicators:
            print(f"  • {indicator}")

def small_dataset_data_augmentation():
    """小数据集专用数据增强策略"""
    print("\n=== 小数据集专用数据增强策略 ===")

    augmentation_levels = {
        "基础增强 (保守)": [
            "--hflip 0.5",
            "--color-jitter 0.3",
            "--aa rand-m7-mstd0.5",
            "--mixup 0.2",
            "--reprob 0.2"
        ],

        "强增强 (推荐)": [
            "--hflip 0.5",
            "--vflip 0.1",  # 根据数据特性调整
            "--color-jitter 0.4",
            "--aa rand-m15-mstd0.5-inc1",
            "--mixup 0.4",
            "--cutmix 1.0",
            "--reprob 0.3",
            "--remode pixel"
        ],

        "极强增强 (数据极少时)": [
            "--hflip 0.5",
            "--vflip 0.2",
            "--color-jitter 0.5",
            "--aa rand-m20-mstd0.5-inc1",
            "--mixup 0.6",
            "--cutmix 1.2",
            "--reprob 0.4",
            "--trivial-augment"  # 如果支持
        ]
    }

    for level, augs in augmentation_levels.items():
        print(f"\n{level}:")
        for aug in augs:
            print(f"  {aug}")

if __name__ == "__main__":
    basic_training_commands()
    training_parameters()
    training_scenarios()
    advanced_training_options()
    config_file_example()

    # 新增小数据集专用指南
    small_dataset_training_guide()
    small_dataset_monitoring()
    small_dataset_data_augmentation()

    print("\n=== 训练监控 ===")
    print("1. TensorBoard: 训练过程中会自动生成日志")
    print("2. Wandb: 添加 --experiment wandb_project_name")
    print("3. 检查点保存: 自动保存在 ./output/ 目录")
    print("4. 最佳模型: 根据验证集性能自动保存")

    print("\n=== 200张图片训练总结 ===")
    print("🎯 核心策略: 预训练模型 + 强数据增强 + 小学习率 + 早停")
    print("📈 预期效果: 在小数据集上也能达到不错的分类效果")
    print("⚠️  注意事项: 密切监控过拟合，及时调整超参数")
