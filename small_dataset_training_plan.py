#!/usr/bin/env python3
"""
200张图片小数据集最佳训练方案
针对小数据集优化的完整训练策略
"""

def optimal_training_plan():
    """200张图片的最佳训练方案"""
    print("🎯 200张图片小数据集最佳训练方案")
    print("=" * 50)
    
    # 1. 推荐模型选择
    print("\n1. 📊 推荐模型选择 (按优先级排序)")
    print("-" * 30)
    
    models = [
        {
            "模型": "efficientnet_b0",
            "参数量": "5.3M",
            "优势": "专为小数据集设计，效果最佳",
            "推荐指数": "⭐⭐⭐⭐⭐"
        },
        {
            "模型": "mobilenetv3_small_100", 
            "参数量": "2.5M",
            "优势": "轻量级，训练快，不易过拟合",
            "推荐指数": "⭐⭐⭐⭐"
        },
        {
            "模型": "resnet18",
            "参数量": "11.7M", 
            "优势": "经典稳定，基线性能好",
            "推荐指数": "⭐⭐⭐"
        }
    ]
    
    for i, model in enumerate(models, 1):
        print(f"{i}. {model['模型']}")
        print(f"   参数量: {model['参数量']}")
        print(f"   优势: {model['优势']}")
        print(f"   推荐指数: {model['推荐指数']}")
        print()

def best_training_command():
    """最佳训练命令"""
    print("2. 🚀 最佳训练命令")
    print("-" * 30)
    
    print("假设您有10个类别，数据集结构如下:")
    print("dataset/")
    print("├── train/ (160张)")
    print("│   ├── class1/")
    print("│   ├── class2/")
    print("│   └── ...")
    print("└── val/ (40张)")
    print("    ├── class1/")
    print("    └── ...")
    print()
    
    command = """python train.py /path/to/your/dataset \\
    --model efficientnet_b0 \\
    --pretrained \\
    --num-classes 10 \\
    --batch-size 16 \\
    --lr 0.001 \\
    --epochs 150 \\
    --opt adamw \\
    --weight-decay 0.01 \\
    --sched cosine \\
    --warmup-epochs 10 \\
    --aa rand-m15-mstd0.5-inc1 \\
    --mixup 0.4 \\
    --cutmix 1.0 \\
    --reprob 0.3 \\
    --drop-path 0.1 \\
    --amp \\
    --model-ema \\
    --model-ema-decay 0.9999 \\
    --patience 20 \\
    --min-lr 1e-6 \\
    --output ./output \\
    --experiment small_dataset_exp"""
    
    print("最佳训练命令:")
    print(command)

def parameter_explanation():
    """参数详细解释"""
    print("\n3. 📋 关键参数解释")
    print("-" * 30)
    
    params = {
        "--model efficientnet_b0": "选择EfficientNet-B0，最适合小数据集",
        "--pretrained": "🔥 必须！使用ImageNet预训练权重",
        "--batch-size 16": "小批次防止过拟合，GPU内存友好",
        "--lr 0.001": "较小学习率，避免破坏预训练特征",
        "--epochs 150": "更多轮次让模型充分学习",
        "--opt adamw": "AdamW优化器，适合微调",
        "--weight-decay 0.01": "权重衰减防止过拟合",
        "--aa rand-m15-mstd0.5-inc1": "🔥 强数据增强，关键技巧！",
        "--mixup 0.4": "Mixup增强，增加数据多样性",
        "--cutmix 1.0": "CutMix增强，提高泛化能力",
        "--reprob 0.3": "Random Erasing，防止过拟合",
        "--model-ema": "指数移动平均，提高稳定性",
        "--patience 20": "早停机制，防止过拟合"
    }
    
    for param, explanation in params.items():
        print(f"{param:<25}: {explanation}")

def training_stages():
    """分阶段训练策略"""
    print("\n4. 📈 分阶段训练策略")
    print("-" * 30)
    
    stages = {
        "阶段1: 特征提取 (推荐)": {
            "轮次": "0-50轮",
            "策略": "冻结backbone，只训练分类头",
            "学习率": "0.01",
            "命令": "--freeze-backbone --lr 0.01 --epochs 50"
        },
        
        "阶段2: 端到端微调": {
            "轮次": "50-150轮", 
            "策略": "解冻所有层，端到端训练",
            "学习率": "0.001",
            "命令": "--lr 0.001 --epochs 150"
        }
    }
    
    for stage, info in stages.items():
        print(f"\n{stage}:")
        for key, value in info.items():
            print(f"  {key}: {value}")
    
    print("\n💡 两阶段训练命令:")
    print("# 阶段1")
    print("python train.py /dataset --model efficientnet_b0 --pretrained --freeze-backbone --lr 0.01 --epochs 50")
    print("\n# 阶段2") 
    print("python train.py /dataset --model efficientnet_b0 --resume ./output/model_best.pth.tar --lr 0.001 --epochs 150")

def data_augmentation_strategy():
    """数据增强策略"""
    print("\n5. 🎨 数据增强策略")
    print("-" * 30)
    
    print("小数据集的数据增强是成功的关键！")
    
    strategies = {
        "基础增强": {
            "适用": "保守训练，确保稳定性",
            "参数": "--hflip 0.5 --color-jitter 0.3 --aa rand-m7-mstd0.5 --mixup 0.2"
        },
        
        "强增强 (推荐)": {
            "适用": "200张图片的最佳选择",
            "参数": "--aa rand-m15-mstd0.5-inc1 --mixup 0.4 --cutmix 1.0 --reprob 0.3"
        },
        
        "极强增强": {
            "适用": "数据极少或效果不佳时",
            "参数": "--aa rand-m20-mstd0.5-inc1 --mixup 0.6 --cutmix 1.2 --reprob 0.4"
        }
    }
    
    for strategy, info in strategies.items():
        print(f"\n{strategy}:")
        print(f"  适用场景: {info['适用']}")
        print(f"  参数设置: {info['参数']}")

def monitoring_and_debugging():
    """训练监控和调试"""
    print("\n6. 📊 训练监控和调试")
    print("-" * 30)
    
    monitoring_tips = [
        "🔍 关键指标监控:",
        "  • 训练损失 vs 验证损失",
        "  • 训练准确率 vs 验证准确率", 
        "  • 学习率变化曲线",
        "",
        "⚠️  过拟合信号:",
        "  • 验证损失开始上升",
        "  • 训练准确率 >> 验证准确率 (差距>10%)",
        "  • 验证准确率不再提升",
        "",
        "🛠️  解决过拟合:",
        "  • 增强数据增强强度",
        "  • 增加weight-decay",
        "  • 减少模型复杂度",
        "  • 启用早停机制",
        "",
        "📈 欠拟合信号:",
        "  • 训练和验证损失都很高",
        "  • 准确率提升缓慢",
        "",
        "🛠️  解决欠拟合:",
        "  • 增加训练轮次",
        "  • 调高学习率",
        "  • 减少正则化强度"
    ]
    
    for tip in monitoring_tips:
        print(tip)

def expected_results():
    """预期效果和基准"""
    print("\n7. 🎯 预期效果和基准")
    print("-" * 30)
    
    benchmarks = {
        "优秀结果": {
            "验证准确率": "> 85%",
            "训练/验证差距": "< 5%",
            "收敛轮次": "< 100轮"
        },
        
        "良好结果": {
            "验证准确率": "75-85%", 
            "训练/验证差距": "5-10%",
            "收敛轮次": "100-150轮"
        },
        
        "需要调优": {
            "验证准确率": "< 75%",
            "训练/验证差距": "> 10%",
            "收敛轮次": "> 150轮"
        }
    }
    
    for level, metrics in benchmarks.items():
        print(f"\n{level}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value}")

def quick_start_checklist():
    """快速开始检查清单"""
    print("\n8. ✅ 快速开始检查清单")
    print("-" * 30)
    
    checklist = [
        "□ 数据集按ImageFolder格式组织",
        "□ 训练集160张，验证集40张",
        "□ 每个类别至少有4-5张验证图片",
        "□ 安装timm: pip install timm",
        "□ 确认GPU可用 (推荐)",
        "□ 准备足够的存储空间 (至少2GB)",
        "□ 设置合适的类别数量参数",
        "□ 选择EfficientNet-B0模型",
        "□ 启用强数据增强",
        "□ 使用预训练权重",
        "□ 设置早停机制",
        "□ 准备监控训练过程"
    ]
    
    for item in checklist:
        print(f"  {item}")

if __name__ == "__main__":
    optimal_training_plan()
    best_training_command()
    parameter_explanation()
    training_stages()
    data_augmentation_strategy()
    monitoring_and_debugging()
    expected_results()
    quick_start_checklist()
    
    print("\n" + "="*50)
    print("🎉 总结: 200张图片训练成功秘诀")
    print("="*50)
    print("1. 🔥 必须使用预训练模型 (efficientnet_b0)")
    print("2. 🎨 强数据增强是关键 (AutoAugment + Mixup + CutMix)")
    print("3. 📉 小学习率 + 长训练 (0.001, 150轮)")
    print("4. 🛡️  防过拟合 (早停 + 权重衰减 + EMA)")
    print("5. 📊 密切监控训练过程")
    print("\n预期效果: 验证准确率可达80-90%+ 🚀")
