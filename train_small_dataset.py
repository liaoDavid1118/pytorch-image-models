#!/usr/bin/env python3
"""
200张图片小数据集专用训练脚本
针对小数据集优化的完整训练流程
"""

import os
import sys
import subprocess
import argparse
from datetime import datetime
from pathlib import Path

def check_dataset_structure(dataset_path):
    """检查数据集结构"""
    dataset_path = Path(dataset_path)
    
    if not dataset_path.exists():
        raise ValueError(f"数据集路径不存在: {dataset_path}")
    
    train_dir = dataset_path / "train"
    val_dir = dataset_path / "val"
    
    if not train_dir.exists():
        raise ValueError(f"缺少训练集目录: {train_dir}")
    
    if not val_dir.exists():
        raise ValueError(f"缺少验证集目录: {val_dir}")
    
    # 统计数据集信息
    train_classes = [d.name for d in train_dir.iterdir() if d.is_dir()]
    val_classes = [d.name for d in val_dir.iterdir() if d.is_dir()]
    
    print(f"✅ 数据集结构检查通过")
    print(f"📊 训练集类别: {len(train_classes)} 个")
    print(f"📊 验证集类别: {len(val_classes)} 个")
    print(f"📊 类别列表: {train_classes}")
    
    # 统计图片数量
    total_train = 0
    total_val = 0
    
    for class_dir in train_dir.iterdir():
        if class_dir.is_dir():
            count = len([f for f in class_dir.iterdir() 
                        if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']])
            total_train += count
            print(f"  训练集 {class_dir.name}: {count} 张")
    
    for class_dir in val_dir.iterdir():
        if class_dir.is_dir():
            count = len([f for f in class_dir.iterdir() 
                        if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']])
            total_val += count
            print(f"  验证集 {class_dir.name}: {count} 张")
    
    print(f"📈 总计: 训练集 {total_train} 张, 验证集 {total_val} 张")
    
    if total_train < 50:
        print("⚠️  警告: 训练集图片过少，建议至少50张")
    
    if total_val < 20:
        print("⚠️  警告: 验证集图片过少，建议至少20张")
    
    return len(train_classes)

def get_optimal_training_config(num_classes, total_images):
    """根据数据集大小获取最优训练配置"""
    
    if total_images <= 100:
        # 极小数据集
        config = {
            "model": "mobilenetv3_small_100",
            "batch_size": 8,
            "lr": 0.0005,
            "epochs": 200,
            "mixup": 0.6,
            "cutmix": 1.2,
            "reprob": 0.4,
            "aa": "rand-m20-mstd0.5-inc1"
        }
    elif total_images <= 200:
        # 小数据集 (推荐配置)
        config = {
            "model": "efficientnet_b0", 
            "batch_size": 16,
            "lr": 0.001,
            "epochs": 150,
            "mixup": 0.4,
            "cutmix": 1.0,
            "reprob": 0.3,
            "aa": "rand-m15-mstd0.5-inc1"
        }
    else:
        # 中等数据集
        config = {
            "model": "efficientnet_b0",
            "batch_size": 32,
            "lr": 0.01,
            "epochs": 100,
            "mixup": 0.2,
            "cutmix": 0.8,
            "reprob": 0.25,
            "aa": "rand-m9-mstd0.5-inc1"
        }
    
    return config

def run_training(dataset_path, num_classes, config, output_dir):
    """执行训练"""

    # 使用虚拟环境的Python
    python_exe = ".venv/Scripts/python.exe" if os.name == 'nt' else ".venv/bin/python"
    if not os.path.exists(python_exe):
        python_exe = "python"  # 回退到系统Python

    # 构建训练命令
    cmd = [
        python_exe, "train.py", str(dataset_path),
        "--model", config["model"],
        "--pretrained",
        "--num-classes", str(num_classes),
        "--batch-size", str(config["batch_size"]),
        "--lr", str(config["lr"]),
        "--epochs", str(config["epochs"]),
        "--opt", "adamw",
        "--weight-decay", "0.01",
        "--sched", "cosine",
        "--warmup-epochs", "10",
        "--aa", config["aa"],
        "--mixup", str(config["mixup"]),
        "--cutmix", str(config["cutmix"]),
        "--reprob", str(config["reprob"]),
        "--drop-path", "0.1",
        "--amp",
        "--model-ema",
        "--model-ema-decay", "0.9999",
        "--patience", "20",
        "--min-lr", "1e-6",
        "--output", str(output_dir),
        "--experiment", "small_dataset_training",
        "--log-interval", "5"
    ]
    
    print(f"🚀 开始训练...")
    print(f"📝 训练命令: {' '.join(cmd)}")
    
    # 执行训练
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print("✅ 训练完成!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ 训练失败: {e}")
        return False

def run_validation(dataset_path, model_name, checkpoint_path, output_dir):
    """执行验证"""

    # 使用虚拟环境的Python
    python_exe = ".venv/Scripts/python.exe" if os.name == 'nt' else ".venv/bin/python"
    if not os.path.exists(python_exe):
        python_exe = "python"  # 回退到系统Python

    cmd = [
        python_exe, "validate.py", str(dataset_path),
        "--model", model_name,
        "--checkpoint", str(checkpoint_path),
        "--batch-size", "32",
        "--amp",
        "--results-file", str(output_dir / "validation_results.csv")
    ]
    
    print(f"🔍 开始验证...")
    print(f"📝 验证命令: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print("✅ 验证完成!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ 验证失败: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="200张图片小数据集训练脚本")
    parser.add_argument("dataset_path", help="数据集路径")
    parser.add_argument("--model", default="auto", help="模型名称 (默认: auto)")
    parser.add_argument("--output-dir", default=None, help="输出目录")
    parser.add_argument("--skip-validation", action="store_true", help="跳过验证")
    
    args = parser.parse_args()
    
    print("🎯 200张图片小数据集训练脚本")
    print("=" * 50)
    
    try:
        # 1. 检查数据集
        print("\n1. 📊 检查数据集结构...")
        num_classes = check_dataset_structure(args.dataset_path)
        
        # 2. 创建输出目录
        if args.output_dir:
            output_dir = Path(args.output_dir)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = Path(f"./output_small_dataset_{timestamp}")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"📁 输出目录: {output_dir}")
        
        # 3. 获取最优配置
        print("\n2. ⚙️  获取最优训练配置...")
        # 简单估算总图片数
        total_images = 200  # 假设值，实际可以通过遍历计算
        config = get_optimal_training_config(num_classes, total_images)
        
        if args.model != "auto":
            config["model"] = args.model
        
        print(f"🎯 选择模型: {config['model']}")
        print(f"📦 批次大小: {config['batch_size']}")
        print(f"📈 学习率: {config['lr']}")
        print(f"🔄 训练轮次: {config['epochs']}")
        print(f"🎨 数据增强: {config['aa']}")
        
        # 4. 执行训练
        print("\n3. 🚀 开始训练...")
        success = run_training(args.dataset_path, num_classes, config, output_dir)
        
        if not success:
            print("❌ 训练失败，请检查错误信息")
            return 1
        
        # 5. 验证模型
        if not args.skip_validation:
            print("\n4. 🔍 验证模型...")
            checkpoint_path = output_dir / "model_best.pth.tar"
            if checkpoint_path.exists():
                run_validation(args.dataset_path, config["model"], 
                             checkpoint_path, output_dir)
            else:
                print("⚠️  找不到最佳模型检查点，跳过验证")
        
        # 6. 总结
        print("\n" + "=" * 50)
        print("🎉 训练流程完成!")
        print(f"📁 结果保存在: {output_dir}")
        print(f"🏆 最佳模型: {output_dir}/model_best.pth.tar")
        print(f"📊 验证结果: {output_dir}/validation_results.csv")
        print(f"📈 查看训练日志: tensorboard --logdir {output_dir}")
        
        print("\n💡 小数据集训练技巧总结:")
        print("1. ✅ 使用预训练模型")
        print("2. ✅ 强数据增强")
        print("3. ✅ 小学习率长训练")
        print("4. ✅ 早停防过拟合")
        print("5. ✅ 模型EMA提升稳定性")
        
        return 0
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
