#!/usr/bin/env python3
"""
PyTorch Image Models (timm) 完整使用示例
从模型创建到训练验证的完整流程
"""

import timm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os

def complete_workflow_example():
    """完整的工作流程示例"""
    print("=== PyTorch Image Models (timm) 完整使用示例 ===")
    
    # 1. 模型创建和配置
    print("\n1. 模型创建和配置")
    print("-" * 30)
    
    # 查看可用模型
    print("可用的ResNet模型:")
    resnet_models = timm.list_models('resnet*', pretrained=True)
    print(f"  共 {len(resnet_models)} 个预训练ResNet模型")
    print(f"  示例: {resnet_models[:5]}")
    
    # 创建模型
    model = timm.create_model('resnet50', pretrained=True, num_classes=10)
    print(f"\n创建的模型: ResNet50")
    print(f"  参数数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  分类数量: {model.num_classes}")
    
    # 查看模型配置
    print(f"\n模型预训练配置:")
    config = model.pretrained_cfg
    print(f"  输入尺寸: {config['input_size']}")
    print(f"  均值: {config['mean']}")
    print(f"  标准差: {config['std']}")
    
    # 2. 数据准备
    print("\n2. 数据准备")
    print("-" * 30)
    
    # 数据变换
    data_config = timm.data.resolve_data_config({}, model=model)
    transform = timm.data.create_transform(**data_config, is_training=True)
    
    print("训练数据变换:")
    print(f"  {transform}")
    
    # 3. 训练配置示例
    print("\n3. 训练配置示例")
    print("-" * 30)
    
    training_commands = {
        "基础训练": [
            "python train.py /path/to/dataset",
            "--model resnet50",
            "--pretrained",
            "--num-classes 10",
            "--batch-size 128",
            "--lr 0.01",
            "--epochs 50"
        ],
        
        "高级训练": [
            "python train.py /path/to/dataset",
            "--model resnet50",
            "--pretrained",
            "--num-classes 10",
            "--batch-size 128",
            "--lr 0.01",
            "--epochs 100",
            "--opt adamw",
            "--weight-decay 0.05",
            "--sched cosine",
            "--warmup-epochs 5",
            "--mixup 0.2",
            "--cutmix 1.0",
            "--aa rand-m9-mstd0.5-inc1",
            "--amp",
            "--model-ema"
        ]
    }
    
    for name, commands in training_commands.items():
        print(f"\n{name}:")
        command = " \\\n    ".join(commands)
        print(f"  {command}")
    
    # 4. 验证配置示例
    print("\n4. 验证配置示例")
    print("-" * 30)
    
    validation_commands = [
        "# 验证预训练模型",
        "python validate.py /path/to/dataset --model resnet50 --pretrained",
        "",
        "# 验证训练后的模型",
        "python validate.py /path/to/dataset --model resnet50 --checkpoint ./output/model_best.pth.tar",
        "",
        "# 高性能验证",
        "python validate.py /path/to/dataset --model resnet50 --pretrained --batch-size 256 --amp -j 8"
    ]
    
    for cmd in validation_commands:
        print(f"  {cmd}")

def practical_examples():
    """实际应用示例"""
    print("\n=== 实际应用示例 ===")
    
    # 示例1: 图像分类推理
    print("\n示例1: 图像分类推理")
    print("-" * 30)
    
    inference_code = '''
import timm
import torch
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

# 加载预训练模型
model = timm.create_model('resnet50', pretrained=True)
model.eval()

# 准备数据变换
config = resolve_data_config({}, model=model)
transform = create_transform(**config)

# 加载和预处理图像
img = Image.open('image.jpg')
tensor = transform(img).unsqueeze(0)  # 添加batch维度

# 推理
with torch.no_grad():
    output = model(tensor)
    probabilities = torch.nn.functional.softmax(output[0], dim=0)

# 获取预测结果
top5_prob, top5_catid = torch.topk(probabilities, 5)
for i in range(top5_prob.size(0)):
    print(f"类别 {top5_catid[i]}: {top5_prob[i].item():.4f}")
'''
    
    print(inference_code)
    
    # 示例2: 特征提取
    print("\n示例2: 特征提取")
    print("-" * 30)
    
    feature_extraction_code = '''
import timm
import torch

# 创建特征提取器
model = timm.create_model('resnet50', pretrained=True, features_only=True)
model.eval()

# 输入图像
x = torch.randn(1, 3, 224, 224)

# 提取特征
with torch.no_grad():
    features = model(x)

# 显示特征图尺寸
for i, feat in enumerate(features):
    print(f"特征层 {i}: {feat.shape}")

# 输出示例:
# 特征层 0: torch.Size([1, 64, 112, 112])
# 特征层 1: torch.Size([1, 256, 56, 56])
# 特征层 2: torch.Size([1, 512, 28, 28])
# 特征层 3: torch.Size([1, 1024, 14, 14])
# 特征层 4: torch.Size([1, 2048, 7, 7])
'''
    
    print(feature_extraction_code)

def dataset_preparation_guide():
    """数据集准备指南"""
    print("\n=== 数据集准备指南 ===")
    
    # ImageFolder格式
    print("\n标准ImageFolder格式:")
    folder_structure = '''
dataset/
├── train/
│   ├── class1/
│   │   ├── img1.jpg
│   │   ├── img2.jpg
│   │   └── ...
│   ├── class2/
│   │   ├── img1.jpg
│   │   └── ...
│   └── ...
└── val/
    ├── class1/
    └── class2/
'''
    print(folder_structure)
    
    # 数据集创建脚本
    print("数据集创建脚本:")
    dataset_script = '''
import os
import shutil
from sklearn.model_selection import train_test_split

def create_imagefolder_dataset(source_dir, target_dir, test_size=0.2):
    """将图像按类别组织成ImageFolder格式"""
    
    # 创建目录
    train_dir = os.path.join(target_dir, 'train')
    val_dir = os.path.join(target_dir, 'val')
    
    # 假设source_dir包含按类别命名的子目录
    for class_name in os.listdir(source_dir):
        class_path = os.path.join(source_dir, class_name)
        if os.path.isdir(class_path):
            # 获取该类别的所有图像
            images = [f for f in os.listdir(class_path) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            # 分割训练集和验证集
            train_imgs, val_imgs = train_test_split(
                images, test_size=test_size, random_state=42
            )
            
            # 创建类别目录
            os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
            os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)
            
            # 复制文件
            for img in train_imgs:
                shutil.copy2(
                    os.path.join(class_path, img),
                    os.path.join(train_dir, class_name, img)
                )
            
            for img in val_imgs:
                shutil.copy2(
                    os.path.join(class_path, img),
                    os.path.join(val_dir, class_name, img)
                )

# 使用示例
create_imagefolder_dataset('/source/images', '/dataset')
'''
    print(dataset_script)

def performance_optimization():
    """性能优化建议"""
    print("\n=== 性能优化建议 ===")
    
    optimizations = {
        "训练优化": [
            "使用混合精度训练 (--amp)",
            "启用通道最后内存布局 (--channels-last)",
            "使用多GPU训练 (DistributedDataParallel)",
            "优化数据加载 (增加workers数量)",
            "使用梯度累积减少内存使用",
            "启用模型编译 (--torchcompile)"
        ],
        
        "推理优化": [
            "使用TorchScript或ONNX导出",
            "量化模型减少内存占用",
            "批量推理提高吞吐量",
            "使用TensorRT加速",
            "启用混合精度推理"
        ],
        
        "内存优化": [
            "使用梯度检查点 (--grad-checkpointing)",
            "减少批次大小",
            "使用CPU卸载",
            "清理不必要的中间变量"
        ]
    }
    
    for category, tips in optimizations.items():
        print(f"\n{category}:")
        for tip in tips:
            print(f"  • {tip}")

def troubleshooting_guide():
    """故障排除指南"""
    print("\n=== 故障排除指南 ===")
    
    issues = {
        "CUDA内存不足": [
            "减少批次大小 (--batch-size)",
            "启用梯度检查点 (--grad-checkpointing)",
            "使用混合精度训练 (--amp)",
            "减少模型尺寸或使用更小的模型"
        ],
        
        "训练速度慢": [
            "增加数据加载工作进程 (-j 8)",
            "使用SSD存储数据集",
            "启用混合精度训练",
            "使用更大的批次大小",
            "检查GPU利用率"
        ],
        
        "精度不收敛": [
            "检查学习率设置",
            "验证数据集标签正确性",
            "尝试不同的优化器",
            "增加训练轮数",
            "调整数据增强策略"
        ],
        
        "模型加载错误": [
            "检查模型名称拼写",
            "确认预训练权重可用",
            "验证网络连接",
            "检查timm版本兼容性"
        ]
    }
    
    for issue, solutions in issues.items():
        print(f"\n{issue}:")
        for solution in solutions:
            print(f"  • {solution}")

if __name__ == "__main__":
    complete_workflow_example()
    practical_examples()
    dataset_preparation_guide()
    performance_optimization()
    troubleshooting_guide()
    
    print("\n=== 总结 ===")
    print("PyTorch Image Models (timm) 提供了:")
    print("1. 丰富的预训练模型库")
    print("2. 统一的模型接口")
    print("3. 高效的训练和验证脚本")
    print("4. 灵活的数据加载和增强")
    print("5. 完善的性能优化选项")
    print("\n开始使用timm，构建您的图像识别项目！")
