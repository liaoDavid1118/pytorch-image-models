#!/usr/bin/env python3
"""
PyTorch Image Models (timm) 数据集格式指南
"""

import os

# 1. 支持的数据集格式
def supported_dataset_formats():
    """支持的数据集格式"""
    print("=== 支持的数据集格式 ===")
    
    formats = {
        "ImageFolder格式": {
            "描述": "PyTorch标准的文件夹结构格式",
            "适用场景": "本地数据集，自定义数据集",
            "示例": "dataset/train/class1/image1.jpg"
        },
        
        "ImageTar格式": {
            "描述": "压缩的tar文件格式",
            "适用场景": "大型数据集，网络传输",
            "示例": "dataset.tar"
        },
        
        "Torch数据集": {
            "描述": "torchvision内置数据集",
            "适用场景": "标准数据集如CIFAR, MNIST",
            "示例": "torch/cifar10, torch/imagenet"
        },
        
        "HuggingFace数据集": {
            "描述": "HuggingFace Datasets格式",
            "适用场景": "云端数据集，流式数据",
            "示例": "hfds/imagenet-1k"
        },
        
        "TensorFlow数据集": {
            "描述": "TensorFlow Datasets格式",
            "适用场景": "TF生态系统数据集",
            "示例": "tfds/imagenet2012"
        },
        
        "WebDataset格式": {
            "描述": "高性能的tar-based格式",
            "适用场景": "大规模分布式训练",
            "示例": "wds/imagenet-{000000..001281}.tar"
        }
    }
    
    for format_name, info in formats.items():
        print(f"\n{format_name}:")
        for key, value in info.items():
            print(f"  {key}: {value}")

# 2. ImageFolder格式详解
def imagefolder_format():
    """ImageFolder格式详解"""
    print("\n=== ImageFolder格式详解 ===")
    
    structure = '''
数据集目录结构:
dataset/
├── train/
│   ├── class1/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   ├── class2/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   └── ...
├── val/
│   ├── class1/
│   │   ├── image1.jpg
│   │   └── ...
│   ├── class2/
│   │   ├── image1.jpg
│   │   └── ...
│   └── ...
└── test/ (可选)
    ├── class1/
    └── ...

支持的图像格式:
- JPEG (.jpg, .jpeg)
- PNG (.png)
- BMP (.bmp)
- TIFF (.tiff, .tif)
- WebP (.webp)
'''
    
    print(structure)
    
    # 创建示例数据集结构的脚本
    script = '''
# 创建ImageFolder格式数据集的Python脚本
import os
import shutil
from pathlib import Path

def create_imagefolder_dataset(source_dir, target_dir, class_mapping):
    """
    创建ImageFolder格式的数据集
    
    Args:
        source_dir: 源图像目录
        target_dir: 目标数据集目录
        class_mapping: 类别映射字典 {image_name: class_name}
    """
    
    # 创建目录结构
    train_dir = Path(target_dir) / "train"
    val_dir = Path(target_dir) / "val"
    
    for class_name in set(class_mapping.values()):
        (train_dir / class_name).mkdir(parents=True, exist_ok=True)
        (val_dir / class_name).mkdir(parents=True, exist_ok=True)
    
    # 复制图像到对应类别目录
    for image_name, class_name in class_mapping.items():
        source_path = Path(source_dir) / image_name
        target_path = train_dir / class_name / image_name
        shutil.copy2(source_path, target_path)

# 使用示例
class_mapping = {
    "dog1.jpg": "dog",
    "dog2.jpg": "dog", 
    "cat1.jpg": "cat",
    "cat2.jpg": "cat"
}

create_imagefolder_dataset("/source/images", "/dataset", class_mapping)
'''
    
    print("\n创建ImageFolder数据集的脚本:")
    print(script)

# 3. 数据集配置示例
def dataset_configuration_examples():
    """数据集配置示例"""
    print("\n=== 数据集配置示例 ===")
    
    examples = {
        "本地ImageFolder数据集": {
            "命令": "python train.py /path/to/dataset --train-split train --val-split val",
            "说明": "使用本地ImageFolder格式数据集"
        },
        
        "CIFAR-10数据集": {
            "命令": "python train.py . --dataset torch/cifar10 --dataset-download",
            "说明": "自动下载并使用CIFAR-10数据集"
        },
        
        "ImageNet数据集": {
            "命令": "python train.py /imagenet --dataset '' --train-split train --val-split val",
            "说明": "使用ImageNet数据集"
        },
        
        "HuggingFace数据集": {
            "命令": "python train.py . --dataset hfds/imagenet-1k --dataset-download",
            "说明": "使用HuggingFace的ImageNet数据集"
        },
        
        "自定义类别映射": {
            "命令": "python train.py /dataset --class-map class_mapping.txt",
            "说明": "使用自定义类别映射文件"
        },
        
        "WebDataset格式": {
            "命令": "python train.py . --dataset wds/imagenet-{000000..001281}.tar",
            "说明": "使用WebDataset格式的大规模数据集"
        }
    }
    
    for name, info in examples.items():
        print(f"\n{name}:")
        print(f"  命令: {info['命令']}")
        print(f"  说明: {info['说明']}")

# 4. 类别映射文件格式
def class_mapping_format():
    """类别映射文件格式"""
    print("\n=== 类别映射文件格式 ===")
    
    mapping_example = '''
类别映射文件 (class_mapping.txt):
格式: class_name class_index

示例内容:
airplane 0
automobile 1
bird 2
cat 3
deer 4
dog 5
frog 6
horse 7
ship 8
truck 9

或者JSON格式 (class_mapping.json):
{
    "airplane": 0,
    "automobile": 1,
    "bird": 2,
    "cat": 3,
    "deer": 4,
    "dog": 5,
    "frog": 6,
    "horse": 7,
    "ship": 8,
    "truck": 9
}
'''
    
    print(mapping_example)

# 5. 数据预处理要求
def data_preprocessing_requirements():
    """数据预处理要求"""
    print("\n=== 数据预处理要求 ===")
    
    requirements = {
        "图像格式": [
            "支持RGB和灰度图像",
            "推荐使用RGB格式 (3通道)",
            "图像尺寸可以不一致，会自动resize"
        ],
        
        "图像质量": [
            "最小分辨率建议32x32像素",
            "避免过度压缩的JPEG图像",
            "确保图像清晰度足够"
        ],
        
        "数据组织": [
            "每个类别至少包含10张图像",
            "训练集和验证集比例建议8:2或9:1",
            "类别分布尽量均衡"
        ],
        
        "文件命名": [
            "避免特殊字符和空格",
            "使用英文字母、数字和下划线",
            "文件扩展名小写"
        ]
    }
    
    for category, items in requirements.items():
        print(f"\n{category}:")
        for item in items:
            print(f"  • {item}")

# 6. 数据增强配置
def data_augmentation_config():
    """数据增强配置"""
    print("\n=== 数据增强配置 ===")
    
    augmentation_options = {
        "基础增强": [
            "--hflip 0.5  # 水平翻转概率",
            "--vflip 0.0  # 垂直翻转概率", 
            "--color-jitter 0.4  # 颜色抖动强度",
            "--scale 0.08 1.0  # 随机缩放范围",
            "--ratio 0.75 1.33  # 宽高比范围"
        ],
        
        "高级增强": [
            "--aa rand-m9-mstd0.5-inc1  # AutoAugment策略",
            "--mixup 0.2  # Mixup alpha值",
            "--cutmix 1.0  # CutMix alpha值",
            "--reprob 0.25  # Random Erasing概率",
            "--remode pixel  # Random Erasing模式"
        ],
        
        "归一化": [
            "--mean 0.485 0.456 0.406  # ImageNet均值",
            "--std 0.229 0.224 0.225   # ImageNet标准差",
            "--interpolation bilinear   # 插值方法"
        ]
    }
    
    for category, options in augmentation_options.items():
        print(f"\n{category}:")
        for option in options:
            print(f"  {option}")

# 7. 数据集验证脚本
def dataset_validation_script():
    """数据集验证脚本"""
    print("\n=== 数据集验证脚本 ===")
    
    script = '''
# validate_dataset.py - 验证数据集格式和质量
import os
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

def validate_imagefolder_dataset(dataset_path):
    """验证ImageFolder格式数据集"""
    
    dataset_path = Path(dataset_path)
    
    # 检查目录结构
    train_dir = dataset_path / "train"
    val_dir = dataset_path / "val"
    
    if not train_dir.exists():
        print("错误: 缺少train目录")
        return False
    
    if not val_dir.exists():
        print("警告: 缺少val目录")
    
    # 统计类别和图像数量
    train_classes = [d.name for d in train_dir.iterdir() if d.is_dir()]
    print(f"训练集类别数: {len(train_classes)}")
    print(f"类别列表: {train_classes}")
    
    total_train_images = 0
    class_counts = {}
    
    for class_dir in train_dir.iterdir():
        if class_dir.is_dir():
            images = list(class_dir.glob("*"))
            image_count = len([img for img in images if img.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']])
            class_counts[class_dir.name] = image_count
            total_train_images += image_count
    
    print(f"总训练图像数: {total_train_images}")
    print("各类别图像数:")
    for class_name, count in class_counts.items():
        print(f"  {class_name}: {count}")
    
    # 检查图像质量
    print("\\n检查图像质量...")
    sample_images = []
    for class_dir in list(train_dir.iterdir())[:3]:  # 检查前3个类别
        if class_dir.is_dir():
            images = list(class_dir.glob("*.jpg"))[:2]  # 每个类别检查2张图
            sample_images.extend(images)
    
    for img_path in sample_images:
        try:
            with Image.open(img_path) as img:
                print(f"  {img_path.name}: {img.size}, {img.mode}")
        except Exception as e:
            print(f"  错误: 无法打开 {img_path.name}: {e}")
    
    return True

# 使用示例
if __name__ == "__main__":
    validate_imagefolder_dataset("/path/to/your/dataset")
'''
    
    print(script)

if __name__ == "__main__":
    supported_dataset_formats()
    imagefolder_format()
    dataset_configuration_examples()
    class_mapping_format()
    data_preprocessing_requirements()
    data_augmentation_config()
    dataset_validation_script()
    
    print("\n=== 数据集准备最佳实践 ===")
    print("1. 确保数据集目录结构正确")
    print("2. 验证所有图像都能正常打开")
    print("3. 检查类别分布是否均衡")
    print("4. 准备适当的验证集")
    print("5. 考虑数据增强策略")
    print("6. 使用合适的图像分辨率")
