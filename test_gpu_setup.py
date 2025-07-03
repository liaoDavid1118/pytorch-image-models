#!/usr/bin/env python3
"""
GPU训练环境测试脚本
验证所有组件是否正确安装和配置
"""

import torch
import torchvision
import timm
import os
import sys

def test_pytorch_gpu():
    """测试PyTorch GPU支持"""
    print("🔍 测试PyTorch GPU支持...")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"TorchVision版本: {torchvision.__version__}")
    print(f"TIMM版本: {timm.__version__}")
    
    if torch.cuda.is_available():
        print("✅ CUDA可用")
        print(f"GPU数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            props = torch.cuda.get_device_properties(i)
            print(f"  显存: {props.total_memory / 1024**3:.1f} GB")
            print(f"  计算能力: {props.major}.{props.minor}")
    else:
        print("❌ CUDA不可用")
        return False
    
    return True

def test_model_creation():
    """测试模型创建"""
    print("\n🔍 测试模型创建...")
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 测试创建EfficientNet模型
        model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=2)
        model = model.to(device)
        print("✅ EfficientNet-B0模型创建成功")
        
        # 测试前向传播
        dummy_input = torch.randn(1, 3, 224, 224).to(device)
        with torch.no_grad():
            output = model(dummy_input)
        print(f"✅ 前向传播成功，输出形状: {output.shape}")
        
        return True
    except Exception as e:
        print(f"❌ 模型测试失败: {e}")
        return False

def test_mixed_precision():
    """测试混合精度训练"""
    print("\n🔍 测试混合精度训练...")
    try:
        if torch.cuda.is_available():
            device = torch.device('cuda')
            model = timm.create_model('efficientnet_b0', num_classes=2).to(device)
            scaler = torch.cuda.amp.GradScaler()
            
            dummy_input = torch.randn(2, 3, 224, 224).to(device)
            dummy_target = torch.randint(0, 2, (2,)).to(device)
            
            optimizer = torch.optim.Adam(model.parameters())
            criterion = torch.nn.CrossEntropyLoss()
            
            # 测试混合精度前向和反向传播
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                output = model(dummy_input)
                loss = criterion(output, dummy_target)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            print("✅ 混合精度训练测试成功")
            return True
        else:
            print("⚠️ 无GPU，跳过混合精度测试")
            return True
    except Exception as e:
        print(f"❌ 混合精度测试失败: {e}")
        return False

def test_data_loading():
    """测试数据加载"""
    print("\n🔍 测试数据加载...")
    try:
        from torchvision import transforms, datasets
        from torch.utils.data import DataLoader
        
        # 检查数据集目录
        data_dir = "./dataset"
        if not os.path.exists(data_dir):
            print(f"⚠️ 数据集目录不存在: {data_dir}")
            print("请确保数据集已准备完毕")
            return False
        
        train_dir = os.path.join(data_dir, "train")
        val_dir = os.path.join(data_dir, "val")
        
        if not os.path.exists(train_dir):
            print(f"⚠️ 训练数据目录不存在: {train_dir}")
            return False
        
        if not os.path.exists(val_dir):
            print(f"⚠️ 验证数据目录不存在: {val_dir}")
            return False
        
        # 创建数据变换
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # 创建数据集
        train_dataset = datasets.ImageFolder(train_dir, transform=transform)
        val_dataset = datasets.ImageFolder(val_dir, transform=transform)
        
        print(f"✅ 训练样本数: {len(train_dataset)}")
        print(f"✅ 验证样本数: {len(val_dataset)}")
        print(f"✅ 类别: {train_dataset.classes}")
        
        # 测试数据加载器
        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)
        
        # 测试加载一个批次
        for batch_idx, (data, target) in enumerate(train_loader):
            print(f"✅ 数据批次形状: {data.shape}, 标签形状: {target.shape}")
            break
        
        return True
    except Exception as e:
        print(f"❌ 数据加载测试失败: {e}")
        return False

def test_training_script():
    """测试训练脚本"""
    print("\n🔍 测试训练脚本...")
    try:
        # 检查训练脚本是否存在
        if not os.path.exists("gpu_train.py"):
            print("❌ gpu_train.py 不存在")
            return False
        
        print("✅ gpu_train.py 存在")
        
        # 检查批处理脚本
        if os.path.exists("gpu_train.bat"):
            print("✅ gpu_train.bat 存在")
        
        return True
    except Exception as e:
        print(f"❌ 训练脚本测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("🚀 PyTorch Image Models GPU训练环境测试")
    print("=" * 50)
    
    tests = [
        ("PyTorch GPU支持", test_pytorch_gpu),
        ("模型创建", test_model_creation),
        ("混合精度训练", test_mixed_precision),
        ("数据加载", test_data_loading),
        ("训练脚本", test_training_script),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                print(f"❌ {test_name} 测试失败")
        except Exception as e:
            print(f"❌ {test_name} 测试异常: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过！GPU训练环境配置完成！")
        print("\n🚀 可以开始训练:")
        print("   方法1: 运行 gpu_train.bat")
        print("   方法2: python gpu_train.py --amp --epochs 50")
    else:
        print("⚠️ 部分测试失败，请检查配置")
        
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
