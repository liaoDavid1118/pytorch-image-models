#!/usr/bin/env python3
"""
PyTorch Image Models (timm) 模型配置指南
"""

import timm
from pprint import pprint

# 1. 查看所有可用模型
def list_available_models():
    """列出所有可用的模型"""
    print("=== 所有可用模型 ===")
    all_models = timm.list_models()
    print(f"总共有 {len(all_models)} 个模型")
    
    # 显示前20个模型作为示例
    print("前20个模型:")
    pprint(all_models[:20])

# 2. 查看有预训练权重的模型
def list_pretrained_models():
    """列出有预训练权重的模型"""
    print("\n=== 有预训练权重的模型 ===")
    pretrained_models = timm.list_models(pretrained=True)
    print(f"有预训练权重的模型数量: {len(pretrained_models)}")
    
    # 按类型分类显示
    model_types = {}
    for model in pretrained_models[:50]:  # 只显示前50个
        model_type = model.split('_')[0]
        if model_type not in model_types:
            model_types[model_type] = []
        model_types[model_type].append(model)
    
    for model_type, models in model_types.items():
        print(f"\n{model_type}: {len(models)} 个模型")
        print(f"  示例: {models[:3]}")

# 3. 创建不同类型的模型
def create_different_models():
    """创建不同类型的模型示例"""
    print("\n=== 创建不同类型的模型 ===")
    
    # ResNet系列
    print("1. ResNet系列:")
    resnet50 = timm.create_model('resnet50', pretrained=True)
    print(f"   ResNet50: {resnet50.num_classes} 类")
    
    # EfficientNet系列
    print("2. EfficientNet系列:")
    efficientnet = timm.create_model('efficientnet_b0', pretrained=True)
    print(f"   EfficientNet-B0: {efficientnet.num_classes} 类")
    
    # Vision Transformer系列
    print("3. Vision Transformer系列:")
    vit = timm.create_model('vit_base_patch16_224', pretrained=True)
    print(f"   ViT-Base: {vit.num_classes} 类")
    
    # MobileNet系列
    print("4. MobileNet系列:")
    mobilenet = timm.create_model('mobilenetv3_large_100', pretrained=True)
    print(f"   MobileNetV3-Large: {mobilenet.num_classes} 类")

# 4. 自定义模型配置
def create_custom_models():
    """创建自定义配置的模型"""
    print("\n=== 自定义模型配置 ===")
    
    # 自定义类别数
    print("1. 自定义类别数:")
    custom_resnet = timm.create_model('resnet50', pretrained=True, num_classes=10)
    print(f"   自定义ResNet50 (10类): {custom_resnet.num_classes} 类")
    
    # 自定义输入尺寸
    print("2. 自定义输入尺寸:")
    custom_vit = timm.create_model('vit_base_patch16_224', pretrained=True, 
                                   img_size=384, num_classes=100)
    print(f"   自定义ViT (384x384, 100类): {custom_vit.num_classes} 类")
    
    # 特征提取模式
    print("3. 特征提取模式:")
    feature_extractor = timm.create_model('resnet50', pretrained=True, features_only=True)
    print(f"   特征提取器: {type(feature_extractor)}")

# 5. 模型配置信息
def show_model_config():
    """显示模型配置信息"""
    print("\n=== 模型配置信息 ===")
    
    model = timm.create_model('resnet50', pretrained=True)
    
    print("1. 预训练配置:")
    pprint(model.pretrained_cfg)
    
    print("\n2. 模型参数数量:")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   总参数: {total_params:,}")
    print(f"   可训练参数: {trainable_params:,}")

# 6. 搜索特定模型
def search_models():
    """搜索特定类型的模型"""
    print("\n=== 搜索特定模型 ===")
    
    # 搜索ResNet模型
    resnet_models = timm.list_models('resnet*', pretrained=True)
    print(f"ResNet模型 (有预训练权重): {len(resnet_models)} 个")
    print(f"示例: {resnet_models[:5]}")
    
    # 搜索EfficientNet模型
    efficientnet_models = timm.list_models('efficientnet*', pretrained=True)
    print(f"\nEfficientNet模型 (有预训练权重): {len(efficientnet_models)} 个")
    print(f"示例: {efficientnet_models[:5]}")
    
    # 搜索Vision Transformer模型
    vit_models = timm.list_models('vit*', pretrained=True)
    print(f"\nVision Transformer模型 (有预训练权重): {len(vit_models)} 个")
    print(f"示例: {vit_models[:5]}")

if __name__ == "__main__":
    # 运行所有示例
    list_available_models()
    list_pretrained_models()
    create_different_models()
    create_custom_models()
    show_model_config()
    search_models()
    
    print("\n=== 常用模型创建命令 ===")
    print("# 基础模型创建")
    print("model = timm.create_model('resnet50', pretrained=True)")
    print("model = timm.create_model('efficientnet_b0', pretrained=True)")
    print("model = timm.create_model('vit_base_patch16_224', pretrained=True)")
    
    print("\n# 自定义配置")
    print("model = timm.create_model('resnet50', pretrained=True, num_classes=10)")
    print("model = timm.create_model('vit_base_patch16_224', pretrained=True, img_size=384)")
    
    print("\n# 特征提取")
    print("model = timm.create_model('resnet50', pretrained=True, features_only=True)")
