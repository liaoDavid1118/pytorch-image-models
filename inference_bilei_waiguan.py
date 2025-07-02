#!/usr/bin/env python3
"""
bilei vs waiguan 分类模型推理脚本
使用训练好的EfficientNet-B0模型进行图像分类
"""

import torch
import timm
from PIL import Image
import numpy as np
from pathlib import Path
import argparse

class BileiWaiguanClassifier:
    def __init__(self, model_path, device='cpu'):
        """
        初始化分类器
        
        Args:
            model_path: 模型权重文件路径
            device: 计算设备 ('cpu' 或 'cuda')
        """
        self.device = device
        self.class_names = ['bilei', 'waiguan']  # 根据您的数据集类别
        
        # 创建模型
        self.model = timm.create_model('efficientnet_b0', num_classes=2)
        
        # 加载权重
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
            
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.model.to(device)
        
        # 获取数据预处理配置
        self.data_config = timm.data.resolve_data_config({}, model=self.model)
        self.transform = timm.data.create_transform(**self.data_config, is_training=False)
        
        print(f"✅ 模型加载成功!")
        print(f"📱 设备: {device}")
        print(f"🏷️  类别: {self.class_names}")
        print(f"📏 输入尺寸: {self.data_config['input_size']}")

    def predict_single(self, image_path):
        """
        对单张图片进行预测
        
        Args:
            image_path: 图片路径
            
        Returns:
            dict: 包含预测结果的字典
        """
        # 加载和预处理图片
        image = Image.open(image_path).convert('RGB')
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # 推理
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
        # 格式化结果
        result = {
            'image_path': str(image_path),
            'predicted_class': self.class_names[predicted.item()],
            'confidence': confidence.item(),
            'probabilities': {
                self.class_names[i]: prob.item() 
                for i, prob in enumerate(probabilities[0])
            }
        }
        
        return result

    def predict_batch(self, image_paths):
        """
        批量预测多张图片
        
        Args:
            image_paths: 图片路径列表
            
        Returns:
            list: 预测结果列表
        """
        results = []
        for image_path in image_paths:
            try:
                result = self.predict_single(image_path)
                results.append(result)
                print(f"✅ {Path(image_path).name}: {result['predicted_class']} ({result['confidence']:.3f})")
            except Exception as e:
                print(f"❌ {Path(image_path).name}: 处理失败 - {e}")
                
        return results

    def predict_folder(self, folder_path, extensions=('.jpg', '.jpeg', '.png', '.bmp')):
        """
        预测文件夹中的所有图片
        
        Args:
            folder_path: 文件夹路径
            extensions: 支持的图片格式
            
        Returns:
            list: 预测结果列表
        """
        folder_path = Path(folder_path)
        image_paths = []
        
        for ext in extensions:
            image_paths.extend(folder_path.glob(f"*{ext}"))
            image_paths.extend(folder_path.glob(f"*{ext.upper()}"))
        
        if not image_paths:
            print(f"⚠️  在 {folder_path} 中没有找到图片文件")
            return []
        
        print(f"📁 找到 {len(image_paths)} 张图片")
        return self.predict_batch(image_paths)

def main():
    parser = argparse.ArgumentParser(description="bilei vs waiguan 图像分类推理")
    parser.add_argument("--model", required=True, help="模型权重文件路径")
    parser.add_argument("--input", required=True, help="输入图片路径或文件夹路径")
    parser.add_argument("--device", default="cpu", help="计算设备 (cpu/cuda)")
    parser.add_argument("--output", help="输出结果文件路径 (可选)")
    
    args = parser.parse_args()
    
    # 初始化分类器
    classifier = BileiWaiguanClassifier(args.model, args.device)
    
    # 执行预测
    input_path = Path(args.input)
    
    if input_path.is_file():
        # 单张图片预测
        print(f"\n🔍 预测单张图片: {input_path}")
        result = classifier.predict_single(input_path)
        
        print(f"\n📊 预测结果:")
        print(f"🏷️  类别: {result['predicted_class']}")
        print(f"🎯 置信度: {result['confidence']:.3f}")
        print(f"📈 详细概率:")
        for class_name, prob in result['probabilities'].items():
            print(f"   {class_name}: {prob:.3f}")
            
        results = [result]
        
    elif input_path.is_dir():
        # 文件夹批量预测
        print(f"\n📁 预测文件夹: {input_path}")
        results = classifier.predict_folder(input_path)
        
        # 统计结果
        if results:
            bilei_count = sum(1 for r in results if r['predicted_class'] == 'bilei')
            waiguan_count = sum(1 for r in results if r['predicted_class'] == 'waiguan')
            avg_confidence = sum(r['confidence'] for r in results) / len(results)
            
            print(f"\n📊 批量预测统计:")
            print(f"📸 总图片数: {len(results)}")
            print(f"🔵 bilei: {bilei_count} 张")
            print(f"🔴 waiguan: {waiguan_count} 张")
            print(f"🎯 平均置信度: {avg_confidence:.3f}")
    else:
        print(f"❌ 输入路径不存在: {input_path}")
        return
    
    # 保存结果到文件
    if args.output and results:
        import json
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"💾 结果已保存到: {args.output}")

if __name__ == "__main__":
    # 示例用法
    print("🎯 bilei vs waiguan 分类器")
    print("=" * 50)
    
    # 如果直接运行，显示使用说明
    import sys
    if len(sys.argv) == 1:
        print("📖 使用方法:")
        print("1. 预测单张图片:")
        print("   python inference_bilei_waiguan.py --model model_best.pth.tar --input image.jpg")
        print()
        print("2. 预测文件夹:")
        print("   python inference_bilei_waiguan.py --model model_best.pth.tar --input ./test_images/")
        print()
        print("3. 保存结果:")
        print("   python inference_bilei_waiguan.py --model model_best.pth.tar --input ./images/ --output results.json")
        print()
        print("💡 提示: 模型文件路径示例:")
        print("   ./output_bilei_waiguan/bilei_waiguan_classification/model_best.pth.tar")
    else:
        main()
