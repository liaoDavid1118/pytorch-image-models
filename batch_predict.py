#!/usr/bin/env python3
"""
批量预测脚本 - 6类别图像分类
支持单张图片、文件夹批量预测、CSV导出等功能
"""

import torch
import timm
from PIL import Image
import numpy as np
from pathlib import Path
import argparse
import json
import csv
import time
from datetime import datetime
import pandas as pd
from tqdm import tqdm
import os

class BatchPredictor:
    def __init__(self, model_path, device='auto', num_classes=6):
        """
        初始化批量预测器
        
        Args:
            model_path: 模型权重文件路径
            device: 计算设备 ('auto', 'cpu', 'cuda')
            num_classes: 类别数量
        """
        # 自动选择设备
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        self.num_classes = num_classes
        
        # 6类别名称
        self.class_names = ['bilei', 'fuban', 'genbu', 'mengpi', 'qianhou', 'waiguan']
        
        print(f"🚀 初始化批量预测器...")
        print(f"📱 设备: {self.device}")
        print(f"🏷️  类别数: {num_classes}")
        print(f"📂 模型路径: {model_path}")
        
        # 创建模型
        self.model = timm.create_model('efficientnet_b0', num_classes=num_classes)
        
        # 加载权重
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
            
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.model.to(self.device)
        
        # 获取数据预处理配置
        self.data_config = timm.data.resolve_data_config({}, model=self.model)
        self.transform = timm.data.create_transform(**self.data_config, is_training=False)
        
        print(f"✅ 模型加载成功!")
        print(f"📏 输入尺寸: {self.data_config['input_size']}")

    def predict_single(self, image_path):
        """
        预测单张图片
        
        Args:
            image_path: 图片路径
            
        Returns:
            dict: 预测结果
        """
        try:
            # 加载和预处理图片
            image = Image.open(image_path).convert('RGB')
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # 推理
            start_time = time.time()
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
            inference_time = (time.time() - start_time) * 1000  # 转换为毫秒
            
            # 格式化结果
            result = {
                'image_path': str(image_path),
                'image_name': Path(image_path).name,
                'predicted_class': self.class_names[predicted.item()],
                'predicted_index': predicted.item(),
                'confidence': float(confidence.item()),
                'inference_time_ms': float(inference_time),
                'probabilities': {
                    self.class_names[i]: float(prob.item()) 
                    for i, prob in enumerate(probabilities[0])
                },
                'timestamp': datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            return {
                'image_path': str(image_path),
                'image_name': Path(image_path).name,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    def predict_folder(self, folder_path, extensions=('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
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
        
        # 收集所有图片文件
        for ext in extensions:
            image_paths.extend(folder_path.glob(f"*{ext}"))
            image_paths.extend(folder_path.glob(f"*{ext.upper()}"))

        # 去除重复文件 (Windows系统不区分大小写会导致重复)
        image_paths = list(set(image_paths))

        if not image_paths:
            print(f"⚠️  在 {folder_path} 中没有找到图片文件")
            return []

        print(f"📁 找到 {len(image_paths)} 张图片")
        
        # 批量预测
        results = []
        successful = 0
        failed = 0
        
        for image_path in tqdm(image_paths, desc="预测进度"):
            result = self.predict_single(image_path)
            results.append(result)
            
            if 'error' in result:
                failed += 1
            else:
                successful += 1
        
        print(f"✅ 预测完成: 成功 {successful} 张, 失败 {failed} 张")
        return results

    def predict_recursive(self, root_path, extensions=('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
        """
        递归预测目录及子目录中的所有图片
        
        Args:
            root_path: 根目录路径
            extensions: 支持的图片格式
            
        Returns:
            list: 预测结果列表
        """
        root_path = Path(root_path)
        image_paths = []
        
        # 递归收集所有图片文件
        for ext in extensions:
            image_paths.extend(root_path.rglob(f"*{ext}"))
            image_paths.extend(root_path.rglob(f"*{ext.upper()}"))

        # 去除重复文件 (Windows系统不区分大小写会导致重复)
        image_paths = list(set(image_paths))

        if not image_paths:
            print(f"⚠️  在 {root_path} 及其子目录中没有找到图片文件")
            return []

        print(f"📁 递归找到 {len(image_paths)} 张图片")
        
        # 批量预测
        results = []
        successful = 0
        failed = 0
        
        for image_path in tqdm(image_paths, desc="递归预测进度"):
            result = self.predict_single(image_path)
            results.append(result)
            
            if 'error' in result:
                failed += 1
            else:
                successful += 1
        
        print(f"✅ 递归预测完成: 成功 {successful} 张, 失败 {failed} 张")
        return results

    def save_results(self, results, output_path, format='json'):
        """
        保存预测结果
        
        Args:
            results: 预测结果列表
            output_path: 输出文件路径
            format: 输出格式 ('json', 'csv', 'excel')
        """
        output_path = Path(output_path)
        
        if format.lower() == 'json':
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"💾 JSON结果已保存: {output_path}")
            
        elif format.lower() == 'csv':
            # 准备CSV数据
            csv_data = []
            for result in results:
                if 'error' not in result:
                    row = {
                        'image_name': result['image_name'],
                        'image_path': result['image_path'],
                        'predicted_class': result['predicted_class'],
                        'confidence': result['confidence'],
                        'inference_time_ms': result['inference_time_ms'],
                        'timestamp': result['timestamp']
                    }
                    # 添加各类别概率
                    for class_name, prob in result['probabilities'].items():
                        row[f'prob_{class_name}'] = prob
                    csv_data.append(row)
                else:
                    csv_data.append({
                        'image_name': result['image_name'],
                        'image_path': result['image_path'],
                        'error': result['error'],
                        'timestamp': result['timestamp']
                    })
            
            # 写入CSV
            if csv_data:
                df = pd.DataFrame(csv_data)
                df.to_csv(output_path, index=False, encoding='utf-8-sig')
                print(f"💾 CSV结果已保存: {output_path}")
            
        elif format.lower() == 'excel':
            # 准备Excel数据
            excel_data = []
            for result in results:
                if 'error' not in result:
                    row = {
                        'image_name': result['image_name'],
                        'image_path': result['image_path'],
                        'predicted_class': result['predicted_class'],
                        'confidence': result['confidence'],
                        'inference_time_ms': result['inference_time_ms'],
                        'timestamp': result['timestamp']
                    }
                    # 添加各类别概率
                    for class_name, prob in result['probabilities'].items():
                        row[f'prob_{class_name}'] = prob
                    excel_data.append(row)
            
            if excel_data:
                df = pd.DataFrame(excel_data)
                df.to_excel(output_path, index=False)
                print(f"💾 Excel结果已保存: {output_path}")

    def generate_summary_report(self, results):
        """
        生成预测结果统计报告
        
        Args:
            results: 预测结果列表
            
        Returns:
            dict: 统计报告
        """
        successful_results = [r for r in results if 'error' not in r]
        failed_results = [r for r in results if 'error' in r]
        
        if not successful_results:
            return {
                'total_images': len(results),
                'successful': 0,
                'failed': len(failed_results),
                'error': '没有成功预测的图片'
            }
        
        # 统计各类别数量
        class_counts = {}
        confidences = []
        inference_times = []
        
        for result in successful_results:
            predicted_class = result['predicted_class']
            class_counts[predicted_class] = class_counts.get(predicted_class, 0) + 1
            confidences.append(result['confidence'])
            inference_times.append(result['inference_time_ms'])
        
        # 计算统计信息
        report = {
            'summary': {
                'total_images': len(results),
                'successful_predictions': len(successful_results),
                'failed_predictions': len(failed_results),
                'success_rate': len(successful_results) / len(results) * 100
            },
            'class_distribution': class_counts,
            'performance': {
                'avg_confidence': np.mean(confidences),
                'min_confidence': np.min(confidences),
                'max_confidence': np.max(confidences),
                'avg_inference_time_ms': np.mean(inference_times),
                'total_inference_time_ms': np.sum(inference_times)
            },
            'timestamp': datetime.now().isoformat()
        }
        
        return report

def main():
    parser = argparse.ArgumentParser(description="6类别图像分类批量预测脚本")
    parser.add_argument("--model", required=True, help="模型权重文件路径")
    parser.add_argument("--input", required=True, help="输入图片路径或文件夹路径")
    parser.add_argument("--output", help="输出结果文件路径")
    parser.add_argument("--format", choices=['json', 'csv', 'excel'], default='json', help="输出格式")
    parser.add_argument("--device", choices=['auto', 'cpu', 'cuda'], default='auto', help="计算设备")
    parser.add_argument("--recursive", action='store_true', help="递归处理子目录")
    parser.add_argument("--report", action='store_true', help="生成统计报告")
    
    args = parser.parse_args()
    
    print("🎯 6类别图像分类批量预测")
    print("=" * 50)
    
    # 初始化预测器
    predictor = BatchPredictor(args.model, args.device)
    
    # 执行预测
    input_path = Path(args.input)
    
    if input_path.is_file():
        # 单张图片预测
        print(f"\n🔍 预测单张图片: {input_path}")
        result = predictor.predict_single(input_path)
        
        if 'error' not in result:
            print(f"\n📊 预测结果:")
            print(f"🏷️  类别: {result['predicted_class']}")
            print(f"🎯 置信度: {result['confidence']:.3f}")
            print(f"⏱️  推理时间: {result['inference_time_ms']:.1f}ms")
            print(f"📈 详细概率:")
            for class_name, prob in result['probabilities'].items():
                print(f"   {class_name}: {prob:.3f}")
        else:
            print(f"❌ 预测失败: {result['error']}")
            
        results = [result]
        
    elif input_path.is_dir():
        # 文件夹批量预测
        if args.recursive:
            print(f"\n📁 递归预测目录: {input_path}")
            results = predictor.predict_recursive(input_path)
        else:
            print(f"\n📁 预测文件夹: {input_path}")
            results = predictor.predict_folder(input_path)
    else:
        print(f"❌ 输入路径不存在: {input_path}")
        return
    
    # 生成统计报告
    if args.report and results:
        print(f"\n📊 生成统计报告...")
        report = predictor.generate_summary_report(results)
        
        print(f"\n📈 预测统计:")
        print(f"📸 总图片数: {report['summary']['total_images']}")
        print(f"✅ 成功预测: {report['summary']['successful_predictions']}")
        print(f"❌ 失败预测: {report['summary']['failed_predictions']}")
        print(f"📊 成功率: {report['summary']['success_rate']:.1f}%")
        
        if 'class_distribution' in report:
            print(f"\n🏷️  类别分布:")
            for class_name, count in report['class_distribution'].items():
                print(f"   {class_name}: {count} 张")
        
        if 'performance' in report:
            print(f"\n⚡ 性能统计:")
            print(f"   平均置信度: {report['performance']['avg_confidence']:.3f}")
            print(f"   平均推理时间: {report['performance']['avg_inference_time_ms']:.1f}ms")
            print(f"   总推理时间: {report['performance']['total_inference_time_ms']:.1f}ms")
    
    # 保存结果
    if args.output and results:
        predictor.save_results(results, args.output, args.format)
        
        # 同时保存统计报告
        if args.report:
            report_path = Path(args.output).with_suffix('.report.json')
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            print(f"📊 统计报告已保存: {report_path}")

if __name__ == "__main__":
    main()
