#!/usr/bin/env python3
"""
简化版批量预测脚本
快速预测文件夹中的所有图片并生成结果报告
"""

import torch
import timm
from PIL import Image
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import time
from datetime import datetime

def load_model(model_path, device='auto'):
    """加载模型"""
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"🚀 加载模型...")
    print(f"📱 设备: {device}")
    
    # 创建模型
    model = timm.create_model('efficientnet_b0', num_classes=6)
    
    # 加载权重
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    
    # 数据预处理
    data_config = timm.data.resolve_data_config({}, model=model)
    transform = timm.data.create_transform(**data_config, is_training=False)
    
    print(f"✅ 模型加载成功!")
    return model, transform, device

def predict_folder(model, transform, device, folder_path, output_file=None):
    """预测文件夹中的所有图片"""
    
    # 类别名称
    class_names = ['bilei', 'fuban', 'genbu', 'mengpi', 'qianhou', 'waiguan']
    
    # 收集图片文件
    folder_path = Path(folder_path)
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_paths = []
    
    for ext in image_extensions:
        image_paths.extend(folder_path.glob(f"*{ext}"))
        image_paths.extend(folder_path.glob(f"*{ext.upper()}"))

    # 去除重复文件 (Windows系统不区分大小写会导致重复)
    image_paths = list(set(image_paths))

    if not image_paths:
        print(f"❌ 在 {folder_path} 中没有找到图片文件")
        return

    print(f"📁 找到 {len(image_paths)} 张图片")
    
    # 批量预测
    results = []
    class_counts = {}
    total_time = 0
    
    for image_path in tqdm(image_paths, desc="预测进度"):
        try:
            # 加载图片
            image = Image.open(image_path).convert('RGB')
            input_tensor = transform(image).unsqueeze(0).to(device)
            
            # 预测
            start_time = time.time()
            with torch.no_grad():
                outputs = model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
            
            inference_time = (time.time() - start_time) * 1000
            total_time += inference_time
            
            # 记录结果
            predicted_class = class_names[predicted.item()]
            class_counts[predicted_class] = class_counts.get(predicted_class, 0) + 1
            
            result = {
                'image_name': image_path.name,
                'predicted_class': predicted_class,
                'confidence': f"{confidence.item():.3f}",
                'inference_time_ms': f"{inference_time:.1f}"
            }
            
            # 添加各类别概率
            for i, class_name in enumerate(class_names):
                result[f'prob_{class_name}'] = f"{probabilities[0][i].item():.3f}"
            
            results.append(result)
            
        except Exception as e:
            print(f"❌ 处理失败 {image_path.name}: {e}")
            results.append({
                'image_name': image_path.name,
                'error': str(e)
            })
    
    # 显示统计结果
    print(f"\n📊 预测完成统计:")
    print(f"📸 总图片数: {len(image_paths)}")
    print(f"✅ 成功预测: {len([r for r in results if 'error' not in r])}")
    print(f"❌ 失败预测: {len([r for r in results if 'error' in r])}")
    print(f"⏱️  总耗时: {total_time:.1f}ms")
    print(f"⚡ 平均速度: {total_time/len(image_paths):.1f}ms/张")
    
    print(f"\n🏷️  类别分布:")
    for class_name, count in class_counts.items():
        percentage = count / len([r for r in results if 'error' not in r]) * 100
        print(f"   {class_name}: {count} 张 ({percentage:.1f}%)")
    
    # 保存结果
    if output_file:
        df = pd.DataFrame(results)
        
        if output_file.endswith('.csv'):
            df.to_csv(output_file, index=False, encoding='utf-8-sig')
        elif output_file.endswith('.xlsx'):
            df.to_excel(output_file, index=False)
        else:
            # 默认保存为CSV
            output_file = output_file + '.csv'
            df.to_csv(output_file, index=False, encoding='utf-8-sig')
        
        print(f"💾 结果已保存: {output_file}")
    
    return results, class_counts

def main():
    """主函数"""
    print("🎯 6类别图像分类批量预测工具")
    print("=" * 50)
    
    # 配置参数 (可以根据需要修改)
    MODEL_PATH = "./output_6class_gpu/six_class_gpu_training/model_best.pth.tar"
    INPUT_FOLDER = input("请输入图片文件夹路径: ").strip()
    OUTPUT_FILE = input("请输入输出文件名 (可选，直接回车跳过): ").strip()
    
    if not INPUT_FOLDER:
        print("❌ 请提供图片文件夹路径")
        return
    
    if not Path(INPUT_FOLDER).exists():
        print(f"❌ 文件夹不存在: {INPUT_FOLDER}")
        return
    
    if not Path(MODEL_PATH).exists():
        print(f"❌ 模型文件不存在: {MODEL_PATH}")
        print("请确保已完成模型训练")
        return
    
    # 如果没有指定输出文件，使用默认名称
    if not OUTPUT_FILE:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        OUTPUT_FILE = f"prediction_results_{timestamp}.csv"
    
    try:
        # 加载模型
        model, transform, device = load_model(MODEL_PATH)
        
        # 执行预测
        results, class_counts = predict_folder(model, transform, device, INPUT_FOLDER, OUTPUT_FILE)
        
        print(f"\n🎉 批量预测完成!")
        
    except Exception as e:
        print(f"❌ 预测过程中出现错误: {e}")

if __name__ == "__main__":
    main()
