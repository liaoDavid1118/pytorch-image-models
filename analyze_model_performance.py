#!/usr/bin/env python3
"""
模型性能分析脚本
分析训练好的bilei vs waiguan分类模型的详细性能
"""

import torch
import timm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
import json
from pathlib import Path
from PIL import Image
import pandas as pd

class ModelAnalyzer:
    def __init__(self, model_path, dataset_path, device='cpu'):
        """
        初始化模型分析器
        
        Args:
            model_path: 模型权重文件路径
            dataset_path: 数据集路径
            device: 计算设备
        """
        self.device = device
        self.dataset_path = Path(dataset_path)
        self.class_names = ['bilei', 'waiguan']
        
        # 加载模型
        self.model = timm.create_model('efficientnet_b0', num_classes=2)
        checkpoint = torch.load(model_path, map_location=device)
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.model.to(device)
        
        # 数据预处理
        self.data_config = timm.data.resolve_data_config({}, model=self.model)
        self.transform = timm.data.create_transform(**self.data_config, is_training=False)
        
        print(f"✅ 模型分析器初始化完成")

    def load_dataset_split(self, split='test'):
        """加载指定数据集分割"""
        split_path = self.dataset_path / split
        
        images = []
        labels = []
        image_paths = []
        
        for class_idx, class_name in enumerate(self.class_names):
            class_path = split_path / class_name
            if not class_path.exists():
                continue
                
            for img_path in class_path.glob('*'):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    images.append(img_path)
                    labels.append(class_idx)
                    image_paths.append(str(img_path))
        
        return images, labels, image_paths

    def predict_dataset(self, images, labels):
        """对数据集进行预测"""
        predictions = []
        probabilities = []
        
        print(f"🔍 正在预测 {len(images)} 张图片...")
        
        for i, img_path in enumerate(images):
            try:
                # 加载和预处理图片
                image = Image.open(img_path).convert('RGB')
                input_tensor = self.transform(image).unsqueeze(0).to(self.device)
                
                # 预测
                with torch.no_grad():
                    outputs = self.model(input_tensor)
                    probs = torch.nn.functional.softmax(outputs, dim=1)
                    pred = torch.argmax(probs, dim=1)
                    
                predictions.append(pred.item())
                probabilities.append(probs.cpu().numpy()[0])
                
                if (i + 1) % 50 == 0:
                    print(f"   已处理: {i + 1}/{len(images)}")
                    
            except Exception as e:
                print(f"❌ 处理失败 {img_path}: {e}")
                predictions.append(-1)  # 错误标记
                probabilities.append([0, 0])
        
        return np.array(predictions), np.array(probabilities)

    def generate_confusion_matrix(self, y_true, y_pred, save_path=None):
        """生成混淆矩阵"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names)
        plt.title('混淆矩阵 (Confusion Matrix)')
        plt.ylabel('真实标签 (True Label)')
        plt.xlabel('预测标签 (Predicted Label)')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 混淆矩阵已保存: {save_path}")
        
        plt.show()
        return cm

    def generate_classification_report(self, y_true, y_pred):
        """生成分类报告"""
        report = classification_report(y_true, y_pred, 
                                     target_names=self.class_names,
                                     output_dict=True)
        
        print("📋 详细分类报告:")
        print("=" * 50)
        print(classification_report(y_true, y_pred, target_names=self.class_names))
        
        return report

    def plot_roc_curve(self, y_true, y_probs, save_path=None):
        """绘制ROC曲线"""
        plt.figure(figsize=(10, 8))
        
        # 二分类ROC曲线
        fpr, tpr, _ = roc_curve(y_true, y_probs[:, 1])
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                label=f'ROC曲线 (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('假正率 (False Positive Rate)')
        plt.ylabel('真正率 (True Positive Rate)')
        plt.title('ROC曲线')
        plt.legend(loc="lower right")
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📈 ROC曲线已保存: {save_path}")
        
        plt.show()
        return roc_auc

    def analyze_confidence_distribution(self, y_true, y_probs, save_path=None):
        """分析置信度分布"""
        confidences = np.max(y_probs, axis=1)
        correct_mask = (np.argmax(y_probs, axis=1) == y_true)
        
        plt.figure(figsize=(12, 5))
        
        # 正确预测的置信度分布
        plt.subplot(1, 2, 1)
        plt.hist(confidences[correct_mask], bins=20, alpha=0.7, color='green', label='正确预测')
        plt.hist(confidences[~correct_mask], bins=20, alpha=0.7, color='red', label='错误预测')
        plt.xlabel('置信度')
        plt.ylabel('频次')
        plt.title('预测置信度分布')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 各类别的置信度分布
        plt.subplot(1, 2, 2)
        for i, class_name in enumerate(self.class_names):
            class_mask = (y_true == i)
            class_confidences = confidences[class_mask]
            plt.hist(class_confidences, bins=15, alpha=0.7, label=f'{class_name}')
        
        plt.xlabel('置信度')
        plt.ylabel('频次')
        plt.title('各类别置信度分布')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 置信度分布图已保存: {save_path}")
        
        plt.show()

    def find_misclassified_samples(self, images, y_true, y_pred, y_probs, top_k=5):
        """找出分类错误的样本"""
        misclassified_indices = np.where(y_true != y_pred)[0]
        
        if len(misclassified_indices) == 0:
            print("🎉 没有分类错误的样本！")
            return []
        
        # 按置信度排序，找出最"自信"的错误预测
        confidences = np.max(y_probs[misclassified_indices], axis=1)
        sorted_indices = misclassified_indices[np.argsort(confidences)[::-1]]
        
        print(f"❌ 发现 {len(misclassified_indices)} 个分类错误的样本")
        print(f"📋 置信度最高的 {min(top_k, len(sorted_indices))} 个错误样本:")
        
        misclassified_samples = []
        for i, idx in enumerate(sorted_indices[:top_k]):
            true_label = self.class_names[y_true[idx]]
            pred_label = self.class_names[y_pred[idx]]
            confidence = np.max(y_probs[idx])
            
            sample_info = {
                'image_path': str(images[idx]),
                'true_label': true_label,
                'predicted_label': pred_label,
                'confidence': confidence,
                'probabilities': {
                    self.class_names[j]: y_probs[idx][j] for j in range(len(self.class_names))
                }
            }
            
            misclassified_samples.append(sample_info)
            
            print(f"  {i+1}. {Path(images[idx]).name}")
            print(f"     真实: {true_label} | 预测: {pred_label} | 置信度: {confidence:.3f}")
        
        return misclassified_samples

    def comprehensive_analysis(self, split='test', output_dir='./analysis_results'):
        """执行全面的模型分析"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        print(f"🔬 开始全面分析模型性能 (数据集: {split})")
        print("=" * 60)
        
        # 1. 加载数据
        images, labels, image_paths = self.load_dataset_split(split)
        print(f"📊 数据集统计:")
        print(f"   总样本数: {len(images)}")
        for i, class_name in enumerate(self.class_names):
            count = sum(1 for label in labels if label == i)
            print(f"   {class_name}: {count} 张")
        
        # 2. 模型预测
        predictions, probabilities = self.predict_dataset(images, labels)
        
        # 3. 基本性能指标
        accuracy = np.mean(predictions == labels)
        print(f"\n🎯 整体准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        # 4. 生成混淆矩阵
        cm = self.generate_confusion_matrix(labels, predictions, 
                                          output_dir / 'confusion_matrix.png')
        
        # 5. 分类报告
        report = self.generate_classification_report(labels, predictions)
        
        # 6. ROC曲线
        roc_auc = self.plot_roc_curve(labels, probabilities,
                                    output_dir / 'roc_curve.png')
        
        # 7. 置信度分析
        self.analyze_confidence_distribution(labels, probabilities,
                                           output_dir / 'confidence_distribution.png')
        
        # 8. 错误样本分析
        misclassified = self.find_misclassified_samples(images, labels, predictions, probabilities)
        
        # 9. 保存详细结果
        results = {
            'dataset_split': split,
            'total_samples': len(images),
            'accuracy': float(accuracy),
            'roc_auc': float(roc_auc),
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'misclassified_samples': misclassified
        }
        
        with open(output_dir / 'analysis_results.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 分析结果已保存到: {output_dir}")
        print(f"📁 包含文件:")
        print(f"   - analysis_results.json (详细结果)")
        print(f"   - confusion_matrix.png (混淆矩阵)")
        print(f"   - roc_curve.png (ROC曲线)")
        print(f"   - confidence_distribution.png (置信度分布)")
        
        return results

def main():
    # 配置参数
    model_path = "./output_bilei_waiguan/bilei_waiguan_classification/model_best.pth.tar"
    dataset_path = "./dataset"
    
    # 创建分析器
    analyzer = ModelAnalyzer(model_path, dataset_path, device='cpu')
    
    # 执行全面分析
    results = analyzer.comprehensive_analysis(split='test')
    
    print("\n🎉 模型分析完成！")

if __name__ == "__main__":
    main()
