#!/usr/bin/env python3
"""
预测结果分析脚本
分析批量预测的结果，生成详细的性能报告
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
import argparse

def analyze_predictions(csv_file, true_class=None):
    """
    分析预测结果
    
    Args:
        csv_file: CSV结果文件路径
        true_class: 真实类别名称 (如果已知)
    """
    print(f"🔍 分析预测结果: {csv_file}")
    
    # 读取CSV文件
    df = pd.read_csv(csv_file)
    
    print(f"📊 总预测数量: {len(df)}")
    
    # 基本统计
    print("\n📈 基本统计:")
    print(f"  平均置信度: {df['confidence'].mean():.4f}")
    print(f"  置信度标准差: {df['confidence'].std():.4f}")
    print(f"  最高置信度: {df['confidence'].max():.4f}")
    print(f"  最低置信度: {df['confidence'].min():.4f}")
    print(f"  平均推理时间: {df['inference_time_ms'].mean():.2f}ms")
    
    # 类别分布
    print("\n🏷️  预测类别分布:")
    class_counts = df['predicted_class'].value_counts()
    for class_name, count in class_counts.items():
        percentage = (count / len(df)) * 100
        print(f"  {class_name}: {count} 张 ({percentage:.1f}%)")
    
    # 如果提供了真实类别，计算准确率
    if true_class:
        correct_predictions = df['predicted_class'] == true_class
        accuracy = correct_predictions.sum() / len(df)
        print(f"\n🎯 {true_class} 类别准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        # 错误预测分析
        wrong_predictions = df[~correct_predictions]
        if len(wrong_predictions) > 0:
            print(f"\n❌ 错误预测分析 ({len(wrong_predictions)} 张):")
            wrong_class_counts = wrong_predictions['predicted_class'].value_counts()
            for class_name, count in wrong_class_counts.items():
                percentage = (count / len(wrong_predictions)) * 100
                print(f"  误分类为 {class_name}: {count} 张 ({percentage:.1f}%)")
            
            # 显示置信度最低的错误预测
            print(f"\n🔍 置信度最低的错误预测:")
            low_conf_wrong = wrong_predictions.nsmallest(5, 'confidence')
            for _, row in low_conf_wrong.iterrows():
                print(f"  {row['image_name']}: {row['predicted_class']} (置信度: {row['confidence']:.4f})")
    
    # 置信度分布分析
    print(f"\n📊 置信度分布:")
    confidence_ranges = [
        (0.9, 1.0, "非常高 (0.9-1.0)"),
        (0.8, 0.9, "高 (0.8-0.9)"),
        (0.7, 0.8, "中等 (0.7-0.8)"),
        (0.6, 0.7, "较低 (0.6-0.7)"),
        (0.0, 0.6, "低 (0.0-0.6)")
    ]
    
    for min_conf, max_conf, label in confidence_ranges:
        count = len(df[(df['confidence'] >= min_conf) & (df['confidence'] < max_conf)])
        percentage = (count / len(df)) * 100
        print(f"  {label}: {count} 张 ({percentage:.1f}%)")
    
    # 生成详细分析报告
    analysis_report = {
        "file_analyzed": csv_file,
        "total_predictions": len(df),
        "true_class": true_class,
        "statistics": {
            "mean_confidence": float(df['confidence'].mean()),
            "std_confidence": float(df['confidence'].std()),
            "max_confidence": float(df['confidence'].max()),
            "min_confidence": float(df['confidence'].min()),
            "mean_inference_time_ms": float(df['inference_time_ms'].mean())
        },
        "class_distribution": class_counts.to_dict(),
        "confidence_distribution": {}
    }
    
    # 添加置信度分布到报告
    for min_conf, max_conf, label in confidence_ranges:
        count = len(df[(df['confidence'] >= min_conf) & (df['confidence'] < max_conf)])
        analysis_report["confidence_distribution"][label] = {
            "count": count,
            "percentage": (count / len(df)) * 100
        }
    
    # 如果有真实类别，添加准确率信息
    if true_class:
        correct_predictions = df['predicted_class'] == true_class
        accuracy = correct_predictions.sum() / len(df)
        analysis_report["accuracy"] = {
            "overall": float(accuracy),
            "correct_predictions": int(correct_predictions.sum()),
            "wrong_predictions": int((~correct_predictions).sum())
        }
        
        # 错误预测分析
        wrong_predictions = df[~correct_predictions]
        if len(wrong_predictions) > 0:
            wrong_class_counts = wrong_predictions['predicted_class'].value_counts()
            analysis_report["error_analysis"] = {
                "misclassified_as": wrong_class_counts.to_dict(),
                "lowest_confidence_errors": []
            }
            
            # 添加置信度最低的错误预测
            low_conf_wrong = wrong_predictions.nsmallest(5, 'confidence')
            for _, row in low_conf_wrong.iterrows():
                analysis_report["error_analysis"]["lowest_confidence_errors"].append({
                    "image_name": row['image_name'],
                    "predicted_class": row['predicted_class'],
                    "confidence": float(row['confidence'])
                })
    
    # 保存分析报告
    report_file = csv_file.replace('.csv', '_analysis.json')
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(analysis_report, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 详细分析报告已保存: {report_file}")
    
    return analysis_report

def create_visualization(csv_file, true_class=None):
    """创建可视化图表"""
    print(f"\n📊 生成可视化图表...")
    
    df = pd.read_csv(csv_file)
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建子图
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'预测结果分析 - {Path(csv_file).stem}', fontsize=16, fontweight='bold')
    
    # 1. 类别分布饼图
    class_counts = df['predicted_class'].value_counts()
    axes[0, 0].pie(class_counts.values, labels=class_counts.index, autopct='%1.1f%%', startangle=90)
    axes[0, 0].set_title('预测类别分布')
    
    # 2. 置信度分布直方图
    axes[0, 1].hist(df['confidence'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 1].set_xlabel('置信度')
    axes[0, 1].set_ylabel('数量')
    axes[0, 1].set_title('置信度分布')
    axes[0, 1].axvline(df['confidence'].mean(), color='red', linestyle='--', 
                       label=f'平均值: {df["confidence"].mean():.3f}')
    axes[0, 1].legend()
    
    # 3. 推理时间分布
    axes[1, 0].hist(df['inference_time_ms'], bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
    axes[1, 0].set_xlabel('推理时间 (ms)')
    axes[1, 0].set_ylabel('数量')
    axes[1, 0].set_title('推理时间分布')
    axes[1, 0].axvline(df['inference_time_ms'].mean(), color='red', linestyle='--',
                       label=f'平均值: {df["inference_time_ms"].mean():.1f}ms')
    axes[1, 0].legend()
    
    # 4. 置信度 vs 类别箱线图
    df_melted = df.melt(id_vars=['predicted_class'], 
                        value_vars=['confidence'], 
                        var_name='metric', value_name='value')
    sns.boxplot(data=df_melted, x='predicted_class', y='value', ax=axes[1, 1])
    axes[1, 1].set_xlabel('预测类别')
    axes[1, 1].set_ylabel('置信度')
    axes[1, 1].set_title('各类别置信度分布')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    # 保存图表
    plot_file = csv_file.replace('.csv', '_analysis.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"📈 可视化图表已保存: {plot_file}")
    
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='预测结果分析工具')
    parser.add_argument('--csv', required=True, help='CSV结果文件路径')
    parser.add_argument('--true-class', help='真实类别名称')
    parser.add_argument('--plot', action='store_true', help='生成可视化图表')
    
    args = parser.parse_args()
    
    # 分析预测结果
    analysis_report = analyze_predictions(args.csv, args.true_class)
    
    # 生成可视化图表
    if args.plot:
        create_visualization(args.csv, args.true_class)
    
    print(f"\n🎉 分析完成!")

if __name__ == '__main__':
    main()
