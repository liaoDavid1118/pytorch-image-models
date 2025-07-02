#!/usr/bin/env python3
"""
PyTorch Image Models (timm) 验证指南
"""

# 1. 基础验证命令
def basic_validation_commands():
    """基础验证命令"""
    print("=== 基础验证命令 ===")
    
    commands = [
        # 验证预训练模型
        "python validate.py /path/to/imagenet --model resnet50 --pretrained",
        
        # 验证自定义检查点
        "python validate.py /path/to/dataset --model resnet50 --checkpoint /path/to/checkpoint.pth",
        
        # 指定批次大小和工作进程
        "python validate.py /path/to/dataset --model resnet50 --pretrained -b 256 -j 8",
        
        # 验证特定输入尺寸
        "python validate.py /path/to/dataset --model resnet50 --pretrained --img-size 384",
        
        # 使用测试时增强
        "python validate.py /path/to/dataset --model resnet50 --pretrained --use-ema --test-pool",
    ]
    
    for i, cmd in enumerate(commands, 1):
        print(f"{i}. {cmd}")

# 2. 验证参数详解
def validation_parameters():
    """验证参数详解"""
    print("\n=== 验证参数详解 ===")
    
    params = {
        "数据参数": {
            "--data-dir": "数据集根目录",
            "--dataset": "数据集类型",
            "--split": "验证集分割 (默认: validation)",
            "--class-map": "类别映射文件",
            "--num-samples": "手动指定样本数量",
        },
        
        "模型参数": {
            "--model": "模型架构名称",
            "--pretrained": "使用预训练权重",
            "--checkpoint": "检查点文件路径",
            "--num-classes": "分类数量",
            "--img-size": "输入图像尺寸",
            "--input-size": "完整输入尺寸 (C H W)",
        },
        
        "验证设置": {
            "--batch-size": "批次大小",
            "--workers": "数据加载工作进程数",
            "--amp": "使用混合精度",
            "--channels-last": "使用通道最后内存布局",
            "--device": "计算设备 (cuda/cpu)",
        },
        
        "测试时增强": {
            "--test-pool": "测试时池化",
            "--use-ema": "使用指数移动平均权重",
            "--crop-pct": "中心裁剪比例",
            "--interpolation": "插值方法",
            "--mean": "像素均值",
            "--std": "像素标准差",
        }
    }
    
    for category, param_dict in params.items():
        print(f"\n{category}:")
        for param, desc in param_dict.items():
            print(f"  {param:<20}: {desc}")

# 3. 不同验证场景
def validation_scenarios():
    """不同验证场景"""
    print("\n=== 不同验证场景 ===")
    
    scenarios = {
        "ImageNet验证": [
            "python validate.py /imagenet",
            "--model resnet50",
            "--pretrained",
            "--batch-size 256",
            "--workers 8",
            "--amp"
        ],
        
        "自定义数据集验证": [
            "python validate.py /custom_dataset",
            "--model resnet50",
            "--checkpoint ./output/model_best.pth.tar",
            "--num-classes 10",
            "--batch-size 128"
        ],
        
        "多尺度验证": [
            "python validate.py /dataset",
            "--model efficientnet_b0",
            "--pretrained",
            "--img-size 224 256 288 320",
            "--crop-pct 0.875 0.9 0.95 1.0"
        ],
        
        "Vision Transformer验证": [
            "python validate.py /imagenet",
            "--model vit_base_patch16_224",
            "--pretrained",
            "--batch-size 64",
            "--img-size 224",
            "--crop-pct 0.9"
        ],
        
        "测试时增强验证": [
            "python validate.py /dataset",
            "--model resnet50",
            "--pretrained",
            "--test-pool",
            "--use-ema",
            "--crop-pct 0.95"
        ]
    }
    
    for scenario, commands in scenarios.items():
        print(f"\n{scenario}:")
        command = " \\\n    ".join(commands)
        print(f"  {command}")

# 4. 批量验证脚本
def batch_validation_script():
    """批量验证脚本示例"""
    print("\n=== 批量验证脚本 ===")
    
    script = '''#!/bin/bash
# batch_validate.sh - 批量验证多个模型

DATASET_PATH="/path/to/imagenet"
CHECKPOINT_DIR="./output"

# 定义要验证的模型列表
MODELS=(
    "resnet50"
    "resnet101"
    "efficientnet_b0"
    "efficientnet_b3"
    "vit_base_patch16_224"
)

# 验证每个模型
for model in "${MODELS[@]}"; do
    echo "验证模型: $model"
    
    # 检查是否有自定义检查点
    checkpoint_file="$CHECKPOINT_DIR/${model}_best.pth.tar"
    
    if [ -f "$checkpoint_file" ]; then
        # 使用自定义检查点
        python validate.py $DATASET_PATH \\
            --model $model \\
            --checkpoint $checkpoint_file \\
            --batch-size 256 \\
            --workers 8 \\
            --amp \\
            --results-file "results_${model}_custom.csv"
    else
        # 使用预训练权重
        python validate.py $DATASET_PATH \\
            --model $model \\
            --pretrained \\
            --batch-size 256 \\
            --workers 8 \\
            --amp \\
            --results-file "results_${model}_pretrained.csv"
    fi
    
    echo "完成验证: $model"
    echo "------------------------"
done

echo "所有模型验证完成!"
'''
    
    print(script)

# 5. 验证结果分析
def validation_results_analysis():
    """验证结果分析"""
    print("\n=== 验证结果分析 ===")
    
    analysis_script = '''
# Python脚本: analyze_results.py
import pandas as pd
import matplotlib.pyplot as plt

def analyze_validation_results():
    """分析验证结果"""
    
    # 读取结果文件
    results = pd.read_csv('validation_results.csv')
    
    # 显示基本统计
    print("=== 验证结果统计 ===")
    print(f"Top-1 准确率: {results['top1'].mean():.3f} ± {results['top1'].std():.3f}")
    print(f"Top-5 准确率: {results['top5'].mean():.3f} ± {results['top5'].std():.3f}")
    print(f"推理时间: {results['infer_time'].mean():.3f}ms ± {results['infer_time'].std():.3f}ms")
    
    # 绘制结果图表
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Top-1准确率
    axes[0].bar(results['model'], results['top1'])
    axes[0].set_title('Top-1 Accuracy')
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].tick_params(axis='x', rotation=45)
    
    # Top-5准确率
    axes[1].bar(results['model'], results['top5'])
    axes[1].set_title('Top-5 Accuracy')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].tick_params(axis='x', rotation=45)
    
    # 推理时间
    axes[2].bar(results['model'], results['infer_time'])
    axes[2].set_title('Inference Time')
    axes[2].set_ylabel('Time (ms)')
    axes[2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('validation_results.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    analyze_validation_results()
'''
    
    print(analysis_script)

# 6. 验证输出格式
def validation_output_format():
    """验证输出格式说明"""
    print("\n=== 验证输出格式 ===")
    
    output_example = '''
验证输出示例:
Test: [   10/  196]  Time: 0.187 (0.190)  Loss: 0.6234 (0.6891)  Acc@1: 81.250 (79.128)  Acc@5: 95.312 (94.551)
Test: [   20/  196]  Time: 0.185 (0.188)  Loss: 0.5123 (0.6234)  Acc@1: 83.594 (80.234)  Acc@5: 96.094 (94.789)
...
 * Acc@1 80.858 Acc@5 95.434

最终结果:
- Top-1 Accuracy: 80.858%
- Top-5 Accuracy: 95.434%
- 平均推理时间: 0.188s per batch
- 总验证时间: 36.8s

结果文件 (CSV格式):
model,top1,top5,loss,infer_time,params
resnet50,80.858,95.434,0.6234,0.188,25557032
'''
    
    print(output_example)

if __name__ == "__main__":
    basic_validation_commands()
    validation_parameters()
    validation_scenarios()
    batch_validation_script()
    validation_results_analysis()
    validation_output_format()
    
    print("\n=== 验证最佳实践 ===")
    print("1. 使用足够大的批次大小以提高GPU利用率")
    print("2. 启用混合精度 (--amp) 以加速验证")
    print("3. 使用多个工作进程 (-j 8) 以加速数据加载")
    print("4. 保存验证结果到CSV文件以便后续分析")
    print("5. 对于生产环境，考虑使用测试时增强提高准确率")
