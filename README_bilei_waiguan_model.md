# bilei vs waiguan 图像分类模型

## 🎯 项目概述

这是一个基于EfficientNet-B0架构的二分类模型，用于区分"bilei"和"waiguan"两个类别的图像。

### 📊 模型性能
- **验证集准确率**: 98.72%
- **测试集准确率**: 98.68%
- **Top-5准确率**: 100%
- **模型大小**: 4.01M 参数
- **推理速度**: ~10ms (CPU)

### 🏗️ 模型架构
- **基础架构**: EfficientNet-B0
- **预训练**: ImageNet
- **输入尺寸**: 224×224×3
- **输出**: 2个类别的概率分布

## 📁 文件结构

```
├── dataset/                          # 数据集
│   ├── train/                       # 训练集 (720张)
│   ├── val/                         # 验证集 (156张)
│   └── test/                        # 测试集 (152张)
├── output_bilei_waiguan/            # 训练输出
│   └── bilei_waiguan_classification/
│       ├── model_best.pth.tar      # 最佳模型权重
│       └── ...                     # 训练日志等
├── inference_bilei_waiguan.py       # 推理脚本
├── analyze_model_performance.py     # 性能分析脚本
├── convert_model_for_deployment.py  # 模型转换脚本
└── README_bilei_waiguan_model.md    # 本文件
```

## 🚀 快速开始

### 1. 环境准备

```bash
# 创建虚拟环境
python -m venv .venv

# 激活虚拟环境 (Windows)
.venv\Scripts\activate

# 安装依赖
pip install torch torchvision timm pillow numpy matplotlib seaborn scikit-learn
```

### 2. 单张图片预测

```python
from inference_bilei_waiguan import BileiWaiguanClassifier

# 初始化分类器
model_path = "./output_bilei_waiguan/bilei_waiguan_classification/model_best.pth.tar"
classifier = BileiWaiguanClassifier(model_path, device='cpu')

# 预测单张图片
result = classifier.predict_single("test_image.jpg")
print(f"预测类别: {result['predicted_class']}")
print(f"置信度: {result['confidence']:.3f}")
```

### 3. 批量预测

```bash
# 预测文件夹中的所有图片
python inference_bilei_waiguan.py --model model_best.pth.tar --input ./test_images/

# 保存结果到JSON文件
python inference_bilei_waiguan.py --model model_best.pth.tar --input ./test_images/ --output results.json
```

## 📊 模型分析

### 运行性能分析

```bash
python analyze_model_performance.py
```

这将生成：
- 混淆矩阵
- ROC曲线
- 置信度分布图
- 错误样本分析
- 详细分类报告

### 分析结果示例

```
📋 详细分类报告:
              precision    recall  f1-score   support

       bilei       0.99      0.99      0.99        76
     waiguan       0.99      0.99      0.99        76

    accuracy                           0.99       152
   macro avg       0.99      0.99      0.99       152
weighted avg       0.99      0.99      0.99       152
```

## 🔄 模型部署

### 转换为部署格式

```bash
python convert_model_for_deployment.py
```

这将创建：
- TorchScript格式 (推荐)
- ONNX格式 (跨平台)
- 优化PyTorch格式
- 部署配置文件
- 推理示例代码

### 部署示例

```python
import torch

# 加载TorchScript模型
model = torch.jit.load('deployment_package/model_torchscript.pt')
model.eval()

# 推理
with torch.no_grad():
    output = model(input_tensor)
    probabilities = torch.nn.functional.softmax(output, dim=1)
```

## 🎨 数据预处理

模型使用以下预处理流程：

```python
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
```

## 📈 训练详情

### 训练配置
- **模型**: EfficientNet-B0 (预训练)
- **优化器**: AdamW
- **学习率**: 0.01 → 2.7e-05 (余弦调度)
- **批次大小**: 16
- **训练轮次**: 30
- **数据增强**: AutoAugment + Mixup + CutMix

### 关键技术
1. **预训练权重**: 使用ImageNet预训练的EfficientNet-B0
2. **强数据增强**: 防止过拟合，提高泛化能力
3. **模型EMA**: 指数移动平均提升稳定性
4. **早停机制**: 防止过拟合
5. **余弦学习率调度**: 平滑的学习率衰减

## 🔧 故障排除

### 常见问题

1. **CUDA内存不足**
   ```bash
   # 使用CPU推理
   python inference_bilei_waiguan.py --model model.pth --input image.jpg --device cpu
   ```

2. **模型加载错误**
   ```python
   # 确保指定正确的类别数
   model = timm.create_model('efficientnet_b0', num_classes=2)
   ```

3. **图片格式不支持**
   ```python
   # 转换为RGB格式
   image = Image.open(image_path).convert('RGB')
   ```

### 性能优化建议

1. **GPU加速**: 使用CUDA设备可显著提升推理速度
2. **批量推理**: 多张图片一起处理提高吞吐量
3. **TorchScript**: 使用编译后的模型提升性能
4. **量化**: 考虑INT8量化减少模型大小

## 📝 API参考

### BileiWaiguanClassifier

```python
class BileiWaiguanClassifier:
    def __init__(self, model_path, device='cpu')
    def predict_single(self, image_path) -> dict
    def predict_batch(self, image_paths) -> list
    def predict_folder(self, folder_path) -> list
```

### 返回格式

```python
{
    'image_path': 'path/to/image.jpg',
    'predicted_class': 'bilei',  # 或 'waiguan'
    'confidence': 0.987,
    'probabilities': {
        'bilei': 0.987,
        'waiguan': 0.013
    }
}
```

## 📊 数据集要求

### 目录结构
```
dataset/
├── train/
│   ├── bilei/
│   └── waiguan/
├── val/
│   ├── bilei/
│   └── waiguan/
└── test/
    ├── bilei/
    └── waiguan/
```

### 图片要求
- **格式**: JPG, PNG, BMP
- **最小尺寸**: 32×32 (推荐 224×224 或更大)
- **颜色**: RGB (会自动转换)
- **质量**: 清晰，避免过度压缩

## 🤝 贡献指南

1. Fork 项目
2. 创建特性分支
3. 提交更改
4. 推送到分支
5. 创建 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 📞 联系方式

如有问题或建议，请通过以下方式联系：
- 创建 Issue
- 发送邮件
- 提交 Pull Request

---

**最后更新**: 2025年1月

**模型版本**: v1.0

**准确率**: 98.72% (验证集) | 98.68% (测试集)
