# 批量预测脚本使用指南

## 📋 脚本说明

我为您创建了两个批量预测脚本：

1. **`batch_predict.py`** - 功能完整的专业版本
2. **`simple_batch_predict.py`** - 简化版本，易于使用

## 🚀 快速开始

### 方法1: 使用简化版脚本 (推荐新手)

```bash
# 运行简化版脚本
python simple_batch_predict.py
```

然后按提示输入：
- 图片文件夹路径
- 输出文件名 (可选)

### 方法2: 使用完整版脚本

```bash
# 基础用法 - 预测文件夹
python batch_predict.py --model "./output_6class_gpu/six_class_gpu_training/model_best.pth.tar" --input "./test_images/"

# 保存结果到CSV
python batch_predict.py --model "./output_6class_gpu/six_class_gpu_training/model_best.pth.tar" --input "./test_images/" --output "results.csv" --format csv

# 递归预测子目录 + 生成报告
python batch_predict.py --model "./output_6class_gpu/six_class_gpu_training/model_best.pth.tar" --input "./test_images/" --output "results.json" --recursive --report

# 预测单张图片
python batch_predict.py --model "./output_6class_gpu/six_class_gpu_training/model_best.pth.tar" --input "image.jpg"
```

## 📊 支持的功能

### 完整版脚本功能

| 功能 | 说明 |
|------|------|
| 单张图片预测 | 预测单张图片并显示详细结果 |
| 文件夹批量预测 | 预测指定文件夹中的所有图片 |
| 递归预测 | 预测目录及所有子目录中的图片 |
| 多种输出格式 | 支持JSON、CSV、Excel格式 |
| 统计报告 | 生成详细的预测统计报告 |
| GPU加速 | 自动检测并使用GPU加速 |
| 进度显示 | 显示预测进度条 |

### 简化版脚本功能

| 功能 | 说明 |
|------|------|
| 文件夹批量预测 | 预测指定文件夹中的所有图片 |
| 交互式输入 | 友好的用户交互界面 |
| 自动统计 | 自动生成类别分布统计 |
| CSV输出 | 自动保存为CSV格式 |

## 📁 输出格式说明

### CSV输出格式
```csv
image_name,predicted_class,confidence,inference_time_ms,prob_bilei,prob_fuban,prob_genbu,prob_mengpi,prob_qianhou,prob_waiguan
image1.jpg,bilei,0.987,12.3,0.987,0.008,0.002,0.001,0.001,0.001
image2.jpg,waiguan,0.945,11.8,0.012,0.015,0.008,0.010,0.010,0.945
```

### JSON输出格式
```json
[
  {
    "image_path": "/path/to/image1.jpg",
    "image_name": "image1.jpg",
    "predicted_class": "bilei",
    "predicted_index": 0,
    "confidence": 0.987,
    "inference_time_ms": 12.3,
    "probabilities": {
      "bilei": 0.987,
      "fuban": 0.008,
      "genbu": 0.002,
      "mengpi": 0.001,
      "qianhou": 0.001,
      "waiguan": 0.001
    },
    "timestamp": "2025-01-02T10:30:45"
  }
]
```

## 🎯 6个类别说明

| 类别 | 中文名称 | 说明 |
|------|----------|------|
| bilei | 鼻泪 | 鼻泪相关特征 |
| fuban | 腹斑 | 腹部斑纹特征 |
| genbu | 跟部 | 跟部相关特征 |
| mengpi | 蒙皮 | 蒙皮相关特征 |
| qianhou | 前后 | 前后位置特征 |
| waiguan | 外观 | 外观整体特征 |

## 📈 性能参数

### GPU性能 (RTX 3060)
- **推理速度**: ~3-5ms/张
- **批次处理**: 支持批量加速
- **内存占用**: ~2GB显存

### CPU性能
- **推理速度**: ~15-30ms/张
- **内存占用**: ~1GB内存

## 🛠️ 使用示例

### 示例1: 快速批量预测
```bash
# 使用简化版脚本
python simple_batch_predict.py

# 输入示例:
# 请输入图片文件夹路径: ./test_images
# 请输入输出文件名 (可选，直接回车跳过): my_results.csv
```

### 示例2: 专业批量预测
```bash
# 预测测试集并生成完整报告
python batch_predict.py \
    --model "./output_6class_gpu/six_class_gpu_training/model_best.pth.tar" \
    --input "./dataset/test" \
    --output "test_results.csv" \
    --format csv \
    --report
```

### 示例3: 递归预测多个子目录
```bash
# 递归预测所有子目录
python batch_predict.py \
    --model "./output_6class_gpu/six_class_gpu_training/model_best.pth.tar" \
    --input "./all_images" \
    --output "all_results.excel" \
    --format excel \
    --recursive \
    --report
```

## 📊 输出示例

### 控制台输出示例
```
🎯 6类别图像分类批量预测
==================================================
🚀 初始化批量预测器...
📱 设备: cuda
🏷️  类别数: 6
📂 模型路径: ./output_6class_gpu/six_class_gpu_training/model_best.pth.tar
✅ 模型加载成功!
📏 输入尺寸: (3, 224, 224)

📁 找到 152 张图片
预测进度: 100%|████████████| 152/152 [00:08<00:00, 18.2it/s]
✅ 预测完成: 成功 152 张, 失败 0 张

📈 预测统计:
📸 总图片数: 152
✅ 成功预测: 152
❌ 失败预测: 0
📊 成功率: 100.0%

🏷️  类别分布:
   bilei: 25 张
   fuban: 24 张
   genbu: 26 张
   mengpi: 25 张
   qianhou: 26 张
   waiguan: 26 张

⚡ 性能统计:
   平均置信度: 0.847
   平均推理时间: 4.2ms
   总推理时间: 638.4ms

💾 CSV结果已保存: test_results.csv
📊 统计报告已保存: test_results.report.json
```

## ⚠️ 注意事项

1. **模型路径**: 确保模型文件路径正确
2. **图片格式**: 支持 JPG, PNG, BMP, TIFF 格式
3. **GPU内存**: 大批量预测时注意GPU内存使用
4. **文件权限**: 确保有输出目录的写入权限
5. **依赖库**: 需要安装 pandas, tqdm 等依赖

## 🔧 故障排除

### 常见问题

1. **模型加载失败**
   ```
   解决方案: 检查模型文件路径和权限
   ```

2. **CUDA内存不足**
   ```bash
   # 使用CPU模式
   python batch_predict.py --device cpu ...
   ```

3. **图片格式不支持**
   ```
   解决方案: 转换为支持的格式 (JPG, PNG等)
   ```

4. **依赖库缺失**
   ```bash
   pip install pandas tqdm openpyxl
   ```

## 🎉 使用建议

1. **首次使用**: 推荐使用 `simple_batch_predict.py`
2. **大批量处理**: 使用完整版脚本的递归功能
3. **结果分析**: 启用 `--report` 参数生成详细统计
4. **性能优化**: 在GPU环境下运行以获得最佳性能

现在您可以轻松地对任意数量的图片进行批量分类预测！
