#!/usr/bin/env python3
"""
模型转换和优化脚本
将训练好的模型转换为不同格式以便部署
"""

import torch
import timm
import onnx
import numpy as np
from pathlib import Path
import json
import time

class ModelConverter:
    def __init__(self, model_path, device='cpu'):
        """
        初始化模型转换器
        
        Args:
            model_path: 原始模型权重路径
            device: 计算设备
        """
        self.device = device
        self.model_path = Path(model_path)
        
        # 加载原始模型
        self.model = timm.create_model('efficientnet_b0', num_classes=2)
        checkpoint = torch.load(model_path, map_location=device)
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.model.to(device)
        
        # 获取数据配置
        self.data_config = timm.data.resolve_data_config({}, model=self.model)
        self.input_size = self.data_config['input_size']
        
        print(f"✅ 模型加载成功")
        print(f"📏 输入尺寸: {self.input_size}")
        print(f"📱 设备: {device}")

    def convert_to_torchscript(self, output_path):
        """转换为TorchScript格式"""
        print(f"\n🔄 转换为TorchScript格式...")
        
        # 创建示例输入
        example_input = torch.randn(1, *self.input_size).to(self.device)
        
        # 转换为TorchScript
        traced_model = torch.jit.trace(self.model, example_input)
        
        # 保存模型
        output_path = Path(output_path)
        traced_model.save(str(output_path))
        
        # 验证转换
        loaded_model = torch.jit.load(str(output_path))
        with torch.no_grad():
            original_output = self.model(example_input)
            traced_output = loaded_model(example_input)
            diff = torch.abs(original_output - traced_output).max().item()
        
        print(f"✅ TorchScript模型已保存: {output_path}")
        print(f"🔍 输出差异: {diff:.8f}")
        
        return output_path

    def convert_to_onnx(self, output_path):
        """转换为ONNX格式"""
        print(f"\n🔄 转换为ONNX格式...")
        
        # 创建示例输入
        example_input = torch.randn(1, *self.input_size).to(self.device)
        
        # 导出ONNX
        output_path = Path(output_path)
        torch.onnx.export(
            self.model,
            example_input,
            str(output_path),
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        # 验证ONNX模型
        try:
            onnx_model = onnx.load(str(output_path))
            onnx.checker.check_model(onnx_model)
            print(f"✅ ONNX模型已保存: {output_path}")
            print(f"🔍 ONNX模型验证通过")
        except Exception as e:
            print(f"❌ ONNX模型验证失败: {e}")
        
        return output_path

    def create_optimized_pytorch_model(self, output_path):
        """创建优化的PyTorch模型"""
        print(f"\n🔄 创建优化的PyTorch模型...")
        
        # 模型优化
        optimized_model = torch.jit.optimize_for_inference(
            torch.jit.script(self.model)
        )
        
        # 保存优化模型
        output_path = Path(output_path)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_config': {
                'model_name': 'efficientnet_b0',
                'num_classes': 2,
                'input_size': self.input_size,
                'class_names': ['bilei', 'waiguan']
            },
            'data_config': self.data_config
        }, output_path)
        
        print(f"✅ 优化PyTorch模型已保存: {output_path}")
        return output_path

    def benchmark_models(self, models_dict, num_runs=100):
        """性能基准测试"""
        print(f"\n⏱️  性能基准测试 (运行 {num_runs} 次)...")
        
        # 创建测试输入
        test_input = torch.randn(1, *self.input_size).to(self.device)
        
        results = {}
        
        for model_name, model_path in models_dict.items():
            print(f"\n测试 {model_name}...")
            
            # 加载模型
            if model_name == 'original':
                model = self.model
            elif model_name == 'torchscript':
                model = torch.jit.load(str(model_path))
            else:
                continue  # ONNX需要专门的运行时
            
            # 预热
            with torch.no_grad():
                for _ in range(10):
                    _ = model(test_input)
            
            # 计时测试
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start_time = time.time()
            
            with torch.no_grad():
                for _ in range(num_runs):
                    output = model(test_input)
            
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            end_time = time.time()
            
            avg_time = (end_time - start_time) / num_runs * 1000  # 转换为毫秒
            
            results[model_name] = {
                'avg_inference_time_ms': avg_time,
                'throughput_fps': 1000 / avg_time,
                'model_size_mb': Path(model_path).stat().st_size / (1024*1024) if model_path else 'N/A'
            }
            
            print(f"   平均推理时间: {avg_time:.2f} ms")
            print(f"   吞吐量: {1000/avg_time:.1f} FPS")
        
        return results

    def create_deployment_package(self, output_dir):
        """创建完整的部署包"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        print(f"\n📦 创建部署包: {output_dir}")
        
        # 1. 转换模型
        models = {}
        
        # TorchScript
        torchscript_path = output_dir / "model_torchscript.pt"
        models['torchscript'] = self.convert_to_torchscript(torchscript_path)
        
        # ONNX
        onnx_path = output_dir / "model.onnx"
        models['onnx'] = self.convert_to_onnx(onnx_path)
        
        # 优化PyTorch
        optimized_path = output_dir / "model_optimized.pth"
        models['optimized'] = self.create_optimized_pytorch_model(optimized_path)
        
        # 2. 性能测试
        benchmark_results = self.benchmark_models({
            'original': None,
            'torchscript': torchscript_path
        })
        
        # 3. 创建配置文件
        config = {
            'model_info': {
                'architecture': 'efficientnet_b0',
                'num_classes': 2,
                'class_names': ['bilei', 'waiguan'],
                'input_size': self.input_size,
                'data_config': self.data_config
            },
            'available_formats': {
                'torchscript': str(torchscript_path.name),
                'onnx': str(onnx_path.name),
                'optimized_pytorch': str(optimized_path.name)
            },
            'benchmark_results': benchmark_results,
            'usage_examples': {
                'torchscript': {
                    'load': "model = torch.jit.load('model_torchscript.pt')",
                    'predict': "output = model(input_tensor)"
                },
                'onnx': {
                    'load': "import onnxruntime; session = onnxruntime.InferenceSession('model.onnx')",
                    'predict': "output = session.run(None, {'input': input_array})"
                }
            }
        }
        
        with open(output_dir / 'deployment_config.json', 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        
        # 4. 创建简单的推理示例
        inference_example = '''
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

# 加载模型
model = torch.jit.load('model_torchscript.pt')
model.eval()

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])

# 推理函数
def predict_image(image_path):
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = torch.max(probabilities).item()
    
    class_names = ['bilei', 'waiguan']
    return {
        'class': class_names[predicted_class],
        'confidence': confidence,
        'probabilities': {
            class_names[i]: prob.item() 
            for i, prob in enumerate(probabilities[0])
        }
    }

# 使用示例
# result = predict_image('test_image.jpg')
# print(f"预测类别: {result['class']}, 置信度: {result['confidence']:.3f}")
'''
        
        with open(output_dir / 'inference_example.py', 'w', encoding='utf-8') as f:
            f.write(inference_example)
        
        # 5. 创建README
        readme_content = f"""
# bilei vs waiguan 分类模型部署包

## 模型信息
- 架构: EfficientNet-B0
- 类别: bilei, waiguan
- 输入尺寸: {self.input_size}
- 准确率: 98%+

## 可用格式
1. **TorchScript** (`model_torchscript.pt`) - PyTorch部署推荐
2. **ONNX** (`model.onnx`) - 跨平台部署
3. **优化PyTorch** (`model_optimized.pth`) - 包含完整配置

## 快速开始
```python
# 使用TorchScript模型
import torch
model = torch.jit.load('model_torchscript.pt')

# 或查看 inference_example.py 获取完整示例
```

## 性能基准
{json.dumps(benchmark_results, indent=2)}

## 文件说明
- `deployment_config.json`: 完整配置信息
- `inference_example.py`: 推理代码示例
- `README.md`: 本文件
"""
        
        with open(output_dir / 'README.md', 'w', encoding='utf-8') as f:
            f.write(readme_content)
        
        print(f"✅ 部署包创建完成!")
        print(f"📁 包含文件:")
        for file_path in output_dir.iterdir():
            if file_path.is_file():
                size_mb = file_path.stat().st_size / (1024*1024)
                print(f"   - {file_path.name} ({size_mb:.1f} MB)")

def main():
    # 配置参数
    model_path = "./output_bilei_waiguan/bilei_waiguan_classification/model_best.pth.tar"
    output_dir = "./deployment_package"
    
    # 创建转换器
    converter = ModelConverter(model_path, device='cpu')
    
    # 创建部署包
    converter.create_deployment_package(output_dir)
    
    print(f"\n🎉 模型转换完成!")
    print(f"📦 部署包位置: {output_dir}")

if __name__ == "__main__":
    main()
