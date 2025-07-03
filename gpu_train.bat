@echo off
echo ========================================
echo PyTorch Image Models GPU训练脚本
echo ========================================

REM 激活虚拟环境
call .venv\Scripts\activate

REM 检查GPU状态
echo 检查GPU状态...
python -c "import torch; print('PyTorch版本:', torch.__version__); print('CUDA可用:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"

echo.
echo 开始GPU训练...
echo.

REM GPU训练命令 - bilei vs waiguan 二分类
python gpu_train.py ^
    --data-dir ./dataset ^
    --model efficientnet_b0 ^
    --num-classes 2 ^
    --batch-size 32 ^
    --epochs 50 ^
    --lr 0.001 ^
    --img-size 224 ^
    --output ./output ^
    --amp ^
    --label-smoothing 0.1

echo.
echo 训练完成!
pause
