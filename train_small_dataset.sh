#!/bin/bash
# 200张图片小数据集训练脚本
# 使用方法: bash train_small_dataset.sh /path/to/your/dataset 10

set -e  # 遇到错误立即退出

# 检查参数
if [ $# -lt 2 ]; then
    echo "使用方法: $0 <数据集路径> <类别数量> [可选:模型名称]"
    echo "示例: $0 /path/to/dataset 10 efficientnet_b0"
    exit 1
fi

DATASET_PATH=$1
NUM_CLASSES=$2
MODEL_NAME=${3:-efficientnet_b0}  # 默认使用efficientnet_b0

echo "🚀 开始训练小数据集模型"
echo "数据集路径: $DATASET_PATH"
echo "类别数量: $NUM_CLASSES"
echo "模型: $MODEL_NAME"
echo "================================"

# 检查数据集路径
if [ ! -d "$DATASET_PATH" ]; then
    echo "❌ 错误: 数据集路径不存在: $DATASET_PATH"
    exit 1
fi

# 检查数据集结构
if [ ! -d "$DATASET_PATH/train" ]; then
    echo "❌ 错误: 缺少训练集目录: $DATASET_PATH/train"
    exit 1
fi

if [ ! -d "$DATASET_PATH/val" ]; then
    echo "❌ 错误: 缺少验证集目录: $DATASET_PATH/val"
    exit 1
fi

# 创建输出目录
OUTPUT_DIR="./output_small_dataset_$(date +%Y%m%d_%H%M%S)"
mkdir -p $OUTPUT_DIR

echo "📁 输出目录: $OUTPUT_DIR"

# 方案1: 单阶段训练 (推荐新手)
echo "🎯 方案1: 单阶段端到端训练"
python train.py $DATASET_PATH \
    --model $MODEL_NAME \
    --pretrained \
    --num-classes $NUM_CLASSES \
    --batch-size 16 \
    --lr 0.001 \
    --epochs 150 \
    --opt adamw \
    --weight-decay 0.01 \
    --sched cosine \
    --warmup-epochs 10 \
    --aa rand-m15-mstd0.5-inc1 \
    --mixup 0.4 \
    --cutmix 1.0 \
    --reprob 0.3 \
    --drop-path 0.1 \
    --amp \
    --model-ema \
    --model-ema-decay 0.9999 \
    --patience 20 \
    --min-lr 1e-6 \
    --output $OUTPUT_DIR \
    --experiment small_dataset_single_stage \
    --log-interval 10 \
    --recovery-interval 5

echo "✅ 单阶段训练完成!"

# 验证最佳模型
echo "🔍 验证最佳模型..."
python validate.py $DATASET_PATH \
    --model $MODEL_NAME \
    --checkpoint $OUTPUT_DIR/model_best.pth.tar \
    --batch-size 32 \
    --amp \
    --results-file $OUTPUT_DIR/validation_results.csv

echo "📊 验证结果已保存到: $OUTPUT_DIR/validation_results.csv"

# 如果需要两阶段训练，取消下面的注释
: '
echo "🎯 方案2: 两阶段训练 (高级用户)"

# 阶段1: 特征提取 (冻结backbone)
echo "📚 阶段1: 特征提取训练..."
python train.py $DATASET_PATH \
    --model $MODEL_NAME \
    --pretrained \
    --num-classes $NUM_CLASSES \
    --batch-size 32 \
    --lr 0.01 \
    --epochs 50 \
    --opt sgd \
    --momentum 0.9 \
    --weight-decay 0.0001 \
    --sched step \
    --decay-epochs 20 \
    --aa rand-m7-mstd0.5 \
    --mixup 0.2 \
    --output ${OUTPUT_DIR}_stage1 \
    --experiment small_dataset_stage1

# 阶段2: 端到端微调
echo "🔧 阶段2: 端到端微调..."
python train.py $DATASET_PATH \
    --model $MODEL_NAME \
    --resume ${OUTPUT_DIR}_stage1/model_best.pth.tar \
    --num-classes $NUM_CLASSES \
    --batch-size 16 \
    --lr 0.0005 \
    --epochs 100 \
    --opt adamw \
    --weight-decay 0.01 \
    --sched cosine \
    --warmup-epochs 5 \
    --aa rand-m15-mstd0.5-inc1 \
    --mixup 0.4 \
    --cutmix 1.0 \
    --reprob 0.3 \
    --amp \
    --model-ema \
    --patience 15 \
    --output ${OUTPUT_DIR}_stage2 \
    --experiment small_dataset_stage2

echo "✅ 两阶段训练完成!"
'

echo "🎉 训练完成! 结果保存在: $OUTPUT_DIR"
echo "📈 查看训练日志: tensorboard --logdir $OUTPUT_DIR"
echo "🔍 最佳模型: $OUTPUT_DIR/model_best.pth.tar"
