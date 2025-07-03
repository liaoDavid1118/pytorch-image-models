#!/usr/bin/env python3
"""
6分类完整训练脚本 - 自动调试版本
bilei, fuban, genbu, mengpi, qianhou, waiguan
"""

import argparse
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm
from timm.loss import LabelSmoothingCrossEntropy
from timm.utils import accuracy, AverageMeter
import logging
import json
from datetime import datetime

def setup_logging(output_dir):
    """设置日志记录"""
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return log_file

def create_datasets(data_dir, img_size=224, batch_size=32):
    """创建训练和验证数据集"""
    # 训练数据增强 - 针对6分类优化
    train_transform = transforms.Compose([
        transforms.Resize((img_size + 32, img_size + 32)),  # 稍大一些用于裁剪
        transforms.RandomCrop((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),  # 增加垂直翻转
        transforms.RandomRotation(degrees=20),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomGrayscale(p=0.1),  # 随机灰度化
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.1)  # 随机擦除
    ])
    
    # 验证数据预处理
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 创建数据集
    train_dataset = datasets.ImageFolder(
        root=os.path.join(data_dir, 'train'),
        transform=train_transform
    )
    
    val_dataset = datasets.ImageFolder(
        root=os.path.join(data_dir, 'val'),
        transform=val_transform
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True,
        drop_last=True  # 确保批次大小一致
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, val_loader, train_dataset.classes

def train_epoch(model, train_loader, criterion, optimizer, device, scaler, epoch):
    """训练一个epoch"""
    model.train()
    losses = AverageMeter()
    top1 = AverageMeter()
    
    start_time = time.time()
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        # 混合精度训练
        with torch.amp.autocast('cuda'):
            output = model(data)
            loss = criterion(output, target)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # 计算准确率
        acc1 = accuracy(output, target, topk=(1,))[0]
        losses.update(loss.item(), data.size(0))
        top1.update(acc1.item(), data.size(0))
        
        # 每20个batch打印一次
        if batch_idx % 20 == 0:
            elapsed = time.time() - start_time
            logging.info(f'Epoch: {epoch:3d} [{batch_idx:4d}/{len(train_loader):4d}] '
                        f'Loss: {losses.avg:.4f} Acc: {top1.avg:6.2f}% '
                        f'Time: {elapsed:.1f}s')
    
    return losses.avg, top1.avg

def validate(model, val_loader, criterion, device, classes):
    """验证模型"""
    model.eval()
    losses = AverageMeter()
    top1 = AverageMeter()
    
    # 用于计算每个类别的准确率
    class_correct = torch.zeros(len(classes))
    class_total = torch.zeros(len(classes))
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            
            output = model(data)
            loss = criterion(output, target)
            
            # 计算总体准确率
            acc1 = accuracy(output, target, topk=(1,))[0]
            losses.update(loss.item(), data.size(0))
            top1.update(acc1.item(), data.size(0))
            
            # 计算每个类别的准确率
            _, predicted = torch.max(output, 1)
            for i in range(target.size(0)):
                label = target[i].item()
                class_total[label] += 1
                if predicted[i] == target[i]:
                    class_correct[label] += 1
    
    # 打印每个类别的准确率
    logging.info("各类别验证准确率:")
    for i, class_name in enumerate(classes):
        if class_total[i] > 0:
            acc = 100 * class_correct[i] / class_total[i]
            logging.info(f"  {class_name}: {acc:.2f}% ({int(class_correct[i])}/{int(class_total[i])})")
    
    logging.info(f'总体验证 - Loss: {losses.avg:.4f} Acc: {top1.avg:.2f}%')
    return losses.avg, top1.avg

def save_training_info(output_dir, args, classes, best_acc, training_time):
    """保存训练信息"""
    info = {
        'model': args.model,
        'num_classes': len(classes),
        'classes': classes,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'learning_rate': args.lr,
        'best_accuracy': best_acc,
        'training_time_hours': training_time / 3600,
        'timestamp': datetime.now().isoformat()
    }
    
    with open(os.path.join(output_dir, 'training_info.json'), 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

def main():
    # 固定参数配置 - 针对6分类优化
    args = argparse.Namespace(
        data_dir='./dataset',
        model='efficientnet_b0',
        batch_size=32,  # RTX 3060 12GB 适合的批次大小
        epochs=80,      # 增加训练轮数
        lr=0.001,
        img_size=224,
        output='./output_6class',
        label_smoothing=0.1,
        weight_decay=0.01
    )
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 开始6分类训练")
    print(f"设备: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # 设置日志
    log_file = setup_logging(args.output)
    logging.info("=" * 60)
    logging.info("6分类训练开始")
    logging.info("=" * 60)
    
    # 创建数据集
    train_loader, val_loader, classes = create_datasets(
        args.data_dir, args.img_size, args.batch_size
    )
    
    logging.info(f"类别: {classes}")
    logging.info(f"训练样本: {len(train_loader.dataset)}")
    logging.info(f"验证样本: {len(val_loader.dataset)}")
    logging.info(f"批次大小: {args.batch_size}")
    logging.info(f"训练轮数: {args.epochs}")
    
    # 创建模型
    try:
        model = timm.create_model(args.model, pretrained=True, num_classes=len(classes))
        logging.info(f"成功加载预训练模型: {args.model}")
    except:
        model = timm.create_model(args.model, pretrained=False, num_classes=len(classes))
        logging.info(f"使用随机初始化模型: {args.model}")
    
    model = model.to(device)
    
    # 损失函数 - 使用标签平滑
    criterion = LabelSmoothingCrossEntropy(smoothing=args.label_smoothing)
    
    # 优化器 - AdamW with weight decay
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999)
    )
    
    # 学习率调度器 - 余弦退火 + 热身
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2, eta_min=1e-6
    )
    
    # 混合精度训练
    scaler = torch.amp.GradScaler('cuda')
    
    # 训练循环
    best_acc = 0.0
    best_epoch = 0
    training_start_time = time.time()
    
    logging.info("开始训练...")
    
    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        
        # 训练
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, scaler, epoch
        )
        
        # 验证
        val_loss, val_acc = validate(model, val_loader, criterion, device, classes)
        
        # 更新学习率
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # 保存最佳模型
        is_best = val_acc > best_acc
        if is_best:
            best_acc = val_acc
            best_epoch = epoch
            
            # 保存最佳模型
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
                'classes': classes
            }, os.path.join(args.output, 'best_model.pth'))
            
            logging.info(f"🎉 新的最佳模型! 准确率: {best_acc:.2f}%")
        
        # 保存检查点
        if epoch % 10 == 0 or is_best:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
                'classes': classes
            }, os.path.join(args.output, f'checkpoint_epoch_{epoch}.pth'))
        
        epoch_time = time.time() - epoch_start_time
        total_time = time.time() - training_start_time
        
        logging.info(f"Epoch {epoch:3d}/{args.epochs}: "
                    f"训练Loss={train_loss:.4f} 训练Acc={train_acc:.2f}% "
                    f"验证Loss={val_loss:.4f} 验证Acc={val_acc:.2f}% "
                    f"最佳Acc={best_acc:.2f}% LR={current_lr:.6f} "
                    f"耗时={epoch_time:.1f}s 总时间={total_time/60:.1f}min")
        
        # 早停检查
        if epoch - best_epoch > 20:
            logging.info(f"早停: 连续20轮无改善，最佳准确率: {best_acc:.2f}% (Epoch {best_epoch})")
            break
    
    total_training_time = time.time() - training_start_time
    
    logging.info("=" * 60)
    logging.info("训练完成!")
    logging.info(f"最佳验证准确率: {best_acc:.2f}% (Epoch {best_epoch})")
    logging.info(f"总训练时间: {total_training_time/3600:.2f} 小时")
    logging.info(f"日志文件: {log_file}")
    logging.info(f"最佳模型: {os.path.join(args.output, 'best_model.pth')}")
    logging.info("=" * 60)
    
    # 保存训练信息
    save_training_info(args.output, args, classes, best_acc, total_training_time)
    
    return best_acc

if __name__ == '__main__':
    best_accuracy = main()
    print(f"\n🎉 训练完成! 最佳准确率: {best_accuracy:.2f}%")
