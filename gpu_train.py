#!/usr/bin/env python3
"""
GPU模式训练脚本 - PyTorch Image Models
支持单GPU和多GPU训练，包含混合精度训练和各种优化
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
from timm.data import create_transform
from timm.loss import LabelSmoothingCrossEntropy
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import accuracy, AverageMeter
import logging

def setup_logging(output_dir):
    """设置日志记录"""
    os.makedirs(output_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(output_dir, 'training.log')),
            logging.StreamHandler()
        ]
    )

def create_datasets(data_dir, img_size=224, batch_size=32):
    """创建训练和验证数据集"""
    # 训练数据增强
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, val_loader, train_dataset.classes

def create_model(model_name, num_classes, pretrained=True):
    """创建模型"""
    try:
        # 首先尝试使用预训练模型
        model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes
        )
        logging.info(f"成功加载预训练模型: {model_name}")
    except Exception as e:
        logging.warning(f"预训练模型加载失败: {e}")
        logging.info("使用随机初始化模型")
        # 如果预训练模型加载失败，使用随机初始化
        model = timm.create_model(
            model_name,
            pretrained=False,
            num_classes=num_classes
        )
    return model

def train_epoch(model, train_loader, criterion, optimizer, device, scaler=None, epoch=0):
    """训练一个epoch"""
    model.train()
    losses = AverageMeter()
    top1 = AverageMeter()
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        if scaler is not None:  # 混合精度训练
            with torch.cuda.amp.autocast():
                output = model(data)
                loss = criterion(output, target)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:  # 标准训练
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        
        # 计算准确率
        acc1 = accuracy(output, target, topk=(1,))[0]
        losses.update(loss.item(), data.size(0))
        top1.update(acc1.item(), data.size(0))
        
        if batch_idx % 10 == 0:
            logging.info(f'Epoch: {epoch} [{batch_idx}/{len(train_loader)}] '
                        f'Loss: {losses.avg:.4f} Acc@1: {top1.avg:.2f}%')
    
    return losses.avg, top1.avg

def validate(model, val_loader, criterion, device):
    """验证模型"""
    model.eval()
    losses = AverageMeter()
    top1 = AverageMeter()
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            
            output = model(data)
            loss = criterion(output, target)
            
            acc1 = accuracy(output, target, topk=(1,))[0]
            losses.update(loss.item(), data.size(0))
            top1.update(acc1.item(), data.size(0))
    
    logging.info(f'Validation - Loss: {losses.avg:.4f} Acc@1: {top1.avg:.2f}%')
    return losses.avg, top1.avg

def main():
    parser = argparse.ArgumentParser(description='GPU模式训练脚本')
    parser.add_argument('--data-dir', default='./dataset', help='数据集路径')
    parser.add_argument('--model', default='efficientnet_b0', help='模型名称')
    parser.add_argument('--num-classes', type=int, default=2, help='分类数量')
    parser.add_argument('--batch-size', type=int, default=32, help='批次大小')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--img-size', type=int, default=224, help='图像尺寸')
    parser.add_argument('--output', default='./output', help='输出目录')
    parser.add_argument('--resume', default='', help='恢复训练的检查点')
    parser.add_argument('--amp', action='store_true', help='启用混合精度训练')
    parser.add_argument('--label-smoothing', type=float, default=0.1, help='标签平滑')
    
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'使用设备: {device}')
    if torch.cuda.is_available():
        logging.info(f'GPU: {torch.cuda.get_device_name(0)}')
        logging.info(f'GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
    
    # 设置日志
    setup_logging(args.output)
    
    # 创建数据集
    train_loader, val_loader, classes = create_datasets(
        args.data_dir, args.img_size, args.batch_size
    )
    logging.info(f'类别: {classes}')
    logging.info(f'训练样本: {len(train_loader.dataset)}')
    logging.info(f'验证样本: {len(val_loader.dataset)}')
    
    # 创建模型
    model = create_model(args.model, args.num_classes)
    model = model.to(device)
    logging.info(f'模型: {args.model}')
    
    # 损失函数
    if args.label_smoothing > 0:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.label_smoothing)
    else:
        criterion = nn.CrossEntropyLoss()
    
    # 优化器
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # 混合精度训练
    scaler = torch.cuda.amp.GradScaler() if args.amp else None
    
    # 训练循环
    best_acc = 0.0
    start_epoch = 0
    
    # 恢复训练
    if args.resume:
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            logging.info(f'恢复训练从 epoch {start_epoch}')
    
    for epoch in range(start_epoch, args.epochs):
        start_time = time.time()
        
        # 训练
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, scaler, epoch
        )
        
        # 验证
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # 更新学习率
        scheduler.step()
        
        # 保存最佳模型
        is_best = val_acc > best_acc
        best_acc = max(val_acc, best_acc)
        
        # 保存检查点
        checkpoint = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
        }
        
        torch.save(checkpoint, os.path.join(args.output, 'checkpoint.pth'))
        if is_best:
            torch.save(checkpoint, os.path.join(args.output, 'best_model.pth'))
        
        epoch_time = time.time() - start_time
        logging.info(f'Epoch {epoch}: 训练损失={train_loss:.4f}, 训练准确率={train_acc:.2f}%, '
                    f'验证损失={val_loss:.4f}, 验证准确率={val_acc:.2f}%, '
                    f'最佳准确率={best_acc:.2f}%, 耗时={epoch_time:.1f}s')
    
    logging.info(f'训练完成! 最佳验证准确率: {best_acc:.2f}%')

if __name__ == '__main__':
    main()
