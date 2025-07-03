#!/usr/bin/env python3
"""
多GPU分布式训练脚本 - PyTorch Image Models
支持DataParallel和DistributedDataParallel
"""

import argparse
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms
import timm
from timm.loss import LabelSmoothingCrossEntropy
from timm.utils import accuracy, AverageMeter
import logging

def setup_logging(output_dir, rank=0):
    """设置日志记录"""
    if rank == 0:
        os.makedirs(output_dir, exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(output_dir, 'training.log')),
                logging.StreamHandler()
            ]
        )

def setup_distributed(rank, world_size):
    """设置分布式训练"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup_distributed():
    """清理分布式训练"""
    dist.destroy_process_group()

def create_datasets(data_dir, img_size=224, batch_size=32, distributed=False):
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
    
    # 分布式采样器
    train_sampler = DistributedSampler(train_dataset) if distributed else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if distributed else None
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=(train_sampler is None), 
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        sampler=val_sampler,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, val_loader, train_dataset.classes, train_sampler

def train_distributed(rank, world_size, args):
    """分布式训练主函数"""
    # 设置分布式
    setup_distributed(rank, world_size)
    
    # 设置日志
    setup_logging(args.output, rank)
    
    # 创建数据集
    train_loader, val_loader, classes, train_sampler = create_datasets(
        args.data_dir, args.img_size, args.batch_size, distributed=True
    )
    
    if rank == 0:
        logging.info(f'类别: {classes}')
        logging.info(f'训练样本: {len(train_loader.dataset)}')
        logging.info(f'验证样本: {len(val_loader.dataset)}')
    
    # 创建模型
    model = timm.create_model(args.model, pretrained=True, num_classes=args.num_classes)
    model = model.to(rank)
    model = DDP(model, device_ids=[rank])
    
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
    
    for epoch in range(args.epochs):
        if train_sampler:
            train_sampler.set_epoch(epoch)
        
        start_time = time.time()
        
        # 训练
        model.train()
        train_losses = AverageMeter()
        train_top1 = AverageMeter()
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(rank, non_blocking=True), target.to(rank, non_blocking=True)
            
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
            train_losses.update(loss.item(), data.size(0))
            train_top1.update(acc1.item(), data.size(0))
            
            if rank == 0 and batch_idx % 10 == 0:
                logging.info(f'Epoch: {epoch} [{batch_idx}/{len(train_loader)}] '
                            f'Loss: {train_losses.avg:.4f} Acc@1: {train_top1.avg:.2f}%')
        
        # 验证
        model.eval()
        val_losses = AverageMeter()
        val_top1 = AverageMeter()
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(rank, non_blocking=True), target.to(rank, non_blocking=True)
                
                output = model(data)
                loss = criterion(output, target)
                
                acc1 = accuracy(output, target, topk=(1,))[0]
                val_losses.update(loss.item(), data.size(0))
                val_top1.update(acc1.item(), data.size(0))
        
        # 更新学习率
        scheduler.step()
        
        # 保存模型 (只在rank 0保存)
        if rank == 0:
            is_best = val_top1.avg > best_acc
            best_acc = max(val_top1.avg, best_acc)
            
            checkpoint = {
                'epoch': epoch + 1,
                'state_dict': model.module.state_dict(),
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
            }
            
            torch.save(checkpoint, os.path.join(args.output, 'checkpoint.pth'))
            if is_best:
                torch.save(checkpoint, os.path.join(args.output, 'best_model.pth'))
            
            epoch_time = time.time() - start_time
            logging.info(f'Epoch {epoch}: 训练损失={train_losses.avg:.4f}, 训练准确率={train_top1.avg:.2f}%, '
                        f'验证损失={val_losses.avg:.4f}, 验证准确率={val_top1.avg:.2f}%, '
                        f'最佳准确率={best_acc:.2f}%, 耗时={epoch_time:.1f}s')
    
    if rank == 0:
        logging.info(f'分布式训练完成! 最佳验证准确率: {best_acc:.2f}%')
    
    cleanup_distributed()

def main():
    parser = argparse.ArgumentParser(description='多GPU分布式训练脚本')
    parser.add_argument('--data-dir', default='./dataset', help='数据集路径')
    parser.add_argument('--model', default='efficientnet_b0', help='模型名称')
    parser.add_argument('--num-classes', type=int, default=2, help='分类数量')
    parser.add_argument('--batch-size', type=int, default=32, help='批次大小')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--img-size', type=int, default=224, help='图像尺寸')
    parser.add_argument('--output', default='./output', help='输出目录')
    parser.add_argument('--amp', action='store_true', help='启用混合精度训练')
    parser.add_argument('--label-smoothing', type=float, default=0.1, help='标签平滑')
    parser.add_argument('--distributed', action='store_true', help='启用分布式训练')
    
    args = parser.parse_args()
    
    if args.distributed and torch.cuda.device_count() > 1:
        # 多GPU分布式训练
        world_size = torch.cuda.device_count()
        mp.spawn(train_distributed, args=(world_size, args), nprocs=world_size, join=True)
    else:
        # 单GPU训练
        from gpu_train import main as single_gpu_main
        single_gpu_main()

if __name__ == '__main__':
    main()
