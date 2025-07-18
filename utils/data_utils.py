#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch.utils.data import DataLoader, TensorDataset, Subset
from torchvision import datasets, transforms
import os
import numpy as np

def load_data(dataset_name, mode='train'):
    """加载数据集
    Args:
        dataset_name: 数据集名称
        mode: 'train' 或 'test'
    Returns:
        (data, label) 元组，标准化后的数据和标签
    """
    if dataset_name.lower() == 'mnist':
        # 加载MNIST数据集
        from torchvision import datasets, transforms
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        dataset = datasets.MNIST(
            root='./data',
            train=(mode == 'train'),
            download=True,
            transform=transform
        )
        
        # 提取数据和标签
        data_list = []
        label_list = []
        
        # 直接从数据集中提取所有数据
        loader = DataLoader(dataset, batch_size=1000, shuffle=False)
        for batch_data, batch_label in loader:
            data_list.append(batch_data.numpy())
            label_list.append(batch_label.numpy())
        
        # 合并所有批次
        data = np.vstack(data_list)
        label = np.concatenate(label_list)
        
        # 确保数据格式正确
        if len(data.shape) == 4:  # MNIST格式为[N, 1, 28, 28]
            data = data.reshape(data.shape[0], -1)  # 展平为[N, 784]
        
        return data, label
    
    elif dataset_name.lower() == 'cifar10' or dataset_name.lower() == 'cifar':
        # 加载CIFAR10数据集
        from torchvision import datasets, transforms
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        dataset = datasets.CIFAR10(
            root='./data',
            train=(mode == 'train'),
            download=True,
            transform=transform
        )
        
        # 提取数据和标签
        data_list = []
        label_list = []
        
        # 直接从数据集中提取所有数据
        loader = DataLoader(dataset, batch_size=1000, shuffle=False)
        for batch_data, batch_label in loader:
            data_list.append(batch_data.numpy())
            label_list.append(batch_label.numpy())
        
        # 合并所有批次
        data = np.vstack(data_list)
        label = np.concatenate(label_list)
        
        # 确保数据格式正确
        if len(data.shape) == 4 and data.shape[1] == 3:  # CIFAR10格式为[N, 3, 32, 32]
            data = data.reshape(data.shape[0], -1)  # 展平为[N, 3072]
        
        return data, label
    
    else:
        raise ValueError(f"不支持的数据集: {dataset_name}")




def get_dataset(dataset_name, data_path='./data', batch_size=32):
    """
    加载指定名称的数据集
    
    Args:
        dataset_name: 数据集名称字符串，如'mnist'
        data_path: 数据集存储路径
        batch_size: 批次大小
        
    Returns:
        train_loader: 训练数据加载器
        test_loader: 测试数据加载器
    """
    if dataset_name.lower() == 'mnist':
        # MNIST数据集
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        train_dataset = datasets.MNIST(
            root=data_path, 
            train=True, 
            download=True,
            transform=transform
        )
        
        test_dataset = datasets.MNIST(
            root=data_path, 
            train=False, 
            download=True,
            transform=transform
        )
        
    elif dataset_name.lower() == 'cifar10' or dataset_name.lower() == 'cifar':
        # CIFAR10数据集
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        train_dataset = datasets.CIFAR10(
            root=data_path, 
            train=True, 
            download=True,
            transform=transform
        )
        
        test_dataset = datasets.CIFAR10(
            root=data_path, 
            train=False, 
            download=True,
            transform=transform
        )
    else:
        raise ValueError(f"不支持的数据集: {dataset_name}")
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False
    )
    
    return train_loader, test_loader

def get_client_dataset(args, client_id=None):
    """获取客户端数据集
    
    根据客户端ID分配数据
    
    Args:
        args: 配置参数
        client_id: 客户端ID
        
    Returns:
        client_loader: 客户端数据加载器
    """
    # 使用sampling模块中的函数进行数据划分
    from utils.sampling import mnist_iid, mnist_noniid, cifar_iid, cifar_noniid
    
    # 加载完整数据集
    if args.dataset.lower() == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        dataset = datasets.MNIST(
            root=args.data_path, 
            train=True, 
            download=True,
            transform=transform
        )
        
        # 数据划分
        if args.iid:
            user_groups = mnist_iid(dataset, args.num_clients)
        else:
            user_groups = mnist_noniid(dataset, args.num_clients)
            
    elif args.dataset.lower() == 'cifar10' or args.dataset.lower() == 'cifar':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        dataset = datasets.CIFAR10(
            root=args.data_path, 
            train=True, 
            download=True,
            transform=transform
        )
        
        # 数据划分
        if args.iid:
            user_groups = cifar_iid(dataset, args.num_clients)
        else:
            user_groups = cifar_noniid(dataset, args.num_clients)
    else:
        raise ValueError(f"不支持的数据集: {args.dataset}")
    
    # 获取客户端数据
    if client_id is not None:
        idxs = list(user_groups[client_id])
        client_dataset = Subset(dataset, idxs)
        client_loader = DataLoader(
            client_dataset, 
            batch_size=args.batch_size, 
            shuffle=True
        )
        return client_loader
    else:
        # 返回测试数据集
        if args.dataset.lower() == 'mnist':
            test_dataset = datasets.MNIST(
                root=args.data_path, 
                train=False, 
                download=True,
                transform=transform
            )
        else:
            test_dataset = datasets.CIFAR10(
                root=args.data_path, 
                train=False, 
                download=True,
                transform=transform
            )
            
        test_loader = DataLoader(
            test_dataset, 
            batch_size=args.batch_size, 
            shuffle=False
        )
        return test_loader
