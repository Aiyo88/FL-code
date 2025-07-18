#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
通用工具函数模块
集中管理各个模块中重复的工具函数
"""

import os
import sys
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# 导入配置
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *

def set_seed(seed=SEED):
    """设置随机种子，确保结果可复现
    
    Args:
        seed: 随机种子值
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def create_optimizer(model, lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY):
    """创建优化器
    
    Args:
        model: 模型实例
        lr: 学习率
        momentum: 动量参数
        weight_decay: 权重衰减参数
        
    Returns:
        优化器实例
    """
    return torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay
    )

def calculate_model_size(model):
    """计算模型参数量
    
    Args:
        model: 模型实例
        
    Returns:
        模型参数量
    """
    return sum(p.numel() for p in model.parameters())

def calculate_accuracy(outputs, targets):
    """计算分类准确率
    
    Args:
        outputs: 模型输出
        targets: 目标标签
        
    Returns:
        准确率
    """
    _, predicted = torch.max(outputs.data, 1)
    correct = (predicted == targets).sum().item()
    total = targets.size(0)
    return correct / total

def calculate_communication_cost(parameters):
    """计算通信成本（参数量）
    
    Args:
        parameters: 模型参数字典
        
    Returns:
        参数总量
    """
    return sum(p.numel() for p in parameters.values())

def calculate_energy_consumption(compute_time, compute_capability):
    """计算能耗
    
    Args:
        compute_time: 计算时间
        compute_capability: 计算能力
        
    Returns:
        能耗估计值
    """
    # 简化的能耗计算模型
    return compute_time * (1.0 / compute_capability) * 0.1

def calculate_distance(loc1, loc2):
    """计算两点之间的欧氏距离
    
    Args:
        loc1: 位置1 [x, y]
        loc2: 位置2 [x, y]
        
    Returns:
        欧氏距离
    """
    return np.sqrt(np.sum(np.square(np.array(loc1) - np.array(loc2))))

def is_in_coverage(center, point, radius):
    """判断点是否在覆盖范围内
    
    Args:
        center: 中心点 [x, y]
        point: 待判断点 [x, y]
        radius: 覆盖半径
        
    Returns:
        布尔值，表示是否在覆盖范围内
    """
    return calculate_distance(center, point) <= radius

def aggregate_parameters(parameters_list, weights=None):
    """聚合多个参数字典
    
    Args:
        parameters_list: 参数字典列表
        weights: 权重列表，如果为None则使用等权重
        
    Returns:
        聚合后的参数字典
    """
    if not parameters_list:
        return None
    
    # 如果没有提供权重，使用等权重
    if weights is None:
        weights = [1.0 / len(parameters_list)] * len(parameters_list)
    
    # 归一化权重
    weights_sum = sum(weights)
    if weights_sum > 0:
        weights = [w / weights_sum for w in weights]
    
    # 初始化聚合参数字典
    aggregated_parameters = {}
    
    # 获取所有参数键
    keys = parameters_list[0].keys()
    
    # 聚合参数
    for key in keys:
        aggregated_parameters[key] = torch.zeros_like(parameters_list[0][key])
        for i, parameters in enumerate(parameters_list):
            if key in parameters:
                aggregated_parameters[key] += weights[i] * parameters[key]
    
    return aggregated_parameters

def compute_gradient(model, data_loader, loss_fn, device=DEVICE):
    """计算模型在数据集上的梯度
    
    Args:
        model: 模型实例
        data_loader: 数据加载器
        loss_fn: 损失函数
        device: 计算设备
        
    Returns:
        梯度字典
    """
    model.eval()
    
    # 初始化梯度字典
    gradient_dict = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            gradient_dict[name] = torch.zeros_like(param.data)
    
    # 计算梯度
    sample_count = 0
    for data, label in data_loader:
        data, label = data.to(device), label.to(device)
        model.zero_grad()
        pred = model(data)
        loss = loss_fn(pred, label)
        loss.backward()
        
        # 累加梯度
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                gradient_dict[name] += param.grad.data * len(data)
        
        sample_count += len(data)
        
        # 只使用一个批次计算梯度
        break
    
    # 计算平均梯度
    if sample_count > 0:
        for name in gradient_dict:
            gradient_dict[name] /= sample_count
    
    return gradient_dict

def apply_atafl_proxy(local_parameters, global_parameters, global_gradient, eta=ATAFL_ETA):
    """应用ATAFL代理函数
    
    Args:
        local_parameters: 本地模型参数
        global_parameters: 全局模型参数
        global_gradient: 全局梯度
        eta: ATAFL超参数η
        
    Returns:
        处理后的参数
    """
    if global_gradient is None:
        return local_parameters
    
    # 计算本地梯度（使用参数差异作为近似）
    local_gradient = {}
    for key in global_parameters.keys():
        if key in local_parameters and key in global_parameters:
            local_gradient[key] = local_parameters[key] - global_parameters[key]
    
    # 应用ATAFL代理函数（公式3）
    processed_parameters = {}
    for key in local_parameters.keys():
        if key in global_gradient and key in local_gradient:
            # 计算η∇F^(t-1) - ∇F_i(ω^(t-1))
            proxy_term = eta * global_gradient[key] - local_gradient[key]
            # 应用代理函数优化
            processed_parameters[key] = local_parameters[key] + proxy_term
        else:
            processed_parameters[key] = local_parameters[key]
    
    return processed_parameters

def save_model(model, path, epoch=None, optimizer=None, loss=None, accuracy=None):
    """保存模型
    
    Args:
        model: 模型实例
        path: 保存路径
        epoch: 当前轮次
        optimizer: 优化器实例
        loss: 当前损失
        accuracy: 当前准确率
        
    Returns:
        保存路径
    """
    # 确保目录存在
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # 准备保存内容
    save_dict = {
        'model_state_dict': model.state_dict(),
    }
    
    if epoch is not None:
        save_dict['epoch'] = epoch
    
    if optimizer is not None:
        save_dict['optimizer_state_dict'] = optimizer.state_dict()
    
    if loss is not None:
        save_dict['loss'] = loss
    
    if accuracy is not None:
        save_dict['accuracy'] = accuracy
    
    # 保存模型
    torch.save(save_dict, path)
    
    return path

def load_model(model, path, optimizer=None, device=DEVICE):
    """加载模型
    
    Args:
        model: 模型实例
        path: 模型路径
        optimizer: 优化器实例
        device: 计算设备
        
    Returns:
        (model, optimizer, epoch, loss, accuracy) 元组
    """
    if not os.path.exists(path):
        return model, optimizer, 0, None, None
    
    # 加载模型
    checkpoint = torch.load(path, map_location=device)
    
    # 加载模型参数
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 加载优化器参数
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # 获取其他信息
    epoch = checkpoint.get('epoch', 0)
    loss = checkpoint.get('loss', None)
    accuracy = checkpoint.get('accuracy', None)
    
    return model, optimizer, epoch, loss, accuracy

def get_time_str():
    """获取格式化的时间字符串
    
    Returns:
        格式化的时间字符串
    """
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())
