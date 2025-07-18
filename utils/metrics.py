#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
评估指标模块
集中管理系统评估指标
"""

import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy.spatial.distance import jensenshannon

# 导入配置
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *

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

def calculate_precision(outputs, targets, average='macro'):
    """计算精确率
    
    Args:
        outputs: 模型输出
        targets: 目标标签
        average: 平均方式
        
    Returns:
        精确率
    """
    _, predicted = torch.max(outputs.data, 1)
    return precision_score(targets.cpu().numpy(), predicted.cpu().numpy(), average=average)

def calculate_recall(outputs, targets, average='macro'):
    """计算召回率
    
    Args:
        outputs: 模型输出
        targets: 目标标签
        average: 平均方式
        
    Returns:
        召回率
    """
    _, predicted = torch.max(outputs.data, 1)
    return recall_score(targets.cpu().numpy(), predicted.cpu().numpy(), average=average)

def calculate_f1(outputs, targets, average='macro'):
    """计算F1分数
    
    Args:
        outputs: 模型输出
        targets: 目标标签
        average: 平均方式
        
    Returns:
        F1分数
    """
    _, predicted = torch.max(outputs.data, 1)
    return f1_score(targets.cpu().numpy(), predicted.cpu().numpy(), average=average)

def calculate_loss(outputs, targets, loss_fn=F.cross_entropy):
    """计算损失
    
    Args:
        outputs: 模型输出
        targets: 目标标签
        loss_fn: 损失函数
        
    Returns:
        损失值
    """
    return loss_fn(outputs, targets).item()

def calculate_js_divergence(p, q):
    """计算JS散度
    
    Args:
        p: 分布1
        q: 分布2
        
    Returns:
        JS散度
    """
    # 确保p和q是概率分布（和为1）
    p = np.array(p)
    q = np.array(q)
    
    if np.sum(p) > 0:
        p = p / np.sum(p)
    if np.sum(q) > 0:
        q = q / np.sum(q)
    
    return jensenshannon(p, q)

def calculate_label_distribution(dataset):
    """计算数据集的标签分布
    
    Args:
        dataset: 数据集
        
    Returns:
        标签分布（计数）
    """
    if hasattr(dataset, 'targets'):
        labels = dataset.targets
    elif hasattr(dataset, 'labels'):
        labels = dataset.labels
    else:
        # 尝试从数据集中提取标签
        labels = []
        for _, label in dataset:
            labels.append(label)
        labels = torch.tensor(labels)
    
    # 转换为numpy数组
    if isinstance(labels, torch.Tensor):
        labels = labels.numpy()
    
    # 计算每个类别的样本数
    unique_labels = np.unique(labels)
    distribution = np.zeros(len(unique_labels))
    
    for i, label in enumerate(unique_labels):
        distribution[i] = np.sum(labels == label)
    
    return distribution

def calculate_non_iid_degree(distributions):
    """计算非IID程度
    
    Args:
        distributions: 多个客户端的标签分布列表
        
    Returns:
        非IID程度（平均JS散度）
    """
    n = len(distributions)
    if n <= 1:
        return 0.0
    
    # 计算全局分布
    global_dist = np.sum(distributions, axis=0)
    if np.sum(global_dist) > 0:
        global_dist = global_dist / np.sum(global_dist)
    
    # 计算每个客户端分布与全局分布的JS散度
    js_divergences = []
    for dist in distributions:
        if np.sum(dist) > 0:
            dist = dist / np.sum(dist)
        js_div = calculate_js_divergence(dist, global_dist)
        js_divergences.append(js_div)
    
    # 返回平均JS散度
    return np.mean(js_divergences)

def evaluate_model(model, data_loader, device=DEVICE):
    """评估模型性能
    
    Args:
        model: 模型实例
        data_loader: 数据加载器
        device: 计算设备
        
    Returns:
        (loss, accuracy, precision, recall, f1) 元组
    """
    model.eval()
    
    total_loss = 0.0
    all_outputs = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            # 计算损失
            loss = F.cross_entropy(output, target)
            total_loss += loss.item() * data.size(0)
            
            # 收集输出和目标
            all_outputs.append(output)
            all_targets.append(target)
    
    # 合并所有批次的输出和目标
    all_outputs = torch.cat(all_outputs, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    # 计算指标
    accuracy = calculate_accuracy(all_outputs, all_targets)
    precision = calculate_precision(all_outputs, all_targets)
    recall = calculate_recall(all_outputs, all_targets)
    f1 = calculate_f1(all_outputs, all_targets)
    
    # 计算平均损失
    total_samples = len(data_loader.dataset)
    avg_loss = total_loss / total_samples
    
    return avg_loss, accuracy, precision, recall, f1

def calculate_communication_efficiency(num_rounds, num_parameters, num_clients):
    """计算通信效率
    
    Args:
        num_rounds: 通信轮数
        num_parameters: 每轮传输的参数量
        num_clients: 参与客户端数量
        
    Returns:
        通信效率指标
    """
    # 简化的通信效率计算
    total_communication = num_rounds * num_parameters * num_clients
    return 1.0 / total_communication

def calculate_resource_utilization(compute_times, compute_capabilities):
    """计算资源利用率
    
    Args:
        compute_times: 计算时间列表
        compute_capabilities: 计算能力列表
        
    Returns:
        资源利用率
    """
    # 简化的资源利用率计算
    if len(compute_times) != len(compute_capabilities) or len(compute_times) == 0:
        return 0.0
    
    utilizations = []
    for time, capability in zip(compute_times, compute_capabilities):
        utilization = time * capability
        utilizations.append(utilization)
    
    return np.mean(utilizations)

def calculate_energy_efficiency(energy_consumptions, accuracies):
    """计算能源效率
    
    Args:
        energy_consumptions: 能耗列表
        accuracies: 准确率列表
        
    Returns:
        能源效率
    """
    # 简化的能源效率计算
    if len(energy_consumptions) != len(accuracies) or len(energy_consumptions) == 0:
        return 0.0
    
    total_energy = sum(energy_consumptions)
    avg_accuracy = np.mean(accuracies)
    
    if total_energy > 0:
        return avg_accuracy / total_energy
    else:
        return 0.0
