#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
客户端模块
包含ClientBase基类、Client类和EdgeNode类，以及ClientsGroup管理类
"""

import os
import sys
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Dataset
from tqdm import tqdm
import copy

# 导入配置和工具函数
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import *
from utils.common_utils import set_seed, calculate_distance, aggregate_parameters, apply_atafl_proxy
from utils.data_utils import get_dataset

class DatasetSplit(Dataset):
    """一个抽象的数据集类，封装了Pytorch的Dataset类。
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        # 使用 clone().detach() 来避免警告，并确保返回的张量与计算图分离
        return image.clone().detach(), torch.tensor(label)


class ClientBase:
    """客户端基类 - 定义所有客户端共有的属性和方法"""
    
    def __init__(self, trainDataSet, device, client_id=None, is_edge=False):
        """初始化客户端基类
        
        Args:
            trainDataSet: 训练数据集
            device: 计算设备
            client_id: 客户端ID（可选）
            is_edge: 是否为边缘节点
        """
        # 数据和设备属性
        self.train_ds = trainDataSet
        self.device = device
        self.trainloader, self.validloader, self.testloader = None, None, None
        self.client_id = client_id if client_id else f"client_{id(self)}"
        self.is_edge = is_edge
        self.disable_progress_bar = DISABLE_PROGRESS_BAR
        
        # 资源和状态属性
        self.local_data_size = len(trainDataSet) if trainDataSet else 0
        self.compute_capability = MIN_COMPUTE_CAPABILITY if not is_edge else MAX_COMPUTE_CAPABILITY
        self.energy_level = MIN_ENERGY if not is_edge else MAX_ENERGY
        self.comm_rate = MIN_COMM_RATE if not is_edge else MAX_COMM_RATE
        self.location = [random.uniform(0, 1000), random.uniform(0, 1000)]  # 随机位置
        self.available = True
        
        # 训练统计
        self.training_stats = {
            'training_time': [],
            'energy_consumption': [],
            'accuracy': [],
            'loss': []
        }
        
        # 参数缓存
        self.parameters_cache = {}
        self.gradient_cache = {}
        # 显式声明env属性，避免类型检查器报错
        self.env = None
    
    def update_state(self, **kwargs):
        """更新客户端状态
        
        Args:
            **kwargs: 要更新的状态属性
            
        Returns:
            当前客户端状态字典
        """
        # 更新所有提供的属性
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        
        # 更新本地模型参数 (ω_i^t) 和梯度 (∇r_i(ω_i^t))
        if 'local_parameters' in kwargs:
            self.parameters_cache = kwargs['local_parameters']
        if 'local_gradient' in kwargs:
            self.gradient_cache = kwargs['local_gradient']
        
        # 更新剩余能量 E_i^{max}(t) = E_i^{max}(t-1) - E_i^u(t-1)
        if 'energy_consumption' in kwargs:
            energy_used = kwargs['energy_consumption']
            if hasattr(self, 'energy_level'):
                self.energy_level = max(0, self.energy_level - energy_used)
        
        # 能量约束检查: E_i^u(t) ≤ E_i^{max}(t)
        if hasattr(self, 'energy_level') and self.energy_level <= 0:
            self.available = False
        
        # 计算资源分配更新
        if 'resource_allocation' in kwargs and self.is_edge:
            self.resource_allocation = kwargs['resource_allocation']
        
        return self.get_state()
    
    def get_state(self):
        """获取客户端状态字典"""
        return {
            'client_id': self.client_id,
            'is_edge': self.is_edge,
            'data_size': self.local_data_size,
            'compute_capability': self.compute_capability,
            'energy_level': self.energy_level,
            'comm_rate': self.comm_rate,
            'location': self.location,
            'available': self.available
        }
    
    def sync_from_env(self, **kwargs):
        """通过关键字参数更新客户端状态，取代原有的sync_from_env"""
        for key, value in kwargs.items():
            if hasattr(self, key) and value is not None:
                setattr(self, key, value)
        return self.get_state()
    
    def _compute_gradient(self, model, loss_fn, parameters=None):
        """计算本地梯度
        
        Args:
            model: 神经网络模型
            loss_fn: 损失函数
            parameters: 模型参数（可选）
            
        Returns:
            梯度字典
        """
        if parameters is not None:
            # 加载参数到模型
            model.load_state_dict(parameters)
        
        model.train()
        model.zero_grad()
        
        # 创建数据加载器（如果不存在）
        if self.trainloader is None and self.train_ds is not None:
            self.trainloader = DataLoader(
                self.train_ds, 
                batch_size=BATCH_SIZE,
                shuffle=True
            )
        
        # 如果没有数据，返回空字典
        if self.trainloader is None or len(self.trainloader) == 0:
            return {}
        
        # 取第一个批次计算梯度
        for data, target in self.trainloader:
            data, target = data.to(self.device), target.to(self.device)
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            break
        
        # 收集梯度
        gradient_dict = {}
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                gradient_dict[name] = param.grad.data.clone()
        
        return gradient_dict
    
    def send_parameters(self, receiver, parameters, gradient=None):
        """向服务器或边缘节点发送模型参数和梯度
        
        Args:
            receiver: 接收者（服务器或边缘节点）
            parameters: 本地模型参数
            gradient: 本地梯度（可选）
            
        Returns:
            是否发送成功
        """
        # 如果接收者是边缘节点，调用其receive_parameters方法
        if hasattr(receiver, 'receive_parameters'):
            # 边缘节点接收方法
            return receiver.receive_parameters(
                self.client_id, 
                parameters, 
                gradient, 
                self.local_data_size
            )
        # 如果接收者是服务器，服务器不需要特定的接收方法
        # 因为服务器在train_round方法中直接处理客户端返回的参数
        # 这里只需要返回成功即可
        else:
            return True


class Client(ClientBase):
    """客户端类 - 继承自ClientBase，处理单个客户端的本地训练"""
    
    def __init__(self, trainDataSet, device, client_id=None):
        """初始化客户端
        
        Args:
            trainDataSet: 训练数据集
            device: 计算设备
            client_id: 客户端ID（可选）
        """
        super().__init__(trainDataSet, device, client_id, is_edge=False)
        self.trainloader, self.validloader, self.testloader = self.train_val_test(self.train_ds)

    def train_val_test(self, dataset):
        """
        为给定的数据集和用户索引返回训练、验证和测试数据加载器。
        """
        if dataset is None or len(dataset) == 0:
            return None, None, None
            
        # 划分训练、验证和测试集的索引 (80%, 10%, 10%)
        idxs = list(range(len(dataset)))
        idxs_train = idxs[:int(0.8*len(idxs))]
        idxs_val = idxs[int(0.8*len(idxs)):int(0.9*len(idxs))]
        idxs_test = idxs[int(0.9*len(idxs)):]

        trainloader = DataLoader(DatasetSplit(dataset, idxs_train),
                                 batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
        
        val_bs = max(1, int(len(idxs_val)/10)) if len(idxs_val) > 0 else 1
        validloader = DataLoader(DatasetSplit(dataset, idxs_val),
                                 batch_size=val_bs, shuffle=False, num_workers=4)
        
        test_bs = max(1, int(len(idxs_test)/10)) if len(idxs_test) > 0 else 1
        testloader = DataLoader(DatasetSplit(dataset, idxs_test),
                                batch_size=test_bs, shuffle=False, num_workers=4)
        return trainloader, validloader, testloader

    def inference(self, model):
        """ 返回推断的准确率和损失。
        """
        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0
        criterion = nn.NLLLoss().to(self.device)
        
        if not self.testloader:
            return 0, 0

        for batch_idx, (images, labels) in enumerate(self.testloader):
            images, labels = images.to(self.device), labels.to(self.device)

            # 推断
            outputs = model(images)
            batch_loss = criterion(outputs, labels)
            loss += batch_loss.item()

            # 预测
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)
        
        if total == 0:
            return 0.0, 0.0
            
        accuracy = correct/total
        return accuracy, loss

    def train(self, model, global_parameters, lr, local_epochs=3, resource_allocation=1.0, global_gradient=None):
        """
        统一的训练接口，封装了localUpdate
        
        Args:
            model: 神经网络模型
            global_parameters: 全局模型参数
            lr: 学习率
            ...
            
        Returns:
            (local_parameters, local_gradient, loss, data_size)
        """
        # 正常调用localUpdate
        local_params, local_grad = self.localUpdate(
            local_epochs, model, global_parameters, lr, resource_allocation,
            global_gradient=global_gradient
        )
        
        avg_loss = self.training_stats['loss'][-1] if self.training_stats['loss'] else 0
        return local_params, local_grad, avg_loss, self.local_data_size

    def localUpdate(self, localEpoch, model, global_parameters, lr, resource_allocation=1.0, global_gradient=None, eta=ATAFL_ETA):
        """本地训练方法
        
        Args:
            localEpoch: 本地训练轮数
            model: 神经网络模型
            global_parameters: 全局模型参数
            lr: 学习率
            resource_allocation: 资源分配参数
            global_gradient: 全局梯度（用于ATAFL算法）
            eta: ATAFL算法的超参数
            
        Returns:
            本地模型参数和本地梯度
        """
        # 记录训练开始时间
        start_time = time.time()
        
        # 如果没有数据，返回全局参数
        if self.trainloader is None:
            return global_parameters, {}
        
        # 将模型移动到设备上
        model = model.to(self.device)
        
        # 确保初始状态为全局参数 ω^t
        model.load_state_dict(global_parameters)
        
        # 创建优化器和损失函数
        opti = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.5)
        criterion = nn.NLLLoss().to(self.device)

        # 设置为训练模式
        model.train()
        
        # 本地训练循环 (公式1)
        epoch_loss = []
        epoch_range = range(localEpoch)
        if not self.disable_progress_bar:
            epoch_range = tqdm(epoch_range, desc=f"Client {self.client_id} Training")
        
        for epoch in epoch_range:
            batch_loss = []
            for data, label in self.trainloader:
                # 标准SGD训练步骤
                data, label = data.to(self.device), label.to(self.device)
                model.zero_grad()
                opti.zero_grad()
                pred = model(data)
                loss = criterion(pred, label)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opti.step()
                batch_loss.append(loss.item())
            
            # 计算平均损失
            current_epoch_loss = sum(batch_loss) / len(batch_loss) if batch_loss else 0
            epoch_loss.append(current_epoch_loss)

        avg_loss = sum(epoch_loss) / len(epoch_loss) if epoch_loss else 0
        self.training_stats['loss'].append(avg_loss)
        
        # 计算训练时间和能耗
        training_time = time.time() - start_time
        self.training_stats['training_time'].append(training_time)
        
        # 计算能耗 - 使用资源分配参数
        energy_consumption = training_time * (1.0 / self.compute_capability) * 0.1
        self.training_stats['energy_consumption'].append(energy_consumption)
        
        # 更新能量水平
        self.energy_level = max(0, self.energy_level - energy_consumption)
        if self.energy_level <= 0:
            self.available = False
        
        # 获取更新后的参数
        local_parameters = {name: param.data.clone() for name, param in model.named_parameters()}
        
        # 更新客户端状态
        self.update_state(
            energy_consumption=energy_consumption,
            local_parameters=local_parameters,
            # local_gradient=global_grad # [暂时禁用]
        )
        
        # 将参数和梯度存储到缓存中
        self.parameters_cache = local_parameters
        # self.gradient_cache = global_grad # [暂时禁用]
        
        # 返回一个空字典代替旧梯度，以确保兼容性
        return local_parameters, {}

class EdgeNode(ClientBase):
    """边缘节点类 - 继承自ClientBase，具有模型参数聚合能力"""
    
    def __init__(self, device, edge_id=None, coverage_radius=500.0):
        """初始化边缘节点
        
        Args:
            device: 计算设备
            edge_id: 边缘节点ID（可选）
            coverage_radius: 覆盖半径
        """
        # 初始化基类（注意：边缘节点不包含原始数据）
        super().__init__(None, device, edge_id if edge_id else f"edge_{id(self)}", is_edge=True)
        
        # 边缘节点特有属性
        self.coverage_radius = coverage_radius
        self.connected_clients = set()  # 连接的客户端ID集合
        
        # 参数和梯度缓存
        self.client_parameters = {}  # {client_id: parameters}
        self.client_gradients = {}    # {client_id: gradient}
        self.client_data_sizes = {}   # {client_id: data_size}
    
    def is_in_coverage(self, client_location):
        """检查客户端是否在覆盖范围内"""
        distance = calculate_distance(self.location, client_location)
        return distance <= self.coverage_radius
    
    def add_client(self, client):
        """添加客户端到连接列表"""
        if hasattr(client, 'client_id'):
            self.connected_clients.add(client.client_id)
            return True
        return False
    
    def receive_parameters(self, client_id, parameters, gradient, data_size):
        """接收终端设备传输的模型参数和梯度
        
        Args:
            client_id: 终端设备ID
            parameters: 终端设备的模型参数
            gradient: 终端设备的梯度
            data_size: 终端设备的数据量
        """
        # 存储参数和梯度
        self.client_parameters[client_id] = parameters
        if gradient is not None:
            self.client_gradients[client_id] = gradient
        self.client_data_sizes[client_id] = data_size
        
        # 添加到连接列表
        self.connected_clients.add(client_id)
    
    def aggregate_parameters(self, global_parameters=None):
        """聚合所有收到的模型参数（基于FedAvg算法）
        
        实现公式(5): ω_m^t = ∑_{i∈I_m(t)} (|D_i|/|D_{I_m(t)}|) · ω_i^t
        
        Args:
            global_parameters: 全局模型参数（可选）
            
        Returns:
            聚合后的模型参数和梯度
        """
        if not self.client_parameters or global_parameters is None:
            return global_parameters, None
        # 计算边缘域内总数据量 |D_{I_m(t)}|
        total_data = sum(self.client_data_sizes.values())
        aggregated_parameters = {}
        for key in global_parameters.keys():
            aggregated_parameters[key] = torch.zeros_like(global_parameters[key])
        
        # 加权聚合
        for client_id, parameters in self.client_parameters.items():
            weight = self.client_data_sizes[client_id] / total_data
            for key, value in parameters.items():
                if key in aggregated_parameters:
                    aggregated_parameters[key] += weight * value
        
        # 如果没有收到梯度，返回None
        if not self.client_gradients:
            return aggregated_parameters, None
        
        # 聚合梯度 - 使用相同的加权方式
        aggregated_gradient = {}
        for key in self.client_gradients[list(self.client_gradients.keys())[0]].keys():
            aggregated_gradient[key] = torch.zeros_like(self.client_gradients[list(self.client_gradients.keys())[0]][key])
        
        for client_id, gradients in self.client_gradients.items():
            weight = self.client_data_sizes[client_id] / total_data
            for key, value in gradients.items():
                if key in aggregated_gradient:
                    aggregated_gradient[key] += weight * value
        
        return aggregated_parameters, aggregated_gradient
    
    def edge_train(self, clients_dict, model, global_parameters, lr, localEpoch, resource_allocation_map):
        """边缘训练方法 - 处理模型参数而非原始数据
        
        基于ATAFL方案，边缘节点只处理模型参数和梯度，不直接访问原始数据
        
        Args:
            clients_dict: 客户端字典 {client_id: client}
            model: 神经网络模型
            global_parameters: 全局模型参数
            lr: 学习率
            localEpoch: 本地训练轮数
            resource_allocation_map: 资源分配映射
            
        Returns:
            (client_updates, client_grads, client_losses, client_data_sizes)
        """
        # 清除缓存
        self.client_parameters = {}
        self.client_gradients = {}
        self.client_data_sizes = {}
        
        # 记录训练开始时间
        start_time = time.time()
        
        # 只处理连接的客户端
        connected_clients = []
        for client_id in self.connected_clients:
            if client_id in clients_dict and clients_dict[client_id].available:
                connected_clients.append(client_id)
        
        # 如果没有连接的客户端，返回空字典
        if not connected_clients:
            return {}, {}, {}, {}

        # 客户端参数、梯度和损失的字典
        client_updates, client_grads, client_losses = {}, {}, {}
        
        # 处理每个客户端
        for client_id in connected_clients:
            client = clients_dict[client_id]
            
            # 如果客户端是边缘节点，跳过
            if client.is_edge:
                continue
            
            # 资源分配 (此处的分配逻辑可能需要根据实际需求调整)
            client_resource = resource_allocation_map.get(client_id, 1.0)
            
            # 让客户端进行本地训练
            local_parameters, local_gradient, loss, data_size = client.train(
                model=copy.deepcopy(model),
                global_parameters=global_parameters,
                lr=lr,
                local_epochs=localEpoch,
                resource_allocation=client_resource
            )
            
            # 存储结果
            client_updates[client_id] = local_parameters
            client_grads[client_id] = local_gradient
            client_losses[client_id] = loss
            self.client_data_sizes[client_id] = data_size
        
        # 计算训练时间
        training_time = time.time() - start_time
        self.training_stats['training_time'].append(training_time)
        
        # 计算能耗
        energy_consumption = training_time * (1.0 / self.compute_capability) * 0.1
        self.training_stats['energy_consumption'].append(energy_consumption)
        
        # 更新能量水平
        self.energy_level = max(0, self.energy_level - energy_consumption)
        if self.energy_level <= 0:
            self.available = False
        
        return client_updates, client_grads, client_losses, self.client_data_sizes
    
    def process_parameters_atafl(self, client_parameters, global_parameters, global_gradient, eta=ATAFL_ETA):
        """基于ATAFL方案处理模型参数
        
        Args:
            client_parameters: 客户端模型参数
            global_parameters: 全局模型参数
            global_gradient: 全局梯度
            eta: ATAFL算法的超参数
            
        Returns:
            处理后的模型参数
        """
        return apply_atafl_proxy(client_parameters, global_parameters, global_gradient, eta)


class ClientsGroup:
    """管理多个客户端的组"""
    
    def update_client_states(self, env):
        """
        从环境中更新所有客户端的状态。
        这种方法更健壮，因为它直接访问env的属性，而不是解析脆弱的状态向量。
        """
        if not hasattr(env, 'N') or not hasattr(env, 'M'):
            return

        # 更新普通客户端状态
        for i in range(env.N):
            client_id = f"client{i}"
            if client_id in self.clients:
                client = self.clients[client_id]
                # 从Env获取权威状态
                remaining_energy = env.energy_max - env.queues[i].queue
                client_state_update = {
                    'compute_capability': env.f_l[i],
                    'energy_level': remaining_energy, # 基于队列计算剩余能量
                    'available': remaining_energy > 0 # 可用性直接由剩余能量决定
                }
                client.sync_from_env(**client_state_update)
        
        # 更新边缘节点状态
        for i in range(env.M):
            edge_id = f"edge_{i}"
            if edge_id in self.edge_nodes:
                edge_node = self.edge_nodes[edge_id]
                edge_state_update = {
                    'compute_capability': env.F_e[i] # F_e 是边缘节点的最大计算能力
                }
                edge_node.sync_from_env(**edge_state_update)
    
    def __init__(self, dataset_name, is_iid, num_clients, device, num_edges=2, non_iid_level=0.5, disable_progress_bar=False, env_datasets=None):
        """初始化客户端组
        
        Args:
            dataset_name: 数据集名称
            is_iid: 是否使用IID数据分布
            num_clients: 客户端数量
            device: 计算设备
            num_edges: 边缘节点数量
            non_iid_level: 非IID程度
            disable_progress_bar: 是否禁用进度条
            env_datasets: 环境预分配的数据集信息，如果提供则使用环境的数据集
        """
        # 设置随机种子确保可复现性
        set_seed(SEED)
        
        # 基本属性
        self.dataset_name = dataset_name
        self.is_iid = is_iid
        self.num_clients = num_clients
        self.device = device
        self.num_edges = num_edges
        self.non_iid_level = non_iid_level
        self.disable_progress_bar = disable_progress_bar
        
        # 创建客户端和边缘节点
        self.clients = {}
        self.edge_nodes = {}
        
        # 如果提供了env_datasets，直接使用
        if env_datasets and 'train_data' in env_datasets:
            self.train_data = env_datasets['train_data']
            self.test_data = env_datasets.get('test_data', None)
            
            # 初始化客户端和边缘节点
            self._setup_clients(self.train_data)
            self._setup_edge_nodes()
            
            # 分配数据给客户端
            self._allocate_data(self.train_data)
            
            # 连接客户端到边缘节点
            self._connect_clients_to_edges()
            
        else:
            # 如果环境没有预分配数据集，则自己加载
            print("加载并分配数据集")
            train_data, test_data = get_dataset(dataset_name)
            self.test_data = test_data
            
            # 分配数据到客户端
            self._setup_clients(train_data)
            
            # 创建边缘节点
            self._setup_edge_nodes()
            
            # 建立客户端和边缘节点之间的连接
            self._connect_clients_to_edges()
            
            # 分配数据
            self._allocate_data(train_data)
    
    def update_datasets(self, env_datasets):
        """从环境更新数据集信息
        
        Args:
            env_datasets: 环境提供的数据集信息字典
                包含 'train_data', 'test_data', 'is_iid', 'non_iid_level' 等键
        
        Returns:
            是否成功更新
        """
        if env_datasets is None or 'train_data' not in env_datasets or 'test_data' not in env_datasets:
            print("警告: 环境提供的数据集信息不完整，无法更新")
            return False
        
        print("从环境更新数据集信息")
        
        # 更新数据集相关属性
        if 'is_iid' in env_datasets:
            self.is_iid = env_datasets['is_iid']
        if 'non_iid_level' in env_datasets and not self.is_iid:
            self.non_iid_level = env_datasets['non_iid_level']
        
        # 更新测试数据集
        if 'test_data' in env_datasets:
            self.test_data = env_datasets['test_data']
        
        # 如果客户端尚未创建，则使用环境数据集重新分配
        if not self.clients and 'train_data' in env_datasets:
            self._setup_clients(env_datasets['train_data'])
            self._connect_clients_to_edges()
        
        return True
    
    def get_data_sizes(self):
        """
        获取各个客户端的数据集样本数量。
        返回一个只包含样本数量的数组，具体的字节大小转换由环境模型处理。
        """
        data_sizes = []
        # 确保按client ID顺序返回
        for i in range(self.num_clients):
            client_id = f"client{i}"
            if client_id in self.clients:
                data_sizes.append(self.clients[client_id].local_data_size)
            else:
                data_sizes.append(0) # 如果客户端不存在，则为0
        
        return np.array(data_sizes)
    
    def _setup_clients(self, train_data):
        """创建并初始化客户端
        
        Args:
            train_data: 训练数据集
        """
        # 分配数据到客户端
        data_dict = self._allocate_data(train_data)
        
        # 创建客户端
        for i in range(self.num_clients):
            client_id = f"client{i}"
            client_data = data_dict.get(i, None)
            # 创建客户端实例
            self.clients[client_id] = Client(client_data, self.device, client_id)
            # 随机分配客户端位置
            self.clients[client_id].location = [random.uniform(0, 1000), random.uniform(0, 1000)]
    
    def _setup_edge_nodes(self):
        """创建边缘节点"""
        for i in range(self.num_edges):
            edge_id = f"edge_{i}"
            
            # 创建边缘节点
            self.edge_nodes[edge_id] = EdgeNode(self.device, edge_id, coverage_radius=500.0)
            
            # 随机分配边缘节点位置
            self.edge_nodes[edge_id].location = [random.uniform(0, 1000), random.uniform(0, 1000)]
            
            # 将边缘节点也添加到clients字典中，方便统一管理
            self.clients[edge_id] = self.edge_nodes[edge_id]
    
    def _connect_clients_to_edges(self):
        """建立客户端和边缘节点之间的连接"""
        # 对每个客户端，连接到覆盖范围内的边缘节点
        for client_id, client in self.clients.items():
            # 跳过边缘节点
            if client.is_edge:
                continue
                
            # 对每个边缘节点，检查是否在覆盖范围内
            for edge_id, edge_node in self.edge_nodes.items():
                if edge_node.is_in_coverage(client.location):
                    edge_node.add_client(client)
    
    def _allocate_data(self, train_data):
        """根据是否为IID分配数据
        
        Args:
            train_data: 训练数据集
            
        Returns:
            数据字典 {client_id: data}
        """
        # 导入sampling模块中的数据分配函数
        from utils.sampling import mnist_iid, mnist_noniid, cifar_iid, cifar_noniid
        
        # 获取数据集
        if hasattr(train_data, 'dataset'):
            dataset = train_data.dataset
        else:
            dataset = train_data
            
        # 根据数据集类型和是否IID选择不同的分配函数
        if 'mnist' in self.dataset_name.lower():
            if self.is_iid:
                    user_groups = mnist_iid(dataset, self.num_clients)
            else:
                    user_groups = mnist_noniid(dataset, self.num_clients)
        elif 'cifar' in self.dataset_name.lower():
                if self.is_iid:
                    user_groups = cifar_iid(dataset, self.num_clients)
                else:
                    user_groups = cifar_noniid(dataset, self.num_clients)
        else:
                # 默认情况，简单平均分配
                return self._allocate_iid_data(train_data)
            
        # 将索引字典转换为数据集字典
        data_dict = {}
        for i in range(self.num_clients):
            # 获取客户端数据索引
            idxs = user_groups[i]
            
            # 确保索引是列表
            if isinstance(idxs, set):
                idxs = list(idxs)
            
            # Convert indices to integers (FIX ADDED HERE)
            if isinstance(idxs, np.ndarray):
                idxs = idxs.astype(int)
            elif isinstance(idxs, list):
                idxs = [int(idx) for idx in idxs]
            
            from torch.utils.data import Subset
            if isinstance(idxs, np.ndarray):
                idxs = idxs.tolist()
            client_dataset = Subset(dataset, idxs)
            data_dict[i] = client_dataset
        
        return data_dict
    
    def _allocate_iid_data(self, train_data):
        """IID方式分配数据
        
        Args:
            train_data: 训练数据集(DataLoader对象)
            
        Returns:
            数据字典 {client_id: client_dataset}
        """
        # 从DataLoader中提取所有数据
        all_data = []
        all_targets = []
        
        # 逐批次处理数据
        for batch_data, batch_targets in train_data:
            batch_size = batch_data.size(0)
            for i in range(batch_size):
                img = batch_data[i]  # 获取单个图像
                # 确保标签是整数类型
                if isinstance(batch_targets[i], torch.Tensor):
                    label = batch_targets[i].item()
                else:
                    label = int(batch_targets[i])
                all_data.append(img)
                all_targets.append(label)
        
        # 转换为张量
        all_data = torch.stack(all_data)
        all_targets = torch.tensor(all_targets, dtype=torch.int64)
        
        # 随机打乱数据
        indices = torch.randperm(len(all_data))
        all_data = all_data[indices]
        all_targets = all_targets[indices]
        
        # 平均分配给每个客户端
        data_dict = {}
        samples_per_client = len(all_data) // self.num_clients
        
        for i in range(self.num_clients):
            start_idx = i * samples_per_client
            end_idx = (i + 1) * samples_per_client if i < self.num_clients - 1 else len(all_data)
            
            # 创建客户端数据集
            client_data = TensorDataset(all_data[start_idx:end_idx], all_targets[start_idx:end_idx])
            data_dict[i] = client_data
        
        return data_dict
    
    def train_clients(self, selected_clients, model, loss_func, optimizer, global_parameters, 
                        epochs=LOCAL_EPOCHS, batch_size=BATCH_SIZE, resource_allocation=None):
        """在指定的客户端上执行训练

        Args:
            selected_clients: 选定的客户端ID列表
            model: 神经网络模型
            loss_func: 损失函数
            optimizer: 优化器
            global_parameters: 全局模型参数
            epochs: 训练轮数
            batch_size: 批次大小
            resource_allocation: 资源分配字典 {client_id: resource_ratio}

        Returns:
            (updates, gradients, data_sizes)
        """
        updates = {}
        gradients = {}
        data_sizes = {}
        
        if resource_allocation is None:
            resource_allocation = {}
            
        # 设置每个客户端的进度条显示参数
        for client_id in selected_clients:
            if client_id in self.clients:
                self.clients[client_id].disable_progress_bar = self.disable_progress_bar
        
        for client_id in selected_clients:
            if client_id not in self.clients:
                print(f"警告: 客户端 {client_id} 不存在")
                continue
                
            client_obj = self.clients[client_id]
            
            # 如果客户端不可用，跳过
            if not client_obj.available:
                print(f"客户端 {client_id} 不可用 (能量不足或其他原因)")
                continue
                
            # 获取该客户端的资源分配
            client_resource = resource_allocation.get(client_id, 1.0)
            
            # 根据客户端类型选择训练方法
            if client_obj.is_edge:
                # 边缘节点训练
                edge_node = client_obj
                aggregated_params, _ = edge_node.edge_train(
                    self.clients, epochs, model,
                    global_parameters, client_resource
                )
                local_params = aggregated_params
                local_grad = None  # 边缘节点不直接计算梯度
                loss = 0  # 边缘节点不直接计算损失
                data_size = sum(edge_node.client_data_sizes.values()) if edge_node.client_data_sizes else 0
            else:
                # 普通客户端训练
                local_params, local_grad, loss, data_size = client_obj.train(
                    model=model,
                    global_parameters=global_parameters,
                    local_epochs=epochs,
                    lr=0.01, # 修正：需要一个学习率参数
                    resource_allocation=client_resource
                )
            
            # 保存更新和数据大小
            updates[client_id] = local_params
            gradients[client_id] = local_grad
            data_sizes[client_id] = data_size
            
        return updates, gradients, data_sizes
    
    def get_edge_for_client(self, client_id):
        """获取客户端所连接的边缘节点
        
        Args:
            client_id: 客户端ID
            
        Returns:
            与该客户端连接的边缘节点ID列表
        """
        if client_id not in self.clients:
            return []
            
        connected_edges = []
        for edge_id, edge_node in self.edge_nodes.items():
            if client_id in edge_node.connected_clients:
                connected_edges.append(edge_id)
                
        return connected_edges


# 如果直接运行此文件则执行测试代码
if __name__=="__main__":
    MyClients = ClientsGroup('mnist', True, 10, torch.device("cuda"), num_edges=2)
    print("客户端数量:", len(MyClients.clients) - MyClients.num_edges)
    print("边缘节点数量:", MyClients.num_edges)
    
    # 打印一个普通客户端的数据
    print("客户端示例:")
    print(MyClients.clients['client0'].get_state())
    
    # 打印一个边缘节点的数据
    print("边缘节点示例:")
    print(MyClients.edge_nodes['edge_0'].get_state())
    print("连接的客户端:", MyClients.edge_nodes['edge_0'].connected_clients)
