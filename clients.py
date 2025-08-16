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
from torch.utils.data import DataLoader, Dataset, Subset

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# 导入配置参数
from config import *
from utils.common_utils import set_seed, calculate_distance

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image.clone().detach(), torch.tensor(label)

class ClientBase:
    def __init__(self, trainDataSet, device, client_id=None, is_edge=False):
        self.train_ds = trainDataSet
        self.device = device
        self.trainloader, self.validloader, self.testloader = None, None, None
        self.client_id = client_id if client_id else f"client_{id(self)}"
        self.is_edge = is_edge
        self.disable_progress_bar = DISABLE_PROGRESS_BAR
        self.local_data_size = len(trainDataSet) if trainDataSet else 0
        self.compute_capability = MIN_COMPUTE_CAPABILITY if not is_edge else MAX_COMPUTE_CAPABILITY
        self.energy_level = MIN_ENERGY if not is_edge else MAX_ENERGY
        self.comm_rate = MIN_COMM_RATE if not is_edge else MAX_COMM_RATE
        self.location = [random.uniform(0, 1000), random.uniform(0, 1000)]
        self.available = True
        self.training_stats = {'training_time': [], 'loss': []}
        self.parameters_cache = {}
        self.gradient_cache = {}
        self.env = None

    def update_state(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def get_state(self):
        return {
            'client_id': self.client_id, 'is_edge': self.is_edge,
            'data_size': self.local_data_size, 'compute_capability': self.compute_capability,
            'energy_level': self.energy_level, 'comm_rate': self.comm_rate,
            'location': self.location, 'available': self.available
        }

    def train_model(self, model, global_parameters, lr, local_epochs=3, data_loader=None):
        model.load_state_dict(global_parameters)
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        # 核心修正：使用 NLLLoss，因为它与模型输出的 log_softmax 相匹配
        loss_fn = nn.NLLLoss()
        
        loader = data_loader if data_loader is not None else self.trainloader
        if loader is None or len(loader) == 0:
            # 如果没有训练数据，返回一个合理的默认损失值
            return global_parameters, {}, 1.0, 0.01
            
        start_time = time.time()
        model.train()
        epoch_losses = []
        
        for epoch in range(local_epochs):
            batch_losses = []
            for _, (images, labels) in enumerate(loader):
                # 优化GPU数据传输
                images = images.to(self.device, non_blocking=NON_BLOCKING)
                labels = labels.to(self.device, non_blocking=NON_BLOCKING)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()
                
                # 减少CPU-GPU同步开销
                batch_losses.append(loss.item())
            
            epoch_loss = sum(batch_losses) / len(batch_losses) if batch_losses else 0
            epoch_losses.append(epoch_loss)
        
        final_loss = np.mean(epoch_losses) if epoch_losses else 0
        training_time = time.time() - start_time
        
        final_params = model.state_dict()
        final_grads = {name: param.grad.clone() for name, param in model.named_parameters() if param.grad is not None}
        
        self.training_stats['training_time'].append(training_time)
        self.training_stats['loss'].append(final_loss)
        
        return final_params, final_grads, final_loss, training_time

class Client(ClientBase):
    def __init__(self, trainDataSet, device, client_id=None):
        super().__init__(trainDataSet, device, client_id, is_edge=False)
        self.trainloader, self.validloader, self.testloader = self.train_val_test(self.train_ds)

    def train_val_test(self, dataset):
        if dataset is None or len(dataset) == 0:
            return None, None, None
        idxs = list(range(len(dataset)))
        idxs_train = idxs[:int(TRAIN_RATIO*len(idxs))]
        idxs_val = idxs[int(TRAIN_RATIO*len(idxs)):int((TRAIN_RATIO+VAL_RATIO)*len(idxs))]
        idxs_test = idxs[int((TRAIN_RATIO+VAL_RATIO)*len(idxs)):]
        # 优化DataLoader配置以提高训练速度
        trainloader = DataLoader(
            DatasetSplit(dataset, idxs_train), 
            batch_size=BATCH_SIZE, 
            shuffle=True, 
            num_workers=NUM_WORKERS,      # 多线程数据加载
            pin_memory=PIN_MEMORY,        # GPU内存固定
            persistent_workers=True,      # 持久化工作进程
            prefetch_factor=2             # 预取因子
        )
        val_bs = max(1, int(len(idxs_val)/10)) if len(idxs_val) > 0 else 1
        validloader = DataLoader(
            DatasetSplit(dataset, idxs_val), 
            batch_size=val_bs, 
            shuffle=False, 
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY,
            persistent_workers=True
        )
        test_bs = max(1, int(len(idxs_test)/10)) if len(idxs_test) > 0 else 1
        testloader = DataLoader(
            DatasetSplit(dataset, idxs_test), 
            batch_size=test_bs, 
            shuffle=False, 
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY,
            persistent_workers=True
        )
        return trainloader, validloader, testloader

    def train(self, model, global_parameters, lr, local_epochs=3, **kwargs):
        params, grads, loss, train_time = self.train_model(
            model=model, global_parameters=global_parameters,
            lr=lr, local_epochs=local_epochs
        )
        return params, grads, loss, train_time, self.local_data_size

class EdgeNode(ClientBase):
    def __init__(self, device, edge_id=None, coverage_radius=EDGE_COVERAGE_RADIUS):
        super().__init__(None, device, edge_id, is_edge=True)
        self.coverage_radius = coverage_radius
        self.connected_clients = {}
    
    def add_client(self, client):
        if hasattr(client, 'client_id'):
            self.connected_clients[client.client_id] = client
            return True
        return False

    def train(self, model, global_parameters, lr, local_epochs=3, client_on_behalf_of=None, **kwargs):
        if client_on_behalf_of is None:
            raise ValueError("EdgeNode.train需要一个'client_on_behalf_of'参数")
        if not hasattr(client_on_behalf_of, 'trainloader') or client_on_behalf_of.trainloader is None:
            return {}, {}, 0, 0, 0
        params, grads, loss, train_time = self.train_model(
            model=model, global_parameters=global_parameters,
            lr=lr, local_epochs=local_epochs,
            data_loader=client_on_behalf_of.trainloader
        )
        return params, grads, loss, train_time, client_on_behalf_of.local_data_size

class ClientsGroup:
    def __init__(self, dataset_name=DEFAULT_DATASET, is_iid=IID, num_clients=NUM_CLIENTS, 
                 device=DEVICE, num_edges=NUM_EDGES, non_iid_level=NON_IID_LEVEL):
        set_seed(SEED)
        self.dataset_name = dataset_name
        self.is_iid = is_iid
        self.num_clients = num_clients
        self.device = device
        self.num_edges = num_edges
        self.non_iid_level = non_iid_level
        self.clients = {}
        self.edge_nodes = {}
        self.test_data = None

    def setup_infrastructure(self, train_data):
        self._setup_clients(train_data)
        self._setup_edge_nodes()
        self._connect_clients_to_edges()

    def _setup_clients(self, train_data):
        # 根据IID/非IID设置划分数据
        from utils.sampling import mnist_iid, mnist_noniid, cifar_iid, cifar_noniid
        
        # 如果是数据加载器，获取其中的数据集
        if hasattr(train_data, 'dataset'):
            dataset = train_data.dataset
        else:
            dataset = train_data
            
        # 根据dataset_name和IID设置决定使用哪个函数
        if self.is_iid:
            user_groups = mnist_iid(dataset, self.num_clients) if self.dataset_name.lower() == 'mnist' else cifar_iid(dataset, self.num_clients)
        else:
            user_groups = mnist_noniid(dataset, self.num_clients) if self.dataset_name.lower() == 'mnist' else cifar_noniid(dataset, self.num_clients)
        
        # 为每个客户端分配数据
        from torch.utils.data import Subset
        for i in range(self.num_clients):
            client_id = f"client{i}"
            
            # 如果数据分配中有该客户端的索引
            if i in user_groups and len(user_groups[i]) > 0:
                # 创建数据子集
                client_data = Subset(dataset, list(user_groups[i]))
            else:
                client_data = None
            
            # 创建客户端实例
            self.clients[client_id] = Client(client_data, self.device, client_id)

    def _setup_edge_nodes(self):
        for i in range(self.num_edges):
            edge_id = f"edge_{i}"
            edge_node = EdgeNode(self.device, edge_id)
            self.edge_nodes[edge_id] = edge_node
            self.clients[edge_id] = edge_node
            
    def _connect_clients_to_edges(self):
        for client_id, client in self.clients.items():
            if client.is_edge:
                continue
            for edge_id, edge_node in self.edge_nodes.items():
                if self._is_in_coverage(client.location, edge_node.location, edge_node.coverage_radius):
                    edge_node.add_client(client)

    def _is_in_coverage(self, client_loc, edge_loc, radius):
        return calculate_distance(client_loc, edge_loc) <= radius

    def update_client_states(self, env):
        if not hasattr(env, 'N') or not hasattr(env, 'M'): return
        for i in range(env.N):
            client_id = f"client{i}"
            if client_id in self.clients:
                if i < len(env.queue_manager.queues):
                    remaining_energy = env.energy_max - env.queue_manager.queues[i].queue
                    self.clients[client_id].update_state(
                        compute_capability=env.comp_model.f_l[i],
                        energy_level=remaining_energy,
                        available=remaining_energy > 0
                    )
        for i in range(env.M):
            edge_id = f"edge_{i}"
            if edge_id in self.edge_nodes:
                self.edge_nodes[edge_id].update_state(compute_capability=env.comp_model.F_e[i])

    def get_data_sizes(self):
        data_sizes = []
        for i in range(self.num_clients):
            client_id = f"client{i}"
            if client_id in self.clients:
                data_sizes.append(self.clients[client_id].local_data_size)
            else:
                data_sizes.append(0)
        return np.array(data_sizes)
        
    def get_edge_for_client(self, client_id):
        connected_edges = []
        if client_id in self.clients and not self.clients[client_id].is_edge:
            client_loc = self.clients[client_id].location
            for edge_id, edge_node in self.edge_nodes.items():
                if self._is_in_coverage(client_loc, edge_node.location, edge_node.coverage_radius):
                    connected_edges.append(edge_id)
        return connected_edges
