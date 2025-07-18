import os
import argparse
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
import time
import copy
from torch.utils.data import DataLoader
# 使用新的模型导入
from models.models import create_model
import torch.nn as nn

# 模型收敛阈值
CONVERGENCE_EPSILON = 1e-3

class FederatedServer:
    """联邦学习服务器 - 处理模型聚合与评估"""
    def __init__(self, args, model, clients_manager):
        """初始化联邦学习服务器
        
        Args:
            args: 配置参数，可以是字典或argparse.Namespace
            model: 模型实例或模型名称字符串
            clients_manager: 客户端管理器实例
        """
        # 标准化参数处理
        self.args = args if isinstance(args, dict) else args.__dict__ if hasattr(args, '__dict__') else {}
        
        # 设置设备
        self.dev = self.args.get('device', torch.device("cuda"))
        self.disable_progress_bar = bool(self.args.get('disable_progress_bar', False))
        
        # 初始化模型
        if isinstance(model, str):
            self.model = create_model(model, self.args)
        else:
            self.model = model
        self.model = self.model.to(self.dev)
        
        # 初始化全局参数
        self.global_parameters = {}
        for key, var in self.model.state_dict().items():
            self.global_parameters[key] = var.clone()
        
        # 初始化收敛相关变量
        self.best_params = None
        self.best_delta = float('inf')
        
        # 客户端管理
        self.clients_manager = clients_manager
        
        # 训练统计
        self.training_stats = {
            'round': 0,
            'accuracy': [],
            'selected_clients': [],
            'aggregation_location': [],
            'training_time': [],
            'energy_consumption': [],
            'loss': [],
            'rounds_completed': 0,
            'convergence_delta': []
        }
        # 显式声明env属性，避免类型检查器报错
        self.env = None
    
    def select_clients(self, clients_states, num_clients=None):
        """选择客户端进行训练 - 选择所有可用的终端设备
        
        Args:
            clients_states: 客户端状态字典 {client_id: state_dict}
            num_clients: 此参数已不再使用，保留是为了兼容性
            
        Returns:
            选择的客户端ID列表 - 所有可用的终端设备
        """
        # 选择所有可用的终端设备
        available_clients = [
            cid for cid, state in clients_states.items() 
            if state['available'] and not state['is_edge']
        ]
        
        return available_clients
            
    def _default_client_selection(self, clients_states, num_clients):
        """已弃用 - 为兼容性保留"""
        return self.select_clients(clients_states)
    
    def select_aggregation_location(self, aggregation_decision):
        """根据聚合决策选择聚合位置
        
        Args:
            aggregation_decision: 聚合决策 (0=云端, 1=边缘)
            
        Returns:
            聚合位置字符串
        """
        if aggregation_decision == 1:
            # 选择一个边缘节点用于聚合
            if hasattr(self.clients_manager, 'edge_nodes') and self.clients_manager.edge_nodes:
                edge_ids = list(self.clients_manager.edge_nodes.keys())
                return edge_ids[0]  # 简单返回第一个边缘节点
            
        # 默认使用云端聚合
        return 'cloud'
    
    def edge_aggregate(self, updates_dict, data_sizes, edge_id, gradients_dict=None):
        """在边缘节点上聚合模型参数和梯度（ATAFL方案）
        
        Args:
            updates_dict: 客户端模型参数字典 {client_id: parameters}
            data_sizes: 客户端数据量字典 {client_id: data_size}
            edge_id: 边缘节点ID
            gradients_dict: 客户端梯度字典 {client_id: gradient}
            
        Returns:
            (aggregated_parameters, aggregated_gradient, aggregation_time) 元组
        """
        start_time = time.time()
        
        if not updates_dict:
            return self.global_parameters, None, 0
        
        # 计算总数据量（用于加权）
        total_data = sum(data_sizes.values())
        if total_data == 0:
            return self.global_parameters, None, 0
        
        # 初始化聚合参数和梯度
        aggregated_parameters = {}
        aggregated_gradient = {}
        
        # 初始化聚合参数和梯度字典
        for key in self.global_parameters.keys():
            aggregated_parameters[key] = torch.zeros_like(self.global_parameters[key])
            if gradients_dict is not None:
                aggregated_gradient[key] = torch.zeros_like(self.global_parameters[key])
        
        # 应用公式(5)：加权聚合参数和梯度
        for client_id, parameters in updates_dict.items():
            # 数据量加权
            weight = data_sizes[client_id] / total_data
            
            # 聚合模型参数（ω^t = ∑(|D_i|/|D_I(t)|)ω_i^t）
            for key, value in parameters.items():
                if key in aggregated_parameters:
                    aggregated_parameters[key] += weight * value
            
            # 聚合梯度（∇F^t = ∑(|D_i|/|D_I(t)|)∇F_i(ω_i^t)）
            if gradients_dict is not None and client_id in gradients_dict:
                client_gradient = gradients_dict[client_id]
                for key, value in client_gradient.items():
                    if key in aggregated_gradient:
                        aggregated_gradient[key] += weight * value
        
        # 模拟边缘节点的计算延迟
        if hasattr(self.clients_manager, 'edge_nodes') and edge_id in self.clients_manager.edge_nodes:
            edge_node = self.clients_manager.edge_nodes[edge_id]
            compute_delay = 1.0 / edge_node.compute_capability if hasattr(edge_node, 'compute_capability') else 1.0
            time.sleep(compute_delay * 0.01)  # 缩小延迟以加快模拟
        
        # 计算聚合时间
        aggregation_time = time.time() - start_time
        
        # 返回聚合后的参数、梯度和聚合时间
        return aggregated_parameters, aggregated_gradient if gradients_dict else None, aggregation_time
    
    def cloud_aggregate(self, updates_dict, data_sizes, gradients_dict=None):
        """在云服务器上聚合模型参数和梯度（FedAvg算法）
        
        精确实现公式(2): ω^t = ∑_{i∈I(t)} (|D_i|/|D_{I(t)}|) · ω_i^t
        
        Args:
            updates_dict: 客户端模型参数字典 {client_id: parameters}
            data_sizes: 客户端数据量字典 {client_id: data_size}
            gradients_dict: 客户端梯度字典 {client_id: gradient}，可选参数
            
        Returns:
            (aggregated_parameters, aggregated_gradient, aggregation_time) 元组
        """
        start_time = time.time()
        
        # 如果没有更新，返回全局参数
        if not updates_dict:
            return self.global_parameters, None, 0
        
        # 计算总数据量 |D_{I(t)}|
        total_data = sum(data_sizes.values())
        if total_data == 0:
            return self.global_parameters, None, 0
        
        # 初始化聚合参数
        aggregated_parameters = {}
        for key in self.global_parameters.keys():
            aggregated_parameters[key] = torch.zeros_like(self.global_parameters[key])
        
        # 加权聚合
        for client_id, parameters in updates_dict.items():
            weight = data_sizes[client_id] / total_data
            for key, value in parameters.items():
                if key in aggregated_parameters:
                    aggregated_parameters[key] += weight * value
        
        # 检查收敛性 ||ω^t - ω*|| ≤ ε
        convergence_delta = self._check_convergence(aggregated_parameters)
        self.training_stats['convergence_delta'].append(convergence_delta)
        if convergence_delta <= CONVERGENCE_EPSILON:
            print(f"模型已在轮次 {self.training_stats['round']} 收敛 (||Δ|| ≤ {CONVERGENCE_EPSILON})")
        
        # 计算聚合时间
        aggregation_time = time.time() - start_time
        
        # 如果需要聚合梯度，但没有提供
        if gradients_dict is None:
            return aggregated_parameters, None, aggregation_time
        
        # 如果提供了梯度，聚合梯度
        aggregated_gradient = {}
        for key in self.global_parameters.keys():
            aggregated_gradient[key] = torch.zeros_like(self.global_parameters[key])
        
        for client_id, gradients in gradients_dict.items():
            if client_id in data_sizes:
                weight = data_sizes[client_id] / total_data
                for key, value in gradients.items():
                    if key in aggregated_gradient:
                        aggregated_gradient[key] += weight * value
        
        return aggregated_parameters, aggregated_gradient, aggregation_time
    
    def _check_convergence(self, new_params):
        """检查模型是否满足 ||ω^t - ω*|| ≤ ε
        
        计算当前参数与最优参数之间的欧氏距离，判断是否收敛
        
        Args:
            new_params: 新的模型参数
            
        Returns:
            param_delta: 参数变化量
        """
        if not hasattr(self, 'best_params') or self.best_params is None:
            self.best_params = copy.deepcopy(new_params)
            return float('inf')
        
        # 计算参数变化
        param_delta = 0.0
        for name in new_params:
            if name in self.best_params:
                param_delta += torch.norm(new_params[name] - self.best_params[name]).item()
        
        # 更新最优参数
        if param_delta < self.best_delta:
            self.best_params = copy.deepcopy(new_params)
            self.best_delta = param_delta
        
        # 返回参数变化量
        return param_delta
    
    def aggregate(self, updates_dict, data_sizes, aggregation_location="cloud", gradients_dict=None):
        """根据指定位置进行聚合（支持ATAFL方案）
        
        Args:
            updates_dict: 客户端模型参数字典 {client_id: parameters}
            data_sizes: 客户端数据量字典 {client_id: data_size}
            aggregation_location: 聚合位置（"cloud"或边缘节点ID）
            gradients_dict: 客户端梯度字典 {client_id: gradient}
            
        Returns:
            (aggregated_parameters, aggregated_gradient, aggregation_time) 元组
        """
        if aggregation_location == "cloud":
            return self.cloud_aggregate(updates_dict, data_sizes, gradients_dict)
        else:
            # 如果是边缘节点ID，在边缘节点上聚合
            # 注意：边缘聚合方法也需要更新以支持梯度聚合
            return self.edge_aggregate(updates_dict, data_sizes, aggregation_location, gradients_dict)
    
    def update_global_model(self, aggregated_parameters):
        """更新全局模型参数
        
        根据聚合节点计算的聚合参数更新全局模型
        公式(5): ω^t = ∑_{i=1}^{I(t)} |D_i|/|D_{I(t)}| · ω_i^t
        """
        # 更新全局参数 ω^t
        self.global_parameters = aggregated_parameters
        
        # 将参数加载到模型中
        self.model.load_state_dict(self.global_parameters, strict=True)
        
        # 更新训练统计
        self.training_stats['rounds_completed'] += 1
    
    def evaluate(self, test_dataloader=None):
        """评估全局模型性能"""
        if test_dataloader is None:
            # 尝试不同的属性名获取测试数据
            if hasattr(self.clients_manager, 'test_data_loader'):
                test_dataloader = self.clients_manager.test_data_loader
            elif hasattr(self.clients_manager, 'test_data'):
                # 如果是原始测试数据，创建数据加载器
                test_data = self.clients_manager.test_data
                if not isinstance(test_data, DataLoader):
                    batch_size = self.args.get('batch_size', 64)
                    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
                else:
                    test_dataloader = test_data
            else:
                # 如果没有测试数据，返回默认准确率
                print("\n警告: 无法找到测试数据，跳过评估")
                return 0.0
            
        # 修改这部分代码，确保test_dataloader的dataset不是另一个DataLoader
        if isinstance(test_dataloader, DataLoader) and isinstance(test_dataloader.dataset, DataLoader):
            # 如果test_dataloader.dataset也是DataLoader，使用它的dataset
            test_data = test_dataloader.dataset.dataset  # 获取真正的Dataset
            batch_size = self.args.get('batch_size', 64)
            test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
        elif not isinstance(test_dataloader, DataLoader):
            # 原有的检查逻辑...
            pass
        
        self.model.load_state_dict(self.global_parameters, strict=True)
        self.model.eval()
        
        sum_accu = 0
        num = 0
        total_loss = 0
        
        # 使用tqdm包装测试数据
        with torch.no_grad(), tqdm(test_dataloader, desc="Evaluating", leave=False, position=1, 
                                  disable=self.disable_progress_bar) as progress_bar:
            criterion = nn.NLLLoss().to(self.dev)
            for data, label in progress_bar:
                data, label = data.to(self.dev), label.to(self.dev)
                preds = self.model(data)
                
                # 计算损失
                loss = criterion(preds, label)
                total_loss += loss.item()
                
                preds = torch.argmax(preds, dim=1)
                sum_accu += (preds == label).float().mean()
                num += 1
                
                # 更新进度条
                progress_bar.set_postfix({
                    "Acc": f"{(sum_accu/num):.4f}", 
                    "Loss": f"{(total_loss/num):.4f}",
                    "Batch": f"{num}/{len(test_dataloader)}"
                })
                
        accuracy = sum_accu / num  # 直接用float，不用.item()
        avg_loss = total_loss / num
        
        self.training_stats['accuracy'].append(accuracy)
        self.training_stats['loss'].append(avg_loss)
        
        return accuracy
    
    def save_model(self, save_path=None, add_round=True):
        """保存全局模型"""
        if save_path is None:
            save_path = self.args.get('save_path', './checkpoints')
        
        if not os.path.isdir(save_path):
            os.makedirs(save_path, exist_ok=True)
        
        round_num = self.training_stats['round']
        model_name = self.args.get('model_name', 'model')
        
        filename = f"{model_name}"
        if add_round:
            filename += f"_round{round_num}"
        
        torch.save(self.model, os.path.join(save_path, filename))
    
    def get_stats(self):
        """获取训练统计信息"""
        return self.training_stats
    
    def _train_terminal_clients(self, terminal_clients, resource_allocation):
        """在终端设备上执行本地训练
        
        Args:
            terminal_clients: 需要进行本地训练的终端设备ID列表
            resource_allocation: 资源分配字典 {client_id: allocation}
            
        Returns:
            (terminal_updates, terminal_gradients, terminal_data_sizes, terminal_losses) 元组
        """
        terminal_updates, terminal_data_sizes, terminal_gradients, terminal_losses = {}, {}, {}, {}

        if not terminal_clients:
            return terminal_updates, terminal_gradients, terminal_data_sizes, terminal_losses

        with tqdm(total=len(terminal_clients), desc="本地训练", unit="client", 
                  disable=self.disable_progress_bar, leave=False, position=1) as pbar:
            for client_id in terminal_clients:
                client = self.clients_manager.clients.get(client_id)
                if not client or not client.available:
                    pbar.update(1)
                    continue

                pbar.set_description(f"本地训练: {client_id}")
                
                client_resource = resource_allocation.get(client_id, 1.0)
                
                local_updates, local_gradients, client_loss, data_size = client.train(
                    model=copy.deepcopy(self.model),
                    global_parameters=self.global_parameters,
                    lr=self.args.get('learning_rate'),
                    local_epochs=self.args.get('local_epochs', 3),
                    resource_allocation=client_resource
                )
                
                terminal_updates[client_id] = local_updates
                terminal_gradients[client_id] = local_gradients
                terminal_data_sizes[client_id] = data_size
                terminal_losses[client_id] = client_loss
                pbar.update(1)
        
        return terminal_updates, terminal_gradients, terminal_data_sizes, terminal_losses
    
    def _train_edge_nodes(self, client_edge_mapping, resource_allocation):
        """在边缘节点上为卸载的客户端执行训练
        
        Args:
            client_edge_mapping: 客户端到边缘节点的映射 {client_id: edge_id}
            resource_allocation: 资源分配字典，可能包含针对边缘节点的详细分配
            
        Returns:
            (edge_updates, edge_gradients, edge_data_sizes, edge_losses) 元组
        """
        edge_updates, edge_gradients, edge_data_sizes, edge_losses = {}, {}, {}, {}
        
        if not client_edge_mapping:
            return edge_updates, edge_gradients, edge_data_sizes, edge_losses

        # 根据映射关系，将客户端按边缘节点分组
        edge_to_clients_map = {}
        for client_id, edge_id in client_edge_mapping.items():
            if edge_id not in edge_to_clients_map:
                edge_to_clients_map[edge_id] = []
            edge_to_clients_map[edge_id].append(client_id)
            
        with tqdm(total=len(edge_to_clients_map), desc="边缘训练", unit="edge", 
                  disable=self.disable_progress_bar, leave=False, position=1) as pbar:
            for edge_id, client_ids in edge_to_clients_map.items():
                edge_node = self.clients_manager.edge_nodes.get(edge_id)
                if not edge_node or not edge_node.available:
                    pbar.update(1)
                    continue

                pbar.set_description(f"边缘训练 @ {edge_id}")
                
                # 准备在该边缘节点上训练的客户端字典
                clients_for_edge = {cid: self.clients_manager.clients[cid] for cid in client_ids if cid in self.clients_manager.clients}
                
                # 获取该边缘节点的资源分配
                edge_resource = resource_allocation.get(edge_id, {})

                # 在边缘节点上执行训练
                # edge_train现在应该返回每个客户端的更新，而不是聚合后的更新
                client_updates, client_grads, client_losses, client_data_sizes = edge_node.edge_train(
                    clients_dict=clients_for_edge,
                    model=copy.deepcopy(self.model),
                    global_parameters=self.global_parameters,
                    lr=self.args.get('learning_rate'),
                    localEpoch=self.args.get('local_epochs', 3),
                    resource_allocation_map=edge_resource # 传递详细的资源映射
                )
                
                # 收集所有结果
                edge_updates.update(client_updates)
                edge_gradients.update(client_grads)
                edge_losses.update(client_losses)
                edge_data_sizes.update(client_data_sizes)
                
                pbar.update(1)
                
        return edge_updates, edge_gradients, edge_data_sizes, edge_losses
    
    def _get_all_data_sizes(self, client_ids):
        """获取所有客户端的数据大小
        
        Args:
            client_ids: 客户端ID列表
            
        Returns:
            数据大小字典 {client_id: data_size}
        """
        data_sizes = {}
        for client_id in client_ids:
            if client_id in self.clients_manager.clients:
                client = self.clients_manager.clients[client_id]
                data_sizes[client_id] = client.local_data_size
        return data_sizes
    
    def train_round(self, local_training_clients, client_edge_mapping, resource_allocation, aggregation_location, **kwargs):
        """执行一轮联邦学习训练 
        
        Args:
            local_training_clients: 需要本地训练的客户端列表
            client_edge_mapping: 客户端到边缘节点的映射 {client_id: edge_id}
            resource_allocation: 资源分配字典 {node_id: allocation_details}
            aggregation_location: 聚合位置 ('cloud'或边缘节点ID)
            
        Returns:
            (accuracy, global_test_loss, global_training_loss, total_delay, total_energy, total_cost) 元组
        """
        # 增加轮次计数并记录开始时间
        self.training_stats['round'] += 1
        current_round = self.training_stats['round']
        start_time = time.time()
        
        # 记录选择信息
        self.training_stats['selected_clients'].append(local_training_clients + list(client_edge_mapping.keys()))
        self.training_stats['aggregation_location'].append(aggregation_location)
        
        # ================== 1. 执行阶段: 启动训练 ==================
        progress_desc = f"轮次 {current_round} (Agg: {aggregation_location})"
        with tqdm(total=4, desc=progress_desc, unit="阶段", position=0, 
                 disable=self.disable_progress_bar) as pbar:
            
            pbar.set_description("阶段1: 本地训练")
            terminal_updates, terminal_grads, terminal_data_sizes, terminal_losses = self._train_terminal_clients(
                local_training_clients, resource_allocation
            )
            pbar.update(1)
            
            pbar.set_description("阶段2: 边缘训练")
            edge_updates, edge_grads, edge_data_sizes, edge_losses = self._train_edge_nodes(
                client_edge_mapping, resource_allocation
            )
            pbar.update(1)

            # ================== 2. 聚合阶段 ==================
            pbar.set_description(f"阶段3: {'云端' if aggregation_location == 'cloud' else '边缘'}聚合")

            # 合并所有训练结果
            all_updates = {**terminal_updates, **edge_updates}
            all_gradients = {**terminal_grads, **edge_grads}
            all_data_sizes = {**terminal_data_sizes, **edge_data_sizes}
            all_losses = {**terminal_losses, **edge_losses}
            
            # 计算加权全局训练损失 F(ω)
            total_training_data = sum(all_data_sizes.values())
            global_training_loss = 0.0
            if total_training_data > 0:
                for cid, local_loss in all_losses.items():
                    weight = all_data_sizes.get(cid, 0) / total_training_data
                    global_training_loss += weight * local_loss
            
            # 执行聚合
            aggregated_parameters, _, _ = self.aggregate(
                all_updates, 
                all_data_sizes, 
                aggregation_location, 
                all_gradients
            )
            pbar.update(1)
            
            # ================== 3. 更新与评估 ==================
            pbar.set_description("阶段4: 更新与评估")
            self.update_global_model(aggregated_parameters)
            accuracy = self.evaluate()
            pbar.update(1)
        
        # 计算总训练时间
        training_time = time.time() - start_time
        self.training_stats['training_time'].append(training_time)
        global_test_loss = self.training_stats['loss'][-1] if self.training_stats['loss'] else 0.0

        print(f"轮次 {current_round} 完成 => Acc: {accuracy:.4f}, Test Loss: {global_test_loss:.4f}, Train Loss: {global_training_loss:.4f}, Time: {training_time:.2f}s")
        
        # 从环境获取成本信息
        total_delay, total_energy, total_cost = 0, 0, 0
        if hasattr(self, 'env') and self.env is not None:
            total_delay = self.env.info.get("total_delay", 0)
            total_energy = self.env.info.get("total_energy", 0)
            total_cost = self.env.info.get("total_cost", 0)
        
        return accuracy, global_test_loss, global_training_loss, total_delay, total_energy, total_cost


# 增加FederatedServer的兼容性方法，以支持旧版API调用
def create_federated_server_from_args(args):
    """从参数字典创建联邦服务器实例
    
    Args:
        args: 参数字典，包含模型名称、学习率等配置
        
    Returns:
        FederatedServer实例
    """
    # 创建设备
    dev = torch.device("cuda")
    if 'device' in args:
        dev = args['device']
        
    # 创建模型
    model = create_model(args['model_name'], args)
    model = model.to(dev)
    
    # 创建客户端管理器
    from clients import ClientsGroup
    
    # 获取额外参数
    num_edges = args.get('num_edges', 2)
    non_iid_level = args.get('non_iid_level', 0)
    disable_progress_bar = args.get('disable_progress_bar', False)
    
    # 创建客户端管理器，支持新的参数
    clients_manager = ClientsGroup(
        dataset_name=args['model_name'], 
        is_iid=bool(args.get('IID', 0)), 
        num_clients=args.get('num_of_clients', 5), 
        device=dev,
        num_edges=num_edges,
        non_iid_level=non_iid_level,
        disable_progress_bar=disable_progress_bar
    )
    
    # 构建服务器参数
    server_args = {
        'device': dev,
        'learning_rate': args.get('learning_rate', 0.01),
        'model_name': args['model_name'],
        'save_path': args.get('save_path', './checkpoints'),
        'local_epochs': args.get('epoch', 3),
        'batch_size': args.get('batchsize', 64),
        'val_freq': args.get('val_freq', 1),
        'save_freq': args.get('save_freq', 5),
        'num_comm': args.get('num_comm', 20),
        'cfraction': args.get('cfraction', 0.9),
        'disable_progress_bar': disable_progress_bar
    }
    
    # 创建服务器实例
    server = FederatedServer(server_args, model, clients_manager)
    return server


# 如果直接运行此文件，执行简单测试
if __name__=="__main__":
    # 解析参数
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="FedAvg")
    parser.add_argument('-nc', '--num_of_clients', type=int, default=10, help='numer of the clients')
    parser.add_argument('-cf', '--cfraction', type=float, default=0.9, help='C fraction, 0 means 1 client, 1 means total clients')
    parser.add_argument('-E', '--epoch', type=int, default=3, help='local train epoch')
    parser.add_argument('-B', '--batchsize', type=int, default=10, help='local train batch size')
    parser.add_argument('-mn', '--model_name', type=str, default='mnist', help='the model to train')
    parser.add_argument('-vf', "--val_freq", type=int, default=5, help="model validation frequency(of communications)")
    parser.add_argument('-sf', '--save_freq', type=int, default=20, help='global model save frequency(of communication)')
    parser.add_argument('-ncomm', '--num_comm', type=int, default=20, help='number of communications')
    parser.add_argument('-sp', '--save_path', type=str, default='./checkpoints', help='the saving path of checkpoints')
    parser.add_argument('-iid', '--IID', type=int, default=0, help='the way to allocate data to clients')
    
    args = parser.parse_args()
    args = args.__dict__
    
    # 创建保存路径
    if not os.path.isdir(args['save_path']):
        os.makedirs(args['save_path'], exist_ok=True)
    
    # 使用服务器进行训练
    server = create_federated_server_from_args(args)
    
    # 简单测试
    print(f"服务器初始化完成，模型类型: {args['model_name']}")
    print(f"全局模型参数数量: {sum(p.numel() for p in server.model.parameters())}")
    print("准备开始训练...")
    
    # 执行一轮训练测试
    clients_states = {f'client{i}': {'available': True, 'is_edge': False} for i in range(5)}
    selected_clients = server.select_clients(clients_states, 3)
    print(f"选择的客户端: {selected_clients}")
    
    print("测试完成")
