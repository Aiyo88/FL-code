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

# 导入配置参数
from config import *

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
        self.convergence_deltas = []
        self.patience_counter = 0
        self.last_model_params = None # ★★★ 新增：存储上一轮的模型参数 ★★★
        
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
        # 保存上一轮有效结果
        self.last_valid_accuracy = 0.0
        self.last_valid_test_loss = float('inf')
        self.last_valid_training_loss = float('inf')

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
    def _weighted_average_aggregate(self, updates_dict, data_sizes, gradients_dict=None):
        """
        核心的加权平均聚合逻辑，可被云和边缘共用。
        """
        if not updates_dict:
            return self.global_parameters, None
        
        total_data = sum(data_sizes.values())
        if total_data == 0:
            return self.global_parameters, None
            
        # 聚合参数
        aggregated_parameters = {k: torch.zeros_like(v) for k, v in self.global_parameters.items()}
        for client_id, parameters in updates_dict.items():
            weight = data_sizes[client_id] / total_data
            for key, value in parameters.items():
                if key in aggregated_parameters:
                    aggregated_parameters[key] += weight * value
        
        # 聚合梯度（如果提供）
        aggregated_gradient = None
        if gradients_dict:
            # 从第一个梯度字典中获取key来初始化容器
            grad_keys = gradients_dict[list(gradients_dict.keys())[0]].keys()
            aggregated_gradient = {k: torch.zeros_like(v) for k, v in self.global_parameters.items() if k in grad_keys}
            for client_id, gradients in gradients_dict.items():
                weight = data_sizes[client_id] / total_data
                for key, value in gradients.items():
                    if key in aggregated_gradient:
                        aggregated_gradient[key] += weight * value

        return aggregated_parameters, aggregated_gradient
        
    def edge_aggregate(self, updates_dict, data_sizes, edge_id, gradients_dict=None):
        """在边缘节点上聚合模型参数和梯度"""
        start_time = time.time()
        
        aggregated_parameters, aggregated_gradient = self._weighted_average_aggregate(
            updates_dict, data_sizes, gradients_dict
        )
        
        # 模拟边缘节点的计算延迟
        if hasattr(self.clients_manager, 'edge_nodes') and edge_id in self.clients_manager.edge_nodes:
            edge_node = self.clients_manager.edge_nodes[edge_id]
            compute_delay = 1.0 / edge_node.compute_capability if hasattr(edge_node, 'compute_capability') else 1.0
            time.sleep(compute_delay * 0.01)
        
        aggregation_time = time.time() - start_time
        return aggregated_parameters, aggregated_gradient, aggregation_time
    
    def cloud_aggregate(self, updates_dict, data_sizes, gradients_dict=None):
        """在云服务器上聚合模型参数和梯度"""
        start_time = time.time()

        aggregated_parameters, aggregated_gradient = self._weighted_average_aggregate(
            updates_dict, data_sizes, gradients_dict
        )

        # 检查收敛性（云端特有逻辑）
        self._check_convergence(aggregated_parameters)
        
        aggregation_time = time.time() - start_time
        return aggregated_parameters, aggregated_gradient, aggregation_time
    
    def _check_convergence(self, new_params):
        """检查模型是否满足 ||ω^t - ω*|| ≤ ε
        
        计算当前参数与最优参数之间的欧氏距离，判断是否收敛
        
        Args:
            new_params: 新的模型参数
            
        Returns:
            is_converged: 布尔值，表示是否收敛
        """
        if self.best_params is None:
            self.best_params = copy.deepcopy(new_params)
            self.convergence_deltas.append(float('inf'))
            return False
        
        # 计算参数变化
        param_delta = 0.0
        for name in new_params:
            if name in self.best_params:
                param_delta += torch.norm(new_params[name] - self.best_params[name]).item()
        
        self.convergence_deltas.append(param_delta)

        # 更新最优参数
        if param_delta < self.best_delta:
            self.best_params = copy.deepcopy(new_params)
            self.best_delta = param_delta
        
        # 检查收敛耐心
        if param_delta < CONVERGENCE_EPSILON:
            self.patience_counter += 1
        else:
            self.patience_counter = 0 # 重置计数器

        if self.patience_counter >= CONVERGENCE_PATIENCE:
            print(f"\n[Convergence] 模型已收敛，连续 {self.patience_counter} 轮参数变化量 < {CONVERGENCE_EPSILON}")
            return True
            
        return False
    
    def aggregate(self, updates_dict, data_sizes, aggregation_location="cloud", gradients_dict=None):
        """根据指定位置进行聚合（支持ATAFL方案）
        
        Args:
            updates_dict: 客户端模型参数字典 {client_id: parameters}
            data_sizes: 客户端数据量字典 {client_id: data_size}
            aggregation_location: 聚合位置（"cloud"或边缘节点ID）
            gradients_dict: 客户端梯度字典 {client_id: gradient}
            
        Returns:
            (aggregated_parameters, aggregated_gradient, aggregation_time, is_converged) 元组
        """
        if aggregation_location == "cloud":
            aggregated_parameters, aggregated_gradient, aggregation_time = self.cloud_aggregate(updates_dict, data_sizes, gradients_dict)
            is_converged = self._check_convergence(aggregated_parameters)
            return aggregated_parameters, aggregated_gradient, aggregation_time, is_converged
        else:
            # 边缘聚合不检查收敛性
            aggregated_parameters, aggregated_gradient, aggregation_time = self.edge_aggregate(updates_dict, data_sizes, aggregation_location, gradients_dict)
            # 在边缘聚合后，也需要更新模型参数以计算变化量
            self.update_global_model(aggregated_parameters)
            return aggregated_parameters, aggregated_gradient, aggregation_time, False
    
    def update_global_model(self, aggregated_parameters):
        """更新全局模型参数
        
        根据聚合节点计算的聚合参数更新全局模型
        公式(5): ω^t = ∑_{i=1}^{I(t)} |D_i|/|D_{I(t)}| · ω_i^t
        """
        # ★★★ 新增：在更新前计算模型参数变化量 ★★★
        if self.last_model_params is not None:
            param_delta = 0.0
            for name in aggregated_parameters:
                if name in self.last_model_params:
                    param_delta += torch.norm(aggregated_parameters[name] - self.last_model_params[name]).item()
            self.best_delta = param_delta # 使用 best_delta 存储这个值
        
        # 存储当前参数以备下一轮使用
        self.last_model_params = copy.deepcopy(aggregated_parameters)

        # 更新全局参数 ω^t
        self.global_parameters = aggregated_parameters
        
        # 将参数加载到模型中
        self.model.load_state_dict(self.global_parameters, strict=True)
        
        # 更新训练统计
        self.training_stats['rounds_completed'] += 1
    
    def evaluate(self, test_loader):
        """
        在测试集上评估当前全局模型的性能。
        
        Args:
            test_loader: 用于评估的数据加载器
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0.0
        total_samples = 0.0
        
        if test_loader is None:
            raise ValueError("evaluate 方法需要一个有效的 test_loader")

        # 修正：使用 NLLLoss 而不是 CrossEntropyLoss，因为模型输出是 log_softmax
        loss_fn = nn.NLLLoss()

        with torch.no_grad():
            for _, (data, labels) in enumerate(test_loader):
                data, labels = data.to(self.dev), labels.to(self.dev)
                
                outputs = self.model(data)
                batch_loss = loss_fn(outputs, labels)
                total_loss += batch_loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total_samples += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_loss = total_loss / len(test_loader) if len(test_loader) > 0 else 0
        accuracy = correct / total_samples if total_samples > 0 else 0
        
        self.training_stats['accuracy'].append(accuracy)
        self.training_stats['loss'].append(avg_loss)
        
        return accuracy, avg_loss

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

    def _orchestrate_training(self, selected_nodes, client_edge_mapping, resource_allocation):
        """
        统一的训练协调函数。
        根据映射关系，自动调用 Client.train 或 EdgeNode.train。
        """
        updates, grads, data_sizes, losses, train_times = {}, {}, {}, {}, {}
        
        edge_to_clients_map = {}
        if client_edge_mapping:
            for client_id, edge_id in client_edge_mapping.items():
                if edge_id not in edge_to_clients_map:
                    edge_to_clients_map[edge_id] = []
                edge_to_clients_map[edge_id].append(client_id)
        
        # 计算总体训练损失
        total_samples = 0
        total_loss = 0.0
        
        # 调试信息
        print("\n======== 训练调试信息 ========")
        print(f"训练节点数量: {len(selected_nodes)}")
        print(f"客户端到边缘节点映射: {client_edge_mapping}")
        
        with tqdm(total=len(selected_nodes), desc='客户端训练', disable=self.disable_progress_bar) as pbar:
            for node_id in selected_nodes:
                node = self.clients_manager.clients.get(node_id)
                if not node or not node.available:
                    print(f"节点 {node_id} 不可用，跳过")
                    pbar.update(1)
                    continue

                res_alloc = resource_allocation.get(node_id, 1.0)
                
                if node.is_edge:
                    client_ids_on_this_edge = edge_to_clients_map.get(node_id, [])
                    for client_id in client_ids_on_this_edge:
                        client_on_behalf_of = self.clients_manager.clients.get(client_id)
                        if client_on_behalf_of:
                            params, gradients, loss, train_time, data_size = node.train(
                                model=copy.deepcopy(self.model),
                                global_parameters=self.global_parameters,
                                lr=self.args.get('lr', 0.01),
                                local_epochs=self.args.get('epochs', 3),
                                client_on_behalf_of=client_on_behalf_of
                            )
                            updates[client_id] = params
                            grads[client_id] = gradients
                            data_sizes[client_id] = data_size
                            losses[client_id] = loss
                            train_times[client_id] = train_time
                            
                            print(f"边缘节点 {node_id} 为客户端 {client_id} 训练, 损失: {loss:.6f}, 数据量: {data_size}")
                            
                            # 累加样本数和损失值
                            if data_size > 0:
                                total_samples += data_size
                                total_loss += loss * data_size  # 加权损失
                else:
                    params, gradients, loss, train_time, data_size = node.train(
                        model=copy.deepcopy(self.model),
                        global_parameters=self.global_parameters,
                        lr=self.args.get('lr', 0.01),
                        local_epochs=self.args.get('epochs', 3)
                    )
                    updates[node_id] = params
                    grads[node_id] = gradients
                    data_sizes[node_id] = data_size
                    losses[node_id] = loss
                    train_times[node_id] = train_time
                    
                    print(f"本地客户端 {node_id} 训练, 损失: {loss:.6f}, 数据量: {data_size}")
                    
                    # 累加样本数和损失值
                    if data_size > 0:
                        total_samples += data_size
                        total_loss += loss * data_size  # 加权损失

                pbar.update(1)
        
        # 确保在没有有效训练数据时不会导致除零错误
        if total_samples > 0:
            avg_loss = total_loss / total_samples
            print(f"计算平均损失: 总损失 {total_loss:.6f} / 总样本数 {total_samples} = {avg_loss:.6f}")
            
            # 确保所有客户端都有有效的损失值
            for client_id in losses:
                if losses[client_id] == 0 and data_sizes[client_id] > 0:
                    print(f"客户端 {client_id} 损失为0，使用平均损失 {avg_loss:.6f}")
                    losses[client_id] = avg_loss
        else:
            print("警告: 没有有效的训练数据，无法计算平均损失")
            
        print("数据大小统计:", data_sizes)
        print("损失统计:", losses)
        print("===========================\n")
                
        return updates, grads, data_sizes, losses, train_times

    def _train_terminal_clients(self, terminal_clients, resource_allocation):
        """[DEPRECATED] use _orchestrate_training instead"""
        return {}, {}, {}, {}

    def _train_edge_nodes(self, client_edge_mapping, resource_allocation):
        """[DEPRECATED] use _orchestrate_training instead"""
        return {}, {}, {}, {}

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
    
    def train_round(self, local_training_clients, client_edge_mapping, resource_allocation, aggregation_location, test_loader, **kwargs):
        """执行一轮联邦学习训练 
        
        Args:
            local_training_clients: 需要本地训练的客户端列表
            client_edge_mapping: 客户端到边缘节点的映射 {client_id: edge_id}
            resource_allocation: 资源分配字典 {node_id: allocation_details}
            aggregation_location: 聚合位置 ('cloud'或边缘节点ID)
            test_loader: 用于评估的测试数据加载器
            
        Returns:
            (accuracy, global_test_loss, global_training_loss, total_delay, total_energy, total_cost, is_converged) 元组
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
            # 1. 统一协调训练
            all_updates, all_gradients, all_data_sizes, all_losses, all_train_times = self._orchestrate_training(
                local_training_clients, client_edge_mapping, resource_allocation
            )
            pbar.update(1)
            
            pbar.set_description("阶段2: 边缘训练")
            # 2. 模型聚合
            aggregated_parameters, aggregated_gradient, _, is_converged = self.aggregate(
                all_updates, 
                all_data_sizes, 
                aggregation_location,
                gradients_dict=all_gradients
            )
            pbar.update(1)
            
            # ================== 3. 更新与评估 ==================
            pbar.set_description("阶段4: 更新与评估")
            self.update_global_model(aggregated_parameters)
            accuracy, global_test_loss = self.evaluate(test_loader)
            pbar.update(1)
        
        # 计算总训练时间
        training_time = time.time() - start_time
        self.training_stats['training_time'].append(training_time)

        # 计算加权平均的训练损失
        total_training_data = sum(all_data_sizes.values())
        global_training_loss = 0.0
        if total_training_data > 0:
            for cid, local_loss in all_losses.items():
                weight = all_data_sizes.get(cid, 0) / total_training_data
                global_training_loss += weight * local_loss

        # 更新上一轮的有效结果
        self.last_valid_accuracy = accuracy
        self.last_valid_test_loss = global_test_loss
        self.last_valid_training_loss = global_training_loss

        print(f"轮次 {current_round} 完成 => Acc: {accuracy:.4f}, Test Loss: {global_test_loss:.4f}, Train Loss: {global_training_loss:.4f}, Time: {training_time:.2f}s")
        
        # 从环境获取成本信息
        total_delay, total_energy, total_cost = 0, 0, 0
        if hasattr(self, 'env') and self.env is not None:
            total_delay = self.env.info.get("total_delay", 0)
            total_energy = self.env.info.get("total_energy", 0)
            total_cost = self.env.info.get("total_cost", 0)
        
        return accuracy, global_test_loss, global_training_loss, total_delay, total_energy, total_cost, is_converged

    def reset_for_new_episode(self):
        """为新的DRL Episode重置服务器状态"""
        self.best_params = None
        self.best_delta = float('inf')
        self.convergence_deltas = []
        self.patience_counter = 0
        self.last_model_params = None # ★★★ 新增：重置 ★★★
        self.model = create_model(self.args['model_name']).to(self.dev)
        self.global_parameters = self.model.state_dict()
        print("服务器状态已为新Episode重置。")

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
        non_iid_level=non_iid_level
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
