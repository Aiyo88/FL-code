import torch
import numpy as np
import os
import argparse
import time
import sys
from tqdm import tqdm
import tqdm.notebook as tqdm_notebook

# 导入自定义模块
from utils.data_utils import get_dataset, get_client_dataset
from models.models import create_model
from Env import Env
from clients import ClientsGroup
from server import FederatedServer
from drl_adapters import create_drl_agent
from plot.plot_award import log_training_result, plot_param_comparison, plot_all_params

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="联邦学习训练")
    
    # 系统参数
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--device', type=str, default='cpu', help='使用设备 (cpu/cuda)')
    parser.add_argument('--save_path', type=str, default='./checkpoints', help='模型保存路径')
    parser.add_argument('--data_path', type=str, default='./data', help='数据集路径')
    parser.add_argument('--disable_progress_bar', type=int, default=0, help='禁用进度条 (0/1)')
    
    # 联邦学习参数
    parser.add_argument('--dataset', type=str, default='mnist', help='数据集 (mnist/cifar10)')
    parser.add_argument('--model', type=str, default='mnist', help='模型类型 (cnn)')
    parser.add_argument('--num_clients', type=int, default=10, help='客户端数量')
    parser.add_argument('--num_edges', type=int, default=3, help='边缘节点数量')
    parser.add_argument('--iid', type=int, default=0, help='是否为IID数据分布 (0/1)')
    parser.add_argument('--non_iid_level', type=int, default=1, help='非IID数据分布级别 (1/2/3)')
    parser.add_argument('--epochs', type=int, default=5, help='本地训练轮数')
    parser.add_argument('--batch_size', type=int, default=64, help='本地批次大小')
    parser.add_argument('--lr', type=float, default=0.01, help='学习率')
    parser.add_argument('--num_rounds', type=int, default=100, help='联邦学习轮数')
    
    parser.add_argument('--use_drl', type=int, default=1, help='是否使用DRL进行决策 (0/1)')
    parser.add_argument('--drl_train', type=int, default=1, help='是否训练DRL智能体 (0/1)')
    parser.add_argument('--drl_algo', type=str, default='pdqn', help='DRL算法 (a3c/pdqn/ddpg/ppo/sac/td3)')
    parser.add_argument('--drl_load', type=str, default=None, help='加载DRL模型路径')
    
    # 环境参数
    parser.add_argument('--C', type=float, default=10, help='环境参数C')
    parser.add_argument('--gama', type=float, default=0.01, help='环境参数gama')
    parser.add_argument('--delta', type=float, default=1.0, help='环境参数delta')
    
    # 只保留是否需要绘图的开关
    parser.add_argument('--plot', type=int, default=0, help='是否生成参数比较图 (0/1)')
    
    args = parser.parse_args()
    return args

def set_seed(seed):
    """设置随机种子"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def _parse_drl_decisions(action, env, server, clients_manager):
    """解析DRL决策并生成训练参数
    
    Args:
        action: DRL生成的动作向量，Tuple格式[离散动作, 连续动作]
        env: 环境实例
        server: 服务器实例
        clients_manager: 客户端管理器实例
        
    Returns:
        训练参数字典
    """
    # 初始化default_client为None，避免未定义错误
    default_client = None
    
    # 定义常用常量
    DEFAULT_F_L_MIN = 4e8    # 最小CPU频率 (0.4GHz)
    DEFAULT_F_L_MAX = 2.9e9  # 最大CPU频率 (2.9GHz)
    DEFAULT_F_E_MIN = 2.9e9  # 最小CPU频率 (2.9GHz)
    DEFAULT_F_E_MAX = 4.3e9  # 最大CPU频率 (4.3GHz)
    EPSILON = 1e-6  # 浮点比较容差
    
    # 获取可用的终端设备和边缘节点
    available_clients = [
        cid for cid in clients_manager.clients 
        if clients_manager.clients[cid].available and cid.startswith("client")
    ]
    
    available_edges = [
        cid for cid in clients_manager.clients
        if clients_manager.clients[cid].available and cid.startswith("edge")
    ]
    
    discrete_dim = env.train_dim + env.agg_dim
    # 将元组动作转换为NumPy数组
    if isinstance(action, tuple) and len(action) == 2:
        # 如果动作是(离散部分, 连续部分)的元组格式
        discrete_action = action[0]
        continuous_action = action[1]
    else:
        # 如果动作是单一数组
        discrete_action = np.array(action[:discrete_dim])
        continuous_action = np.array(action[discrete_dim:discrete_dim + env.res_dim])
    
    # 1. 解析动作
    # 1.1 解析训练决策
    train_local_offset = 0
    train_local_decisions = discrete_action[train_local_offset:min(env.N, len(discrete_action))]  # x_i^l(t)
    
    # 1.2 解析边缘训练决策 - 使用动态映射而非reshape
    edge_train_offset = env.N
    edge_train_decisions = discrete_action[edge_train_offset:edge_train_offset + env.N * env.M]
    
    # 动态创建边缘训练决策映射
    edge_train_mapping = {}
    for i, client_id in enumerate(available_clients):
        if i >= env.N:  # 超出设备数量范围
            break
        for j, edge_id in enumerate(available_edges):
            idx = i * len(available_edges) + j
            if idx < len(edge_train_decisions) and edge_train_decisions[idx] > 0.5:
                edge_train_mapping[client_id] = edge_id
    
    # 1.3 解析聚合决策
    edge_agg_offset = edge_train_offset + env.N * env.M
    edge_agg_decisions = discrete_action[edge_agg_offset:edge_agg_offset + env.N * env.M]
    
    # 使用类似的动态映射方式处理聚合决策
    edge_agg_mapping = {}
    edge_agg_counts = {edge_id: 0 for edge_id in available_edges}
    
    for i, client_id in enumerate(available_clients):
        if i >= env.N:  # 超出设备数量范围
            break
        for j, edge_id in enumerate(available_edges):
            idx = i * len(available_edges) + j
            if idx < len(edge_agg_decisions) and edge_agg_decisions[idx] > 0.5:
                edge_agg_mapping[client_id] = edge_id
                edge_agg_counts[edge_id] += 1
    
    cloud_agg_offset = edge_agg_offset + env.N * env.M
    cloud_agg_decisions = discrete_action[cloud_agg_offset:cloud_agg_offset + env.N]
    
    # 1.4 解析资源分配决策
    res_alloc_flat = continuous_action
    
    # 2. 处理训练决策
    drl_train_decisions = []  # 最终的训练决策列表 (1=本地训练，0=边缘训练)
    client_edge_mapping = {}  # 客户端到边缘节点的映射
    selected_edges_set = set()  # 选择的边缘节点集合
    
    # 直接选择所有参与训练的节点
    selected_nodes = []
    
    # 为每个可用的客户端分配训练决策
    for i, client_id in enumerate(available_clients):
        if i >= env.N:  # 确保不超过设备数量限制
            break
        
        # 决定该客户端是在本地训练还是边缘训练
        if i < len(train_local_decisions) and train_local_decisions[i] > 0.5:
            # 本地训练
            drl_train_decisions.append(1)  # 1=本地训练
            selected_nodes.append(client_id)
        else:
            # 边缘训练 - 检查是否有边缘节点映射
            if client_id in edge_train_mapping:
                drl_train_decisions.append(0)  # 0=边缘训练
                selected_nodes.append(client_id)
                
                edge_id = edge_train_mapping[client_id]
                client_edge_mapping[client_id] = edge_id
                selected_edges_set.add(edge_id)
            else:
                # 没有明确的边缘节点映射，选择第一个可用的
                if available_edges:
                    drl_train_decisions.append(0)  # 0=边缘训练
                    selected_nodes.append(client_id)
                    
                    edge_id = available_edges[0]
                    client_edge_mapping[client_id] = edge_id
                    selected_edges_set.add(edge_id)
                else:
                    # 没有可用的边缘节点，默认为本地训练
                    drl_train_decisions.append(1)  # 1=本地训练
                    selected_nodes.append(client_id)
    
    # 3. 确定聚合位置
    # 计算云聚合的票数
    cloud_votes = sum(1 for decision in cloud_agg_decisions if decision > 0.5)
    
    # 找出票数最高的边缘节点
    max_edge_votes = 0
    max_edge_id = None
    
    for edge_id, votes in edge_agg_counts.items():
        if votes > max_edge_votes:
            max_edge_votes = votes
            max_edge_id = edge_id
    
    # 确定最终聚合位置
    if cloud_votes > max_edge_votes or not max_edge_id:
        aggregation_location = "cloud"
    else:
        aggregation_location = max_edge_id
    
    # 4. 处理资源分配
    client_resources = {}
    
    # 为终端设备分配资源
    for i, client_id in enumerate(available_clients):
        if i >= env.N:
            continue
            
            # 本地训练设备分配最大资源
            if i < len(drl_train_decisions) and drl_train_decisions[i] == 1:
                f_min = env.f_l_min if hasattr(env, 'f_l_min') else DEFAULT_F_L_MIN
                f_max = env.F_l[i] if hasattr(env, 'F_l') and i < len(env.F_l) else DEFAULT_F_L_MAX
                client_resources[client_id] = f_max  # 本地训练使用最大资源
            else:
                # 边缘训练设备 - 获取对应的边缘节点
                edge_id = client_edge_mapping.get(client_id)
                if edge_id and edge_id in available_edges:
                    edge_idx = int(edge_id.split('_')[1])
                    
                    # 查找对应的连续动作索引
                    if edge_idx < env.M:
                        res_idx = i * env.M + edge_idx
                        if res_idx < len(res_alloc_flat):
                            resource_value = res_alloc_flat[res_idx]
                            
                                    # 计算实际资源值
                            f_min = env.f_e_min if hasattr(env, 'f_e_min') else DEFAULT_F_E_MIN
                            f_max = env.F_e[edge_idx] if hasattr(env, 'F_e') and edge_idx < len(env.F_e) else DEFAULT_F_E_MAX
                            client_resources[client_id] = f_min + resource_value * (f_max - f_min)
                        else:
                            # 索引超出范围，使用默认值
                            client_resources[client_id] = DEFAULT_F_E_MIN
                    else:
                        # 边缘节点索引超出范围，使用默认值
                        client_resources[client_id] = DEFAULT_F_E_MIN
        else:
                # 没有找到对应的边缘节点，使用默认值
                client_resources[client_id] = DEFAULT_F_L_MIN
    
    # 为边缘节点分配资源
    for edge_id in selected_edges_set:
        if edge_id in available_edges:
            edge_idx = int(edge_id.split('_')[1])
            
            # 收集使用此边缘节点的客户端资源分配值
            client_resources_on_edge = []
            for client_id, mapped_edge in client_edge_mapping.items():
                if mapped_edge == edge_id:
                    client_idx = available_clients.index(client_id) if client_id in available_clients else -1
                    if client_idx >= 0 and client_idx < env.N and edge_idx < env.M:
                        res_idx = client_idx * env.M + edge_idx
                        if res_idx < len(res_alloc_flat):
                            client_resources_on_edge.append(res_alloc_flat[res_idx])
            
            # 计算平均资源值
            if client_resources_on_edge:
                avg_resource = np.mean(client_resources_on_edge)
            else:
                avg_resource = 0.5  # 默认值
                
            # 设置边缘节点资源
            f_min = env.f_e_min if hasattr(env, 'f_e_min') else DEFAULT_F_E_MIN
            f_max = env.F_e[edge_idx] if hasattr(env, 'F_e') and edge_idx < len(env.F_e) else DEFAULT_F_E_MAX
            client_resources[edge_id] = f_min + avg_resource * (f_max - f_min)
    
    # 5. 添加所有选定的边缘节点到训练节点列表
    selected_nodes.extend(selected_edges_set)
    
    # 确保至少有一个终端设备被选中
    if not any(cid for cid in selected_nodes if cid.startswith("client")):
        if available_clients:
            default_client = available_clients[0]
            selected_nodes.append(default_client)
            # 默认为本地训练
            for i, client_id in enumerate(available_clients):
                if client_id == default_client and i < len(drl_train_decisions):
                    drl_train_decisions[i] = 1  # 设置为本地训练
                    print(f"警告: 没有终端设备被选择，添加默认设备 {default_client} (本地训练)")
    
    # 将客户端到边缘节点的映射信息保存到服务器中
    if hasattr(server, 'client_edge_mapping'):
        server.client_edge_mapping = client_edge_mapping
    
    # 6. 构建训练参数
    training_args = {
        'selected_nodes': selected_nodes,
        'resource_allocation': client_resources,
        'aggregation_location': aggregation_location,
        'drl_train_decisions': drl_train_decisions,
        'atafl_enabled': True  # 启用ATAFL参数更新机制
    }
    
    return training_args

def _run_standard_federated_round(server, clients_manager, args, round_idx=0, drl_agent=None, env=None):
    """执行标准联邦学习训练轮次
    
    Args:
        server: 服务器实例
        clients_manager: 客户端管理器实例
        args: 参数
        round_idx: 当前轮次索引，默认为0
        drl_agent: DRL智能体实例，默认为None
        env: 环境实例，默认为None
        
    Returns:
        训练准确率
    """
    # 获取客户端状态
    clients_states = {}
    for cid, client in clients_manager.clients.items():
        clients_states[cid] = {'available': client.available, 'is_edge': client.is_edge}
    
    # 第一轮使用默认选择，后续轮次使用DRL决策
    if round_idx == 0:
        # 选择所有可用的终端设备作为客户端
        selected_clients = [cid for cid, state in clients_states.items() 
                          if state['available'] and not state['is_edge'] and cid.startswith('client')]
        
        # 选择一个边缘节点作为聚合节点（如果有的话）
        edge_nodes = [cid for cid, state in clients_states.items() 
                    if state['available'] and state['is_edge']]
        aggregation_edge = edge_nodes[0] if edge_nodes else None
        aggregation_location = aggregation_edge if aggregation_edge else "cloud"
        
        # 创建资源分配字典，为每个终端设备分配最大资源
        resource_allocation = {cid: 1.0 for cid in selected_clients}
        if aggregation_edge:
            resource_allocation[aggregation_edge] = 1.0
            selected_clients.append(aggregation_edge)
        
        # 在第一轮中，所有终端设备都在本地训练
        drl_train_decisions = [0] * len(selected_clients)  # 0=本地训练
        
        print(f"初始轮次: 选择了 {len(selected_clients) - (1 if aggregation_edge else 0)} 个终端设备")
        print(f"  聚合位置: {aggregation_edge if aggregation_edge else '云服务器'}")
        print(f"  训练位置: 所有终端设备在本地训练")
    else:
        # 后续轮次使用DRL决策
        if drl_agent is not None and env is not None:
            # 同步客户端状态到环境
            clients_manager.update_client_states(env)
            
            # 获取当前环境状态
            state = env._get_state()
            
            # 使用DRL智能体生成动作
            action = drl_agent.get_action(state)
            
            # 解析DRL决策
            training_args = _parse_drl_decisions(action, env, server, clients_manager)
            
            print(f"轮次 {round_idx+1}: 使用DRL决策生成训练参数")
            
            # 返回训练参数，而不是在函数内部执行训练
            return training_args
        else:
            # 如果没有DRL智能体或环境，使用默认选择
            print(f"警告: 轮次 {round_idx+1} 没有提供DRL智能体或环境，使用默认选择")
            
            # 选择所有可用的终端设备作为客户端
            selected_clients = [cid for cid, state in clients_states.items() 
                              if state['available'] and not state['is_edge'] and cid.startswith('client')]
            
            # 选择一个边缘节点作为聚合节点
            edge_nodes = [cid for cid, state in clients_states.items() 
                        if state['available'] and state['is_edge']]
            aggregation_edge = edge_nodes[0] if edge_nodes else None
            aggregation_location = aggregation_edge if aggregation_edge else "cloud"
            
            # 创建资源分配字典
            resource_allocation = {cid: 1.0 for cid in selected_clients}
            if aggregation_edge:
                resource_allocation[aggregation_edge] = 1.0
                selected_clients.append(aggregation_edge)
            
            # 所有终端设备都在本地训练
            drl_train_decisions = [0] * len(selected_clients)  # 0=本地训练
    
    # 创建训练参数
    training_args = {
        'selected_nodes': selected_clients,
        'resource_allocation': resource_allocation,
        'aggregation_location': aggregation_location,
        'drl_train_decisions': drl_train_decisions,
        'atafl_enabled': True  # 启用ATAFL参数更新机制
    }
    
    # 返回训练参数，由调用者执行训练
    return training_args

def _finalize_training(server, global_model, test_loader, args, training_stats, drl_agent):
    """完成训练并进行最终评估
    
    Args:
        server: 服务器实例
        global_model: 全局模型
        test_loader: 测试数据加载器
        args: 参数
        training_stats: 训练统计信息
        drl_agent: DRL智能体实例
        
    Returns:
        最终准确率
    """
    print("训练完成!")
    print("=" * 50)
    
    # 最终评估 - 使用服务器的evaluate方法
    acc = server.evaluate(test_loader)
    print(f"最终准确率: {acc:.4f}")
    
    # 保存最终模型
    server.save_model(add_round=False)
    if args.drl_train and drl_agent is not None and hasattr(drl_agent, 'save_model'):
        model_path = os.path.join(args.save_path, f"drl_{args.drl_algo}_final")
        drl_agent.save_model(model_path)
    
    # 打印最终统计
    print("\n训练统计:")
    print(f"总轮次: {len(training_stats['accuracy'])}")
    print(f"最高准确率: {max(training_stats['accuracy']):.4f}")
    print(f"平均准确率: {np.mean(training_stats['accuracy']):.4f}")
    print(f"平均损失: {np.mean(training_stats['loss']):.4f}")
    print(f"平均奖励: {np.mean(training_stats['rewards']):.4f}")
    print(f"平均训练时间: {np.mean(training_stats['times']):.2f}s")
    
    return acc

def main():
    """主程序入口"""
    # 解析参数
    args = parse_args()

    # 设置随机种子
    set_seed(args.seed)
    
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else "cpu")
    args.device = device
    
    # 创建保存路径
    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path, exist_ok=True)
    
    # 设置进度条显示
    tqdm_cls = tqdm_notebook if 'ipykernel' in sys.modules else tqdm
    disable_progress_bar = bool(args.disable_progress_bar)
    
    print("=" * 50)
    print("初始化系统...")
    
    # 1. 创建环境
    env = Env(C=args.C, gama=args.gama, delta=args.delta)
    env.N = args.num_clients  # 设置终端设备数量
    env.M = args.num_edges    # 设置边缘节点数量
    print(f"创建环境: 终端设备={env.N}, 边缘节点={env.M}")
    
    # 在加载数据集部分之前添加：
    args.dataset = args.dataset.lower()
    supported_datasets = ['mnist', 'cifar10']
    if args.dataset not in supported_datasets:
        raise ValueError(f"Supported datasets: {supported_datasets}, got {args.dataset}")

    if not os.path.exists(args.data_path):
        os.makedirs(args.data_path)
        print(f"Created data directory: {args.data_path}")
        
    # 2. 环境初始化数据集
    print("环境初始化数据集...")
    env_datasets, test_loader = env.initialize_dataset(
        args.dataset,
        bool(args.iid),
        args.non_iid_level,
        args.data_path
    )
    print(f"数据集初始化完成: '{args.dataset}', {'IID' if args.iid else f'非IID-L{args.non_iid_level}'}")
    
    # 3. 初始化模型
    print("初始化模型...")
    global_model = create_model(args.model)
    global_model = global_model.to(device)
    
    # 4. 创建客户端管理器 - 使用环境预分配的数据集
    clients_manager = ClientsGroup(
        args.dataset, 
        bool(args.iid), 
        args.num_clients,
        args.device,
        num_edges=args.num_edges,
        non_iid_level=args.non_iid_level,
        disable_progress_bar=disable_progress_bar,
        env_datasets=env_datasets  # 传入环境预分配的数据集
    )
    print(f"创建客户端管理器: 客户端数量={args.num_clients}, 边缘节点数量={args.num_edges}, {'IID' if args.iid else f'非IID-L{args.non_iid_level}'} 数据分布")
    
    # 5. 创建服务器
    server_args = {
        'device': device,
        'learning_rate': args.lr,
        'model_name': f"{args.dataset}_{args.model}",
        'save_path': args.save_path,
        'local_epochs': args.epochs,
        'batch_size': args.batch_size,
        'val_freq': 1,  # 每轮评估
        'save_freq': 5,  # 每5轮保存
        'disable_progress_bar': disable_progress_bar  # 传递进度条禁用参数
    }
    server = FederatedServer(server_args, global_model, clients_manager)
    print("创建服务器")
    
    # 6. 为环境设置客户端和服务器
    env.set_clients_and_server(clients_manager, server)
    
    # 7. 创建DRL智能体
    drl_agent = create_drl_agent(args.drl_algo, env)
    if args.drl_load:
        drl_agent.load_model(args.drl_load)
        print(f"加载DRL模型: {args.drl_load}")
    print(f"创建DRL智能体: {args.drl_algo.upper()}")
    
    # 8. 同步客户端状态与环境
    clients_manager.update_client_states(env)
    print("同步客户端状态与环境")
    
    print("系统初始化完成")
    print("=" * 50)
    
    # 主训练循环
    print(f"开始训练 {args.num_rounds} 轮...")
    training_stats = {'accuracy': [], 'rewards': [], 'times': [], 'loss': []}
    
    # 计算DRL Episode和FL轮次的分配
    fl_rounds_per_episode = 10  # 每个DRL Episode包含的FL轮次数量
    total_episodes = (args.num_rounds + fl_rounds_per_episode - 1) // fl_rounds_per_episode  # 向上取整
    
    # 设置环境中的FL轮次参数
    env.fl_rounds_per_episode = fl_rounds_per_episode
    print(f"设置每个DRL Episode包含 {fl_rounds_per_episode} 个FL轮次，总共需要 {total_episodes} 个DRL Episodes")
    
    # 使用tqdm创建训练轮次的进度条
    with tqdm_cls(total=args.num_rounds, desc="训练进度", unit="round", disable=disable_progress_bar) as rounds_pbar:
        episode_idx = 0
        round_idx = 0
        
        while round_idx < args.num_rounds:
            # 开始新的DRL Episode
            episode_start_time = time.time()
            episode_idx += 1
            print(f"\n===== 开始DRL Episode {episode_idx}/{total_episodes} =====")
            
            # 重置环境以开始新的Episode
            state = env.reset()
            episode_done = False
            episode_rewards = []
            episode_fl_rounds = 0
            
            # 在一个DRL Episode内执行多个FL轮次
            while not episode_done and round_idx < args.num_rounds:
                round_start_time = time.time()
                rounds_pbar.set_description(f"DRL Episode {episode_idx}, FL轮次 {episode_fl_rounds+1}/{fl_rounds_per_episode}")
                
                # 使用DRL智能体进行决策
                if args.use_drl:
                    print(f"FL轮次 {round_idx+1}: 使用DRL智能体进行决策...")
                    action = drl_agent.get_action(state)  # 使用get_action方法
                    
                    # 解析DRL决策
                    training_args = _parse_drl_decisions(action, env, server, clients_manager)
                else:
                    # 使用标准联邦学习训练，传递DRL智能体和环境参数
                    training_args = _run_standard_federated_round(server, clients_manager, args, round_idx, drl_agent, env)
                    
                # 执行联邦学习训练
                accuracy = server.train_round(**training_args)
                
                # 打印详细的训练参数信息
                print(f"\n训练参数详情:")
                print(f"  选择的节点: {training_args['selected_nodes']}")
                print(f"  聚合位置: {training_args['aggregation_location']}")
                print(f"  本地训练决策: {training_args['drl_train_decisions']}")
                
                if hasattr(server, 'client_edge_mapping') and server.client_edge_mapping:
                    print(f"  客户端到边缘节点的映射:")
                    for client_id, edge_id in server.client_edge_mapping.items():
                        print(f"    {client_id} -> {edge_id}")
                
                # 环境步进 - 注意现在step返回的是4个值
                try:
                    next_state, reward, episode_done, info = env.step(action)
                    
                    # 记录和输出额外信息
                    if 'error' in info:
                        print(f"警告: {info['error']}")
                        # 不要中断训练，继续处理
                    
                    # 输出详细的训练决策信息
                    local_count = sum(1 for d in training_args['drl_train_decisions'] if d == 1)
                    edge_count = sum(1 for d in training_args['drl_train_decisions'] if d == 0)
                    
                    # 更新客户端选择信息，确保至少有一个设备被选中
                    if len(training_args['selected_nodes']) == 0:
                        print("警告: 没有设备被选择，将使用默认设备")
                        # 添加默认设备
                        client_ids = [cid for cid in clients_manager.clients.keys() 
                                     if not cid.startswith('edge_') and clients_manager.clients[cid].available]
                        if client_ids:
                            training_args['selected_nodes'] = [client_ids[0]]
                            print(f"  添加默认设备: {client_ids[0]}")
                    
                    print(f"  FL轮次 {round_idx+1} 训练情况:")
                    print(f"    总终端设备: {len(training_args['drl_train_decisions'])}个")
                    print(f"    本地训练: {local_count}个设备")
                    
                    # 显示卸载到各边缘节点的设备数量
                    if hasattr(server, 'client_edge_mapping') and server.client_edge_mapping:
                        edge_counts = {}
                        for client, edge in server.client_edge_mapping.items():
                            edge_counts[edge] = edge_counts.get(edge, 0) + 1
                        
                        for edge, count in edge_counts.items():
                            print(f"    卸载到{edge}: {count}个设备")
                    else:
                        print(f"    卸载到边缘节点: {edge_count}个设备")
                    
                    # 输出奖励和性能指标
                    print(f"  奖励: {reward:.4f}, 延迟: {info.get('avg_delay', 0.0):.4f}s, 能耗: {info.get('avg_energy', 0.0):.4f}J")
                    
                except ValueError as e:
                    print(f"环境步进出错: {e}，可能是环境返回值格式不正确")
                    # 兼容旧格式（如果step方法只返回3个值）
                    try:
                        results = env.step(action)
                        if len(results) == 3:
                            next_state, reward, episode_done = results
                            info = {}
                        else:
                            raise ValueError("环境step方法返回值格式不受支持")
                    except Exception as e2:
                        print(f"再次尝试失败: {e2}，使用默认值")
                        next_state = state
                        reward = 0.0
                        episode_done = False
                        info = {}
                
                # 记录奖励
                episode_rewards.append(reward)
                
                # 训练DRL智能体(仅在Episode结束时)
                if args.drl_train and episode_done:
                    if hasattr(env, 'get_episode_data'):
                        # 获取完整Episode数据进行训练
                        try:
                            episode_data = env.get_episode_data()
                            if episode_data:
                                print(f"使用完整Episode数据进行训练，包含 {len(episode_data.get('rewards', []))} 个FL轮次")
                                drl_agent.process_episode_data(episode_data)
                            else:
                                print("警告：无法获取Episode数据，使用最后一个状态转移进行训练")
                                drl_agent._update_agent(state, action, reward, next_state, episode_done)
                        except Exception as e:
                            print(f"获取Episode数据失败: {e}，使用最后一个状态转移进行训练")
                            drl_agent._update_agent(state, action, reward, next_state, episode_done)
                    else:
                        # 使用最后一个状态转移进行训练
                        drl_agent._update_agent(state, action, reward, next_state, episode_done)
                
                # 更新状态
                state = next_state
                
                # 保存模型和统计信息
                if (round_idx + 1) % 5 == 0 or round_idx == args.num_rounds - 1:
                    server.save_model()
                    if args.drl_train and hasattr(drl_agent, 'save_model'):
                        model_path = os.path.join(args.save_path, f"drl_{args.drl_algo}_round{round_idx+1}")
                        drl_agent.save_model(model_path)
                
                # 记录统计信息
                training_stats['accuracy'].append(accuracy)
                training_stats['loss'].append(server.training_stats['loss'][-1] if server.training_stats['loss'] else 0)
                training_stats['rewards'].append(reward)
                training_stats['times'].append(time.time() - round_start_time)
                
                # 记录每轮的奖励到对应参数的日志
                log_training_result(
                    "LR", args.lr, 
                    round_idx, 
                    reward,  # 当前轮次的奖励值
                    server.training_stats['loss'][-1] if server.training_stats['loss'] else 0
                )
                
                # 增加FL轮次计数
                round_idx += 1
                episode_fl_rounds += 1
                
                # 更新进度条信息
                round_time = time.time() - round_start_time
                rounds_pbar.set_postfix({
                    'Loss': f"{training_stats['loss'][-1]:.4f}", 
                    'Acc': f"{accuracy:.4f}", 
                    'Reward': f"{reward:.4f}",
                    'Episode': f"{episode_idx}/{total_episodes}",
                    'Time': f"{round_time:.2f}s"
                })
                rounds_pbar.update(1)
            
            # DRL Episode结束，打印统计信息
            episode_time = time.time() - episode_start_time
            episode_reward = sum(episode_rewards)
            episode_avg_reward = np.mean(episode_rewards)
            print(f"DRL Episode {episode_idx} 完成 - 共执行 {episode_fl_rounds} 个FL轮次")
            print(f"Episode统计: 总奖励={episode_reward:.4f}, 平均奖励={episode_avg_reward:.4f}, 用时={episode_time:.2f}s")
            
            # 手动重置环境为下一个Episode做准备
            if round_idx < args.num_rounds:
                # 确保在最后一轮后不执行reset
                try:
                    # 重置环境
                    state = env.reset()
                    print(f"为下一个DRL Episode重置环境，新的状态维度: {state.shape}")
                except Exception as e:
                    print(f"重置环境出错: {e}")
                    # 使用上一轮的next_state作为新的state
                    state = next_state
                    print(f"使用上一轮的next_state作为新的state，维度: {state.shape}")
            
            # 如果使用DRL，保存Episode级别的模型
            if args.drl_train and hasattr(drl_agent, 'save_model'):
                try:
                    model_path = os.path.join(args.save_path, f"drl_{args.drl_algo}_episode{episode_idx}")
                    drl_agent.save_model(model_path)
                    print(f"保存DRL模型到 {model_path}")
                except Exception as e:
                    print(f"保存模型出错: {e}")
                    
            # 打印分隔线，增强可读性
            print("=" * 50)
    
    # 训练结束，进行最终评估和统计
    return args, _finalize_training(server, global_model, test_loader, args, training_stats, drl_agent)

# 在训练结束后，如果需要绘图
if __name__ == "__main__":
    args, final_acc = main()
    
    # 如果需要绘图
    if args.plot:
        # 定义要绘制的参数及其值
        params_dict = {
            "LR": [0.001, 0.01, 0.0001],
            "BATCH_SIZE": [64, 128, 256],
            "GAMMA": [0.9, 0.95, 0.99],
            "MEMORY": [256, 512, 1024]
        }
        
        # 绘制单个参数比较图
        for param_name, param_values in params_dict.items():
            plot_param_comparison(param_name, param_values)
        
        # 绘制所有参数组合图
        plot_all_params(params_dict)