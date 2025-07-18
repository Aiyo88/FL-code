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
from plot.plot_award import log_training_result, plot_training_results, plot_all_logs
# 导入config模块
from config import MIN_ENERGY

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="联邦学习训练")
    
    # 系统参数
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--device', type=str, default='cuda', help='使用设备 (cpu/cuda)')
    parser.add_argument('--save_path', type=str, default='./checkpoints', help='模型保存路径')
    parser.add_argument('--data_path', type=str, default='./data', help='数据集路径')
    parser.add_argument('--disable_progress_bar', type=int, default=0, help='禁用进度条 (0/1)')
    
    # 联邦学习参数
    parser.add_argument('--dataset', type=str, default='mnist', help='数据集 (mnist/cifar10)')
    parser.add_argument('--model', type=str, default='mnist', help='模型类型 (cnn)')
    parser.add_argument('--num_clients', type=int, default=5, help='客户端数量')
    parser.add_argument('--num_edges', type=int, default=2, help='边缘节点数量')
    parser.add_argument('--iid', type=int, default=1, help='是否为IID数据分布 (0/1)')
    parser.add_argument('--non_iid_level', type=int, default=1, help='非IID数据分布级别 (1/2/3)')
    parser.add_argument('--epochs', type=int, default=5, help='本地训练轮数')
    parser.add_argument('--batch_size', type=int, default=128, help='本地批次大小')
    parser.add_argument('--lr', type=float, default=0.01, help='联邦学习学习率')
    parser.add_argument('--num_rounds', type=int, default=100, help='每个DRL Episode内联邦学习轮次的总数上限')
    parser.add_argument('--num_episodes', type=int, default=200, help='DRL训练的Episode总数上限')
    
    parser.add_argument('--drl_train', type=int, default=1, help='是否训练DRL智能体 (0/1)')
    parser.add_argument('--drl_algo', type=str, default='pdqn', help='DRL算法 (a3c/pdqn/ddpg/ppo/sac/td3)')
    parser.add_argument('--drl_load', type=str, default=None, help='加载DRL模型路径')
    
    # DRL 智能体超参数
    parser.add_argument('--drl_lr', type=float, default=0.0001, help='DRL Actor网络学习率 (LR)')
    parser.add_argument('--drl_batch_size', type=int, default=128, help='DRL 批次大小 (BATCH_SIZE)')
    parser.add_argument('--drl_gamma', type=float, default=0.95, help='DRL 折扣因子 (GAMMA)')
    parser.add_argument('--drl_memory_size', type=int, default=20000, help='DRL 回放缓存区大小 (MEMORY)')

    # 只保留是否需要绘图的开关
    parser.add_argument('--plot', type=int, default=1, help='是否生成参数比较图 (0/1)')
    
    args = parser.parse_args()
    return args

def set_seed(seed):
    """设置随机种子"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def _get_default_training_params(clients_manager):
    """
    获取默认的训练参数，用于第一轮或禁用DRL时。
    选择所有可用的终端设备在本地训练，并选择第一个可用的边缘节点进行聚合。
    """
    clients_states = {cid: {'available': client.available, 'is_edge': client.is_edge}
                      for cid, client in clients_manager.clients.items()}

    selected_clients = [cid for cid, state in clients_states.items() 
                      if state['available'] and not state['is_edge']]
    
    edge_nodes = [cid for cid, state in clients_states.items() if state['available'] and state['is_edge']]
    
    aggregation_location = edge_nodes[0] if edge_nodes else "cloud"
    
    # 本地训练，所以不需要资源分配给边缘节点
    resource_allocation = {cid: 1.0 for cid in selected_clients}

    # 在默认情况下，所有设备都在本地训练
    drl_train_decisions = [1] * len(selected_clients)

    print(f"默认决策: {len(selected_clients)} 个设备本地训练, 在 {aggregation_location} 聚合。")

    training_args = {
        'selected_nodes': selected_clients,
        'resource_allocation': resource_allocation,
        'aggregation_location': aggregation_location,
        'drl_train_decisions': drl_train_decisions,
    }
    
    # 添加原始决策信息
    raw_decisions = {
        'local_train': np.ones(len(selected_clients)),  # 全部为本地训练
        'edge_train': np.zeros(len(selected_clients) * len(edge_nodes) if edge_nodes else 0),
        'edge_agg': np.zeros(len(edge_nodes)) if edge_nodes else np.array([]),
        'cloud_agg': np.array([0 if edge_nodes else 1]),  # 如果有边缘节点则用边缘聚合，否则用云聚合
        'resource_alloc': np.ones(len(selected_clients))  # 默认分配满资源
    }
    
    return training_args, raw_decisions

def parse_drl_decisions(action, env, server, clients_manager):
    """解析DRL决策并生成训练参数"""
    training_args, raw_decisions = env.get_fl_training_params(action, clients_manager, server)
    
    # 确保local_training_clients只包含普通客户端
    if 'selected_nodes' in training_args:
        training_args['selected_nodes'] = [
            node_id for node_id in training_args['selected_nodes'] 
            if node_id in clients_manager.clients and not clients_manager.clients[node_id].is_edge
        ]
    
    return training_args, raw_decisions
    
def _log_round_stats(round_idx, episode_idx, args, training_args, raw_decisions, info, reward, accuracy, loss, server):
    """记录并打印一轮的统计信息"""
    
    selected_nodes = training_args.get('selected_nodes', [])
    aggregation_location = training_args.get('aggregation_location', 'N/A')
    drl_train_decisions = training_args.get('drl_train_decisions', [])
    
    client_edge_mapping = getattr(server, 'client_edge_mapping', {})
    
    terminal_devices = [node for node in selected_nodes if node.startswith('client')]
    local_train_count = sum(drl_train_decisions)
    edge_train_count = len(drl_train_decisions) - local_train_count

    total_delay = info.get("total_delay", 0.0)
    total_energy = info.get("total_energy", 0.0)
    total_cost = info.get("total_cost", 0.0)
    
    print(f"\n===== DRL Episode {episode_idx}, FL轮次 {round_idx+1} 结果 =====")
    print(f"参与训练的终端设备数量: {len(terminal_devices)}")
    print(f"  - 本地训练数量: {local_train_count}")
    print(f"  - 边缘训练数量: {edge_train_count}")

    if raw_decisions:
        print("原始决策向量:")
        print(f"  - 本地训练: {raw_decisions.get('local_train', 'N/A')}")
        print(f"  - 边缘训练: {raw_decisions.get('edge_train', 'N/A')}")
        print(f"  - 边缘聚合: {raw_decisions.get('edge_agg', 'N/A')}")
        print(f"  - 云聚合:   {raw_decisions.get('cloud_agg', 'N/A')}")
        print(f"  - 资源分配: {np.around(raw_decisions.get('resource_alloc', []), 2)}")

    if client_edge_mapping:
        edge_train_counts = {}
        for _, edge_id in client_edge_mapping.items():
            edge_train_counts[edge_id] = edge_train_counts.get(edge_id, 0) + 1
        print("卸载到各边缘节点的决策:")
        for edge_id, count in edge_train_counts.items():
            print(f"  - {edge_id}: {count}个设备")
    
    print(f"聚合位置: {aggregation_location}")
    print(f"奖励: {reward:.4f}")
    print(f"总延迟={total_delay:.4f}s, 总能耗={total_energy:.4f}J, 总成本={total_cost:.4f}")
    print(f"模型性能: 准确率={accuracy:.4f}, 损失={loss:.4f}")

def finalize_training(server, global_model, test_loader, args, training_stats, drl_agent):
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
    
    # 记录最终训练结果
    # 计算最终轮次和平均指标
    final_round = len(training_stats['accuracy'])
    final_reward = np.mean(training_stats['rewards'][-5:]) if len(training_stats['rewards']) >= 5 else np.mean(training_stats['rewards'])
    final_loss = np.mean(training_stats['loss'][-5:]) if len(training_stats['loss']) >= 5 else np.mean(training_stats['loss'])
    final_cost = np.mean(training_stats['costs'][-5:]) if 'costs' in training_stats and len(training_stats['costs']) >= 5 else np.mean(training_stats.get('costs', [0]))
    
    # 记录最终的学习率结果
    log_training_result(
        "FINAL_LR", args.lr,
        final_round,
        final_reward,
        final_loss,
        cost=final_cost
    )
    
    # 如果使用了DRL，记录最终的DRL算法结果
    log_training_result(
        "FINAL_DRL", args.drl_algo,
        final_round,
        final_reward,
        final_loss,
        cost=final_cost
    )
    
    # 记录最终的准确率
    log_training_result(
        "ACCURACY", round(acc, 4),
        final_round,
        final_reward,
        final_loss,
        cost=final_cost
    )
    
    return acc

def init_system(args):
    """初始化系统组件
    
    Args:
        args: 命令行参数
        
    Returns:
        (env, global_model, test_loader, clients_manager, server, drl_agent) 元组
    """
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else "cpu")
    args.device = device
    
    # 创建保存路径
    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path, exist_ok=True)
    
    print("=" * 50)
    print("初始化系统...")
    
    # 1. 创建环境
    env = Env(num_devices=args.num_clients, num_edges=args.num_edges)
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
    # 修正：根据数据集类型强制选择正确的模型，避免命令行参数混淆
    if args.dataset == 'mnist':
        model_name = 'mnist'  # 强制为MNIST数据集使用CNNMnist模型
    elif args.dataset == 'cifar10':
        model_name = 'cifar'  # 强制为CIFAR-10数据集使用CNNCifar模型
    else:
        model_name = args.model # 其他情况使用指定的模型

    global_model = create_model(model_name)
    global_model = global_model.to(device)
    
    # 4. 创建客户端管理器 - 使用环境预分配的数据集
    disable_progress_bar = bool(args.disable_progress_bar)
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
        'model_name': model_name,
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
    
    # server.env 已在 FederatedServer.__init__ 中声明，无需额外赋值

    # 7. 创建DRL智能体
    drl_agent = create_drl_agent(args, env)
    if args.drl_load:
        drl_agent.load_model(args.drl_load)
        print(f"加载DRL模型: {args.drl_load}")
    print(f"创建DRL智能体: {args.drl_algo.upper()}")
    
    # 8. 同步客户端状态与环境
    clients_manager.update_client_states(env)
    print("同步客户端状态与环境")
    
    print("系统初始化完成")
    print("=" * 50)
    
    return env, global_model, test_loader, clients_manager, server, drl_agent

def main():
    """主程序入口"""
    # 解析参数
    args = parse_args()

    # 设置随机种子
    set_seed(args.seed)
    
    # 初始化系统组件
    env, global_model, test_loader, clients_manager, server, drl_agent = init_system(args)
    
    # 根据DRL超参数为本次运行创建唯一的日志文件名
    log_filename = (f"log_Lr_{args.drl_lr}_batch_{args.drl_batch_size}_"
                    f"gamma_{args.drl_gamma}_memory_{args.drl_memory_size}.txt")
    print(f"本次运行将记录到日志文件: {log_filename}")
    
    # 为每个FL轮次创建并打开详细日志文件
    round_log_path = os.path.join("logs", "fl_round_details.txt")
    if not os.path.exists("logs"):
        os.makedirs("logs")
    round_log_file = open(round_log_path, "w")
    round_log_file.write("round,reward,cost,loss,accuracy\n")
    round_log_file.flush()  # 立即刷新表头
    print(f"每轮FL的详细日志将记录到: {round_log_path}")
    
    # 设置进度条显示
    tqdm_cls = tqdm_notebook.tqdm if 'ipykernel' in sys.modules else tqdm
    disable_progress_bar = bool(args.disable_progress_bar)
    
    # 主训练循环
    print(f"开始训练...")
    print(f"DRL 框架: Episode 上限 = {args.num_episodes}, FL 轮次总上限 = {args.num_rounds}")
    print("每个 Episode 的长度由 FL 模型收敛决定。")
    
    # 初始化训练统计
    training_stats = {'accuracy': [], 'rewards': [], 'times': [], 'loss': [], 'training_loss': [], 'costs': [],
                      'episode_avg_rewards': [], 'episode_avg_loss': [], 'episode_avg_costs': []}
    global_round_idx = 0

    # 使用tqdm创建Episode的进度条
    with tqdm_cls(total=args.num_episodes, desc="DRL Episodes", unit="episode", disable=disable_progress_bar) as episode_pbar:
        # 外层循环：固定数量的DRL Episode
        for episode_idx in range(1, args.num_episodes + 1):
            
            # 1. Episode开始: 重置环境和FL模型
            episode_pbar.set_description(f"Episode {episode_idx}/{args.num_episodes}")
            print(f"\n===== 开始 DRL Episode {episode_idx}/{args.num_episodes} =====")

            # 为新的Episode重置联邦学习模型，开始全新的FL流程
            print("为新的Episode重置联邦学习模型...")
            server.model = create_model(server.args['model_name']).to(server.dev)
            server.global_parameters = server.model.state_dict()
            
            # 重置环境状态
            state = env.reset()
            episode_done = False

            # 为当前episode初始化累加器
            episode_rewards = []
            episode_losses = []
            episode_costs = []

            # 2. 内层循环：在当前Episode内执行FL轮次
            for episode_round in range(args.num_rounds):
                round_start_time = time.time()
                
                # a. DRL 决策
                if global_round_idx > 0:
                    action = drl_agent.get_action(state)
                    training_args, raw_decisions = parse_drl_decisions(action, env, server, clients_manager)
                else:
                    training_args, raw_decisions = _get_default_training_params(clients_manager)
                    action = np.zeros(env.action_dim)
                
                # b. 执行联邦学习训练
                client_edge_mapping = {}
                if 'edge_client_mapping' in raw_decisions:
                    client_edge_mapping = raw_decisions['edge_client_mapping']

                accuracy, global_test_loss, global_training_loss, _, _, _ = server.train_round(
                    local_training_clients=training_args['selected_nodes'],
                    client_edge_mapping=client_edge_mapping,
                    resource_allocation=training_args['resource_allocation'],
                    aggregation_location=training_args['aggregation_location']
                )
                
                # c. 环境步进，传入 全局训练损失 (fl_loss) 以获取核心的 done 标志
                next_state, reward, episode_done, info = env.step(action, fl_loss=global_training_loss)
                
                # d. DRL 智能体立即学习
                if args.drl_train:
                    drl_agent.learn(state, action, reward, next_state, episode_done)

                # e. 更新状态
                state = next_state
                
                # f. 记录和打印统计信息
                _log_round_stats(episode_round, episode_idx, args, training_args, raw_decisions, info, reward, accuracy, global_test_loss, server)

                # g. 更新统计数据
                training_stats['accuracy'].append(accuracy)
                training_stats['loss'].append(global_test_loss)
                training_stats['training_loss'].append(global_training_loss)
                training_stats['rewards'].append(reward)
                training_stats['times'].append(time.time() - round_start_time)
                training_stats['costs'].append(info.get('total_cost', 0.0))
                
                # 累积当前Episode的指标
                episode_rewards.append(reward)
                episode_losses.append(global_test_loss)
                episode_costs.append(info.get('total_cost', 0.0))

                # h. 记录每个FL轮次的详细数据到文件
                log_line = f"{global_round_idx},{reward:.4f},{info.get('total_cost', 0.0):.4f},{global_test_loss:.4f},{accuracy:.4f}\n"
                with open(round_log_path, "a") as f:
                    f.write(log_line)

                # h. 记录到日志文件 (使用全局轮次索引)
                # log_training_result(
                #     "LR", args.lr, global_round_idx, reward, 
                #     global_test_loss, cost=info.get('total_cost', 0.0)
                # )
                # log_training_result(
                #     "DRL_ALGO", args.drl_algo, global_round_idx, reward, 
                #     global_test_loss, cost=info.get('total_cost', 0.0)
                # )
                # log_training_result(
                #     "BATCH_SIZE", args.batch_size, global_round_idx, reward, 
                #     global_test_loss, cost=info.get('total_cost', 0.0)
                # )

                # 更新全局轮次计数器
                global_round_idx += 1
                
                # 更新Episode进度条的后缀信息
                episode_pbar.set_postfix({
                    'FL Round': f"{episode_round+1}/{args.num_rounds}",
                    'Test Loss': f"{global_test_loss:.4f}", 'Acc': f"{accuracy:.4f}", 
                    'Reward': f"{reward:.4f}", 'Done': episode_done
                })
                
                # 如果环境报告Episode结束，则提前跳出内层循环
                if episode_done:
                    print(f"===== DRL Episode {episode_idx} 因FL模型收敛而提前结束 (在第 {episode_round+1} 轮) =====")
                    break

            # Episode结束后的处理
            if episode_rewards:
                avg_reward = np.mean(episode_rewards)
                avg_loss = np.mean(episode_losses)
                avg_cost = np.mean(episode_costs)
                
                training_stats['episode_avg_rewards'].append(avg_reward)
                training_stats['episode_avg_loss'].append(avg_loss)
                training_stats['episode_avg_costs'].append(avg_cost)
                
                print(f"===== DRL Episode {episode_idx} 总结 | 平均奖励: {avg_reward:.4f}, 平均损失: {avg_loss:.4f}, 平均成本: {avg_cost:.4f} =====")

                # 记录Episode的平均奖励到本次运行专属的日志文件
                log_episode_result(
                    log_filename, episode_idx, avg_reward, 
                    avg_loss, cost=avg_cost
                )

            # 更新外层循环(Episode)的进度条
            episode_pbar.update(1)
    
    # 训练结束，进行最终评估和统计
    return args, finalize_training(server, global_model, test_loader, args, training_stats, drl_agent)

# 在训练结束后，如果需要绘图
if __name__ == "__main__":
    # 导入绘图模块 - 移到此处以避免循环导入
    from plot.plot_award import plot_training_results, plot_all_logs, log_episode_result
    
    args, final_acc = main()
    
    # 如果需要绘图
    if args.plot:
        # 批量绘制所有日志文件的图表
        plot_all_logs(save_dir="./plot", show=True)
        
        # 如果使用了DRL，特别绘制DRL算法的结果
        log_name = f"DRL_ALGO_{args.drl_algo}"
        if os.path.exists(os.path.join("./logs", f"{log_name}.txt")):
            plot_training_results(log_name, save_dir="./plot", show=True)
            print(f"已生成DRL算法 {args.drl_algo} 的训练结果图表")
        
        print("所有图表已保存到 ./plot 目录")