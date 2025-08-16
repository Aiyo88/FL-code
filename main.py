import torch
import numpy as np
import os
import argparse
import time
import sys
from tqdm import tqdm
import tqdm.notebook as tqdm_notebook
from torch.utils.data import DataLoader

# 导入自定义模块
from utils.data_utils import get_dataset, get_client_dataset
from models.models import create_model
from Env import Env
from clients import ClientsGroup
from server import FederatedServer
# 直接导入PDQN实现
from PDQN.agents.pdqn import PDQNAgent, QActor, ParamActor
from plot.plot_award import log_training_result, plot_training_results, plot_all_logs, log_episode_result
from models.action_parser import ActionParser
# 导入配置参数
from config import *

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="联邦学习训练")
    
    # 系统参数
    parser.add_argument('--seed', type=int, default=SEED, help='随机种子')
    parser.add_argument('--device', type=str, default='cuda', help='使用设备 (cpu/cuda)')
    parser.add_argument('--save_path', type=str, default=CHECKPOINT_PATH, help='模型保存路径')
    parser.add_argument('--data_path', type=str, default=DATA_PATH, help='数据集路径')
    parser.add_argument('--disable_progress_bar', type=int, default=int(DISABLE_PROGRESS_BAR), help='禁用进度条 (0/1)')
    
    # 联邦学习参数
    parser.add_argument('--dataset', type=str, default=DEFAULT_DATASET, help='数据集 (mnist/cifar10)')
    parser.add_argument('--num_clients', type=int, default=NUM_CLIENTS, help='客户端数量')
    parser.add_argument('--num_edges', type=int, default=NUM_EDGES, help='边缘节点数量')
    parser.add_argument('--iid', type=int, default=int(IID), help='是否为IID数据分布 (0/1)')
    parser.add_argument('--non_iid_level', type=int, default=NON_IID_LEVEL, help='非IID数据分布级别 (1/2/3)')
    parser.add_argument('--epochs', type=int, default=LOCAL_EPOCHS, help='本地训练轮数')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='本地批次大小')
    parser.add_argument('--lr', type=float, default=LEARNING_RATE, help='联邦学习学习率')
    parser.add_argument('--num_rounds', type=int, default=NUM_ROUNDS, help='每个DRL Episode内联邦学习轮次的总数上限')
    parser.add_argument('--num_episodes', type=int, default=NUM_EPISODES, help='DRL训练的Episode总数上限')
    
    parser.add_argument('--drl_load', type=str, default=None, help='加载DRL模型路径')
    
    # DRL 智能体超参数
    parser.add_argument('--drl_lr', type=float, default=DRL_LR, help='DRL Actor网络学习率 (LR)')
    parser.add_argument('--drl_batch_size', type=int, default=DRL_BATCH_SIZE, help='DRL 批次大小 (BATCH_SIZE)')
    parser.add_argument('--drl_gamma', type=float, default=DRL_GAMMA, help='DRL 折扣因子 (GAMMA)')
    parser.add_argument('--drl_memory_size', type=int, default=DRL_MEMORY_SIZE, help='DRL 回放缓存区大小 (MEMORY)')

    # 李雅普诺夫参数
    parser.add_argument('--energy_threshold', type=float, default=ENERGY_THRESHOLD, help='李雅普诺夫队列能量阈值')

    # 绘图开关
    parser.add_argument('--plot', type=int, default=int(PLOT_RESULTS), help='是否生成参数比较图 (0/1)')
    
    args = parser.parse_args()
    return args

def set_seed(seed):
    """设置随机种子"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def _log_round_stats(round_idx, episode_idx, args, training_args, info, reward, accuracy, loss, server, action_info=None):
    """记录并打印一轮的统计信息（精简版）"""
    
    selected_nodes = training_args.get('selected_nodes', [])
    aggregation_location = training_args.get('aggregation_location', 'N/A')
    drl_train_decisions = training_args.get('drl_train_decisions', [])
    
    client_edge_mapping = getattr(server, 'client_edge_mapping', {})
    
    local_train_count = sum(drl_train_decisions)
    edge_train_count = len(drl_train_decisions) - local_train_count

    total_delay = info.get("total_delay", 0.0)
    total_energy = info.get("total_energy", 0.0)
    total_cost = info.get("total_cost", 0.0)
    
    print(f"\n===== DRL Episode {episode_idx}, FL轮次 {round_idx} 结果 =====")
    
    # 1. 训练决策
    if action_info:
        local_train = action_info['local_train_decisions']
        edge_train = action_info['edge_train_matrix']
        print(f"本地训练决策: {local_train}")
        print(f"边缘训练决策: {edge_train.tolist()}")
    
    # 2. 客户端映射和聚合位置  
    print(f"客户端到边缘节点映射: {client_edge_mapping}")
    print(f"聚合位置: {aggregation_location}")
    
    # 3. 资源分配动作
    if action_info:
        res_alloc = action_info['resource_alloc_matrix']
        print(f"资源分配动作: {res_alloc.tolist()}")
        
        # 检查资源冲突
        edge_train = action_info['edge_train_matrix']
        has_conflict = False
        for j in range(res_alloc.shape[1]):
            total_alloc = sum(res_alloc[i, j] * edge_train[i, j] for i in range(res_alloc.shape[0]))
            if total_alloc > 1.0:
                has_conflict = True
                break
        if has_conflict:
            print("⚠️  资源分配超限")
    
    # 4. 性能指标
    print(f"奖励: {reward:.4f}")
    print(f"总延迟={total_delay:.4f}s, 总能耗={total_energy:.4f}J, 总成本={total_cost:.4f}")
    print(f"准确率={accuracy:.4f}, 损失={loss:.4f}")

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
    acc, _ = server.evaluate(test_loader)
    print(f"最终准确率: {acc:.4f}")
    
    # 保存最终模型
    server.save_model(add_round=False)
    model_path = os.path.join(args.save_path, "drl_pdqn_final")
    drl_agent.save_models(model_path)
    
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
                    "FINAL_DRL", "pdqn",
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
    
    # 1. 创建客户端管理器和数据集（顺序调整）
    clients_manager = ClientsGroup(
        dataset_name=args.dataset, 
        is_iid=bool(args.iid), 
        num_clients=args.num_clients,
        device=device,
        num_edges=args.num_edges,
        non_iid_level=args.non_iid_level
    )
    
    # 2. 准备数据集
    print("准备数据集...")
    if not os.path.exists(args.data_path):
        os.makedirs(args.data_path)
    train_data, test_loader = get_dataset(args.dataset, args.data_path, args.batch_size) # get_dataset 直接返回 test_loader
    
    # 使用准备好的数据来设置客户端的基础设施
    clients_manager.setup_infrastructure(train_data=train_data)
    print(f"创建客户端管理器: 客户端数量={args.num_clients}, 边缘节点数量={args.num_edges}, {'IID' if args.iid else f'非IID-L{args.non_iid_level}'} 数据分布")

    # 3. 创建环境，并传入clients_manager以获取连接信息
    env = Env(
        num_devices=args.num_clients, 
        num_edges=args.num_edges, 
        energy_threshold=args.energy_threshold,
        clients_manager=clients_manager  # <--- 传入管理器
    )
    print(f"创建环境: 终端设备={env.N}, 边缘节点={env.M}")

    # 在加载数据集部分之前添加：
    args.dataset = args.dataset.lower()
    supported_datasets = ['mnist', 'cifar10']
    if args.dataset not in supported_datasets:
        raise ValueError(f"Supported datasets: {supported_datasets}, got {args.dataset}")

    if not os.path.exists(args.data_path):
        os.makedirs(args.data_path)
        print(f"Created data directory: {args.data_path}")
        
    # 4. 初始化模型
    print("初始化模型...")

    if args.dataset == 'mnist':
        model_name = 'mnist'  
    elif args.dataset == 'cifar10':
        model_name = 'cifar' 
    else:
        # 默认使用CNN模型
        model_name = 'cnn'

    global_model = create_model(model_name)
    global_model = global_model.to(device)
    
    # 6. 创建服务器
    server_args = {
        'device': device,
        'learning_rate': args.lr,
        'model_name': model_name,
        'save_path': args.save_path,
        'local_epochs': args.epochs,
        'batch_size': args.batch_size,
        'val_freq': 1,  # 每轮评估
        'save_freq': 5,  # 每5轮保存
        'disable_progress_bar': bool(args.disable_progress_bar)  # 传递进度条禁用参数
    }
    server = FederatedServer(server_args, global_model, clients_manager)
    print("创建服务器")
    
    # 7. 为环境设置客户端和服务器
    env.set_clients_and_server(clients_manager, server)

    # 8. 直接创建PDQN智能体
    # PDQN参数配置
    pdqn_params = {
        'batch_size': min(args.drl_batch_size, 256),
        'gamma': args.drl_gamma,
        'replay_memory_size': args.drl_memory_size,
        'learning_rate_actor': args.drl_lr * 0.1,
        'learning_rate_actor_param': args.drl_lr * 0.1,
        'inverting_gradients': True,
        'initial_memory_threshold': 32,
        # 关闭 OU 噪声，启用 ε-分支下的区间均匀采样，便于覆盖[0,1]
        'use_ornstein_noise': False,
        'epsilon_steps': 30,
        'epsilon_final': 0.02,
        'tau_actor': 0.01,
        'tau_actor_param': 0.005,
        'clip_grad': 1.0,
        'zero_index_gradients': False,
    }
    
    # 网络结构参数
    actor_kwargs = {'hidden_layers': (256, 128, 64)}
    actor_param_kwargs = {'hidden_layers': (256, 128, 64)}
    
    # 创建PDQN智能体
    print(f"PDQN智能体使用复合动作空间: {env.action_space}")
    drl_agent = PDQNAgent(
        env.observation_space, 
        env.action_space,
        actor_class=QActor,
        actor_kwargs=actor_kwargs,
        actor_param_class=ParamActor,
        actor_param_kwargs=actor_param_kwargs,
        **pdqn_params
    )
    
    if args.drl_load:
        drl_agent.load_models(args.drl_load)
        print(f"加载DRL模型: {args.drl_load}")
    print(f"创建DRL智能体: PDQN")

    # 9. 创建动作解析器
    action_parser = ActionParser(num_devices=args.num_clients, num_edges=args.num_edges)
    
    # 10. 同步客户端状态与环境
    clients_manager.update_client_states(env)
    print("同步客户端状态与环境")
    
    print("系统初始化完成")
    print("=" * 50)
    
    return env, global_model, test_loader, clients_manager, server, drl_agent, action_parser

def main():
    """主程序入口"""
    # 解析参数
    args = parse_args()

    # 设置随机种子
    set_seed(args.seed)
    
    # GPU优化设置
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True  # 加速卷积运算
        torch.backends.cudnn.deterministic = False  # 允许非确定性以获得更好性能
        # 设置GPU内存分配策略
        torch.cuda.empty_cache()  # 清空GPU缓存
        print(f"GPU加速已启用: {torch.cuda.get_device_name()}")
        print(f"可用GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # 初始化系统组件
    env, global_model, test_loader, clients_manager, server, drl_agent, action_parser = init_system(args)
    
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
    
    # 添加returns数组跟踪Episode表现 - 参考runliang.py
    returns = []              # Episode累积奖励历史
    returns_costs = []        # Episode累积成本历史

    # 使用tqdm创建Episode的进度条
    with tqdm_cls(total=args.num_episodes, desc="DRL Episodes", unit="episode", disable=disable_progress_bar) as episode_pbar:
        # 外层循环：固定数量的DRL Episode
        for episode_idx in range(1, args.num_episodes + 1):
            
            # 1. Episode开始: 重置环境和FL模型
            episode_pbar.set_description(f"Episode {episode_idx}/{args.num_episodes}")
            print(f"\n===== 开始 DRL Episode {episode_idx}/{args.num_episodes} =====")

            # 为新的Episode重置服务器状态和联邦学习模型
            print("为新的Episode重置服务器和联邦学习模型...")
            server.reset_for_new_episode()
            
            # 重置环境状态
            state = env.reset()
            episode_done = False
            
            # DRL智能体Episode开始
            drl_agent.start_episode()

            # Episode级别统计初始化 
            episode_rewards = []      # Episode奖励累积列表
            episode_losses = []       # Episode损失累积列表  
            episode_costs = []        # Episode成本累积列表
            
            # Episode累积变量 - 参考DRL训练流程
            episode_reward = 0.0      # Episode总奖励
            episode_cost = 0.0        # Episode总成本

            # 2. 内层循环：在当前Episode内执行FL轮次
            
            # 格式化初始状态
            state = np.array(state, dtype=np.float32, copy=False)
            
            for episode_round in range(args.num_rounds):
                round_start_time = time.time()
                global_round_idx += 1 # 统一在循环开始时递增，使其从1开始
                
                # a. DRL 智能体选择动作 - 每个step都重新选择
                act, act_param = drl_agent.act(state)
                action = (act, act_param)
                
                # b. 解析动作为联邦学习训练参数
                parsed_action = action_parser.parse_action_for_training(action, clients_manager, server)
                training_args, raw_decisions = action_parser.convert_to_fl_training_params(parsed_action)
                
                # 保存动作信息供后续打印使用（不在此处打印）
                action_info = {
                    'local_train_decisions': parsed_action['local_train_decisions'],
                    'edge_train_matrix': parsed_action['edge_train_matrix'],
                    'resource_alloc_matrix': parsed_action['resource_alloc_matrix'],
                    'client_edge_mapping': parsed_action['client_edge_mapping'],
                    'aggregation_location': parsed_action['aggregation_location']
                }
                
                # 将客户端到边缘节点的映射信息保存到服务器中
                if hasattr(server, 'client_edge_mapping'):
                    server.client_edge_mapping = raw_decisions.get('edge_client_mapping', {})

                # c. 执行联邦学习训练
                client_edge_mapping = raw_decisions.get('edge_client_mapping', {})
                accuracy, global_test_loss, global_training_loss, _, _, _, is_fl_converged = server.train_round(
                    local_training_clients=training_args['selected_nodes'],
                    client_edge_mapping=client_edge_mapping,
                    resource_allocation=training_args['resource_allocation'],
                    aggregation_location=training_args['aggregation_location'],
                    test_loader=test_loader
                )
                
                # d. 环境步进，传入已经解析好的决策和原始动作
                next_state, reward, episode_done, info = env.step(action, raw_decisions, global_round_idx, episode_idx, global_training_loss)
                
                # e. 格式化next_state
                next_state = np.array(next_state, dtype=np.float32, copy=False)
                
                # f. DRL 智能体经验存储和学习 - 参考流程模式
                # PDQN使用 (state, action, reward, next_state, terminal) 五元组
                # next_action设为None，让PDQN在需要时重新计算
                drl_agent.step(state, action, reward, next_state, None, episode_done)

                # g. 状态更新
                state = next_state
                
                # h. Episode奖励和成本累积 - 参考DRL训练流程
                episode_reward += reward
                episode_cost += info.get('total_cost', 0.0)
                
                # i. 记录和打印统计信息 - 直接传递从1开始的全局轮次
                _log_round_stats(global_round_idx, episode_idx, args, training_args, info, reward, accuracy, global_test_loss, server, action_info)

                # j. 更新统计数据
                training_stats['accuracy'].append(accuracy)
                training_stats['loss'].append(global_test_loss)
                training_stats['training_loss'].append(global_training_loss)
                training_stats['rewards'].append(reward)
                training_stats['times'].append(time.time() - round_start_time)
                training_stats['costs'].append(info.get('total_cost', 0.0))
                
                # 累积当前Episode的指标列表
                episode_rewards.append(reward)
                episode_losses.append(global_test_loss)
                episode_costs.append(info.get('total_cost', 0.0))

                # k. 记录每个FL轮次的详细数据到文件 - 直接使用从1开始的全局轮次
                log_line = f"{global_round_idx},{reward:.4f},{info.get('total_cost', 0.0):.4f},{global_test_loss:.4f},{accuracy:.4f}\n"
                with open(round_log_path, "a") as f:
                    f.write(log_line)

                # l. 更新Episode进度条的后缀信息
                episode_pbar.set_postfix({
                    'FL Round': f"{episode_round+1}/{args.num_rounds}",
                    'Test Loss': f"{global_test_loss:.4f}", 'Acc': f"{accuracy:.4f}", 
                    'Reward': f"{reward:.4f}", 'Total': f"{episode_reward:.2f}", 'Done': episode_done
                })
                
                # 如果FL模型收敛，则提前结束当前Episode
                if is_fl_converged:
                    print(f"\n===== DRL Episode {episode_idx} 因联邦学习模型收敛而提前结束 (在FL轮次 {episode_round+1}) =====")
                    break

                # 如果环境报告Episode结束，则提前跳出内层循环
                if episode_done:
                    print(f"===== DRL Episode {episode_idx} 因环境信号而提前结束 (在FL轮次 {episode_round+1}) =====")
                    break

            # 3. Episode结束后：DRL智能体的Episode级处理
            # PDQN在Episode结束时更新epsilon等超参数
            drl_agent.end_episode()

            # 4. Episode结束后的统计和日志处理
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
    from plot.plot_award import plot_training_results, plot_all_logs
    
    args, final_acc = main()
    
    # 如果需要绘图
    if args.plot:
        # 批量绘制所有日志文件的图表
        plot_all_logs(save_dir="./plot", show=True)
        
        # 如果使用了DRL，特别绘制DRL算法的结果
        log_name = "DRL_ALGO_pdqn"
        if os.path.exists(os.path.join("./logs", f"{log_name}.txt")):
            plot_training_results(log_name, save_dir="./plot", show=True)
            print(f"已生成DRL算法 PDQN 的训练结果图表")
        
        print("所有图表已保存到 ./plot 目录")