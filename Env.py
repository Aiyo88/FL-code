import numpy as np
import torch
import math
import time
import gym
from gym import spaces
from utils.data_utils import get_dataset
import os
# 移除GNN相关导入

# 导入配置参数
from config import *

# 导入新的模块化组件
from models.lyapunov_queue import LyapunovQueue, LyapunovQueueManager
from models.communication_model import WirelessCommunicationModel
from models.computation_model import ComputationModel
from models.action_parser import ActionParser
from models.cost_calculator import CostCalculator


class Env:
    """环境类 - 提供状态、动作、奖励计算"""
    def __init__(self, num_devices=DEFAULT_NUM_DEVICES, num_edges=DEFAULT_NUM_EDGES, 
                 energy_threshold=ENERGY_THRESHOLD, clients_manager=None):
        # 系统参数
        self.N = num_devices        # 终端设备数量
        self.M = num_edges         # 边缘服务器数量
        self.K = NUM_CLOUD_SERVERS         # 云服务器数量
        
        self.clients_manager = clients_manager
        self.connectivity_matrix = self._get_connectivity()
        
        # 记录历史数据
        self.T = NUM_ROUNDS       # 最大时隙数量
        self.delays_history = np.zeros((self.N, self.T))
        self.energies_history = np.zeros((self.N, self.T))
        self.costs_history = np.zeros((self.N, self.T))
        
        # 成本计算相关变量
        self.total_cost = 0.0        # 累积的总成本
        self.round_costs = []        # 每轮的成本列表

        # 联邦学习和DRL的分层嵌套关系参数
        self.fl_rounds_per_episode = FL_ROUNDS_PER_EPISODE  # 每个DRL Episode包含的FL轮次数量
        self.current_fl_round = 0       # 当前Episode中的FL轮次计数
        self.current_episode = 0        # 当前DRL Episode计数
        self.episode_rewards = []       # 当前Episode累积的奖励
        self.episode_states = []        # 当前Episode的状态历史
        self.episode_actions = []       # 当前Episode的动作历史
        self.episode_next_states = []   # 当前Episode的下一状态历史
        
        # 数据集相关
        self.dataset_name = None       # 数据集名称
        self.is_iid = IID             # 是否为IID数据分布
        self.non_iid_level = NON_IID_LEVEL       # 非IID程度
        self.train_data = None         # 训练数据
        self.test_data = None          # 测试数据
        self.client_datasets = {}      # 客户端数据集字典
        
        # 数据模型
        self.data_sizes = np.ones(self.N) * DEFAULT_DATA_SIZE  # 默认数据大小
        self.model_sizes = np.ones(self.N) * DEFAULT_MODEL_SIZE   # 默认模型大小

        # 时间模型
        self.time_slot = TIME_SLOT

        # ===== 初始化模块化组件 =====
        
        # 1. 通信模型
        self.comm_model = WirelessCommunicationModel(
            num_devices=self.N, 
            num_edges=self.M,
            bandwidth=BANDWIDTH,
            noise_power=NOISE_POWER
        )
        
        # 2. 计算模型
        self.comp_model = ComputationModel(
            num_devices=self.N,
            num_edges=self.M
        )
        
        # 3. 李雅普诺夫队列管理器
        self.queue_manager = LyapunovQueueManager(
            num_devices=self.N,
            energy_threshold=energy_threshold
        )
        
        # 5. 成本计算器
        self.cost_calculator = CostCalculator(
            communication_model=self.comm_model,
            computation_model=self.comp_model,
            alpha=ALPHA,  # 延迟权重
            beta=BETA    # 能耗权重
        )
        
        # 设置成本计算器的数据和模型大小
        self.cost_calculator.set_data_model_sizes(self.data_sizes, self.model_sizes)

        # 状态空间
        # 每个终端设备状态: 5个基本状态 + M个上行速率 + M个下行速率
        # 每个边缘节点状态: 1个计算资源 + 2个与云的传输速率
        # 修正：状态维度应为 5*N (设备) + 2*N*M (设备-边缘通信) + 3*M (边缘节点状态)
        self.state_dim = get_state_dim(self.N, self.M)
        print(f"计算的状态维度: {self.state_dim} (N={self.N}, M={self.M})")

        # === 新动作空间设计 V2：分离聚合决策 ===
        # 离散动作: 聚合位置 (M个边缘 + 1个云)
        self.discrete_action_space = spaces.Discrete((self.M + 1)**(self.N + 1))
        
        # 连续动作: 仅用于边缘资源分配 (不为本地设备分配)
        # 展平为一维向量，长度为 N*M，解析时再 N×M 矩阵
        self.continuous_action_space = spaces.Box(
            low=0, high=1, shape=(self.N * self.M,), dtype=np.float32
        )
        
        # 组合成PDQN兼容的动作空间
        self.action_space = spaces.Tuple((
            self.discrete_action_space,
            self.continuous_action_space
        ))
        
        print(f"离散部分{self.M+1}维, 连续部分({self.N*self.M},)")
        
        # Lyapunov参数
        self.energy_max = energy_threshold  # 使用可配置的阈值
        self.e_avg = self.energy_max  # 移除平均计算
        self.alpha = ALPHA  # 延迟权重
        self.beta = BETA  # 能耗权重
        self.convergence_epsilon = CONVERGENCE_EPSILON_ENV  # 新的收敛阈值: F(ωt) - F(ω*) <= ϵ
        self.best_loss = float('inf')    # 记录当前episode的最优损失
        
        # --- 新增：奖励归一化参数 ---
        self.max_cost_per_round = MAX_COST_PER_ROUND  # 预估的单轮最大成本
        self.max_q_energy_per_round = MAX_Q_ENERGY_PER_ROUND # 预估的单轮最大队列能量项
        
        # 定义observation_space为状态向量空间
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(self.state_dim,), 
            dtype=np.float32
        )
        print(f"状态空间维度: {self.state_dim} = {self.N}×(4+{self.M}) + {self.M}×4")
        print(f"设置状态空间: {self.state_dim}维向量")
        
        # 定义max_episodes属性供DRL算法使用
        self.max_episodes = 0
        
        # 添加调试信息属性
        self.debug_info = {
            'step_count': 0,
            'reward_history': []
        }
        
        # 添加info属性，用于存储延迟、能耗和成本信息
        self.info = {
            'total_delay': 0.0,
            'total_energy': 0.0,
            'total_cost': 0.0,
            'valid_ratio': 1.0
        }
        
        # 显式声明env属性，避免类型检查器报错
        self.env = None


    def _get_connectivity(self):
        """
        计算并返回客户端到边缘节点的连接矩阵。
        """
        if self.clients_manager is None:
            # 如果没有管理器信息，则退回全连接假设
            return np.ones((self.N, self.M), dtype=bool)
            
        matrix = np.zeros((self.N, self.M), dtype=bool)
        for i in range(self.N):
            client_id = f"client{i}"
            connected_edges = self.clients_manager.get_edge_for_client(client_id)
            for edge_id in connected_edges:
                edge_idx = int(edge_id.split('_')[1])
                if edge_idx < self.M:
                    matrix[i, edge_idx] = True
        return matrix

    def reset(self):
        """重置环境时初始化所有队列"""
        # 重置队列管理器
        self.queue_manager.reset_queues()
        
        # 重置FL轮次和episode计数
        self.current_fl_round = 0
        self.current_episode += 1
        
        # 清空当前Episode的历史数据
        self.episode_rewards = []
        self.episode_states = []
        self.episode_actions = []
        self.episode_next_states = []
        
        # 重置延迟和能耗历史
        self.delays_history = np.zeros((self.N, self.T))
        self.energies_history = np.zeros((self.N, self.T))
        self.costs_history = np.zeros((self.N, self.T))
        
        # 重置Episode内成本累积
        if self.current_episode == 1:
            self.total_cost = 0.0
            self.round_costs = []
        
        self.best_loss = float('inf') # 每个episode开始时重置
        
        # 重置计算模型资源
        self.comp_model.reset_resources()
        
        
        # 环境随机变化
        self._randomize_environment()
        
        # 获取初始状态
        initial_state = self._get_state()
        
        # 记录初始状态
        self.episode_states.append(initial_state)
        
        print(f"开始新的DRL Episode {self.current_episode}，每个Episode包含 {self.fl_rounds_per_episode} 个FL轮次")
        
        # 调试信息更新
        self.debug_info['step_count'] = 0
        self.debug_info['reward_history'] = []
        
        return initial_state
        
    def _get_state(self):
        """构建状态向量 - 使用原始物理量，不进行归一化"""
        # 获取原始队列状态（不归一化）
        queue_states = self.queue_manager.get_queue_states()
        
        # 构建状态向量
        state = []
        
        # 1. 设备状态 - 使用原始物理量
        for i in range(self.N):
            # 设备基本信息（保持原始物理量）
            state.extend([
                self.data_sizes[i],                  # 原始数据大小 (bytes)
                self.model_sizes[i],                 # 原始模型大小 (bytes)
                queue_states[i],                     # 原始队列状态 (energy units)
                self.comp_model.f_l[i],             # 原始本地计算频率 (Hz)
            ])
            
            # 设备与边缘的通信状态（原始值）
            for j in range(self.M):
                if self.connectivity_matrix[i, j]:
                    # 原始通信参数
                    state.extend([
                        self.comm_model.h_up[i, j],      # 原始上行信道增益
                        self.comm_model.h_down[i, j],    # 原始下行信道增益
                        self.comm_model.R_up[i, j],      # 原始上行传输速率 (bps)
                        self.comm_model.R_down[i, j],    # 原始下行传输速率 (bps)
                    ])
                else:
                    # 不连接时全为0
                    state.extend([0.0, 0.0, 0.0, 0.0])
        
        # 2. 边缘节点状态 - 使用原始物理量
        for j in range(self.M):
            state.extend([
                self.comp_model.F_e[j],              # 原始最大计算频率 (Hz)
                self.comp_model.f_e[j],              # 原始当前分配频率 (Hz)
                self.comm_model.rate_CU,             # 原始云上行速率 (bps)
                self.comm_model.rate_CD,             # 原始云下行速率 (bps)
            ])
        
        return np.array(state, dtype=np.float32)

    def step(self, action, raw_decisions, global_round_idx, episode_idx, fl_loss=None):
        """
        执行一步动作 - 对应一个FL轮次
        :param action: 原始动作元组，用于存储和学习
        :param raw_decisions: 已经由 get_fl_training_params 解析好的原始决策字典
        :param global_round_idx: 当前的全局FL轮次索引 (从1开始)
        :param episode_idx: 当前的DRL Episode索引 (从1开始)
        :param fl_loss: 从服务器传来的全局损失，用于判断收敛
        :return: 新状态，奖励，是否结束当前Episode，附加信息
        """
        # 初始化信息字典
        info = {
            "total_delay": 0.0,
            "total_energy": 0.0, 
            "total_cost": 0.0,
            "valid_ratio": 1.0 # 假设所有动作都是有效的
        }
        

        
        # 提取解析结果 - 直接从传入的raw_decisions获取
        train_local_decisions = raw_decisions.get('local_train', np.zeros(self.N,dtype=int))
        edge_train_matrix = raw_decisions.get('edge_train', np.zeros((self.N, self.M),dtype=int))
        edge_agg_decisions = raw_decisions.get('edge_agg', np.zeros(self.M,dtype=int))
        cloud_agg_decision = raw_decisions.get('cloud_agg', 0)
        res_alloc_matrix = raw_decisions.get('resource_alloc', np.zeros((self.N, self.M),dtype=float))

        # 将连续动作从[-1,1]线性映射到[0,1]，再裁剪并按卸载选择掩码
        try:
            res_alloc_matrix = np.array(res_alloc_matrix, dtype=float)
            edge_train_matrix = np.array(edge_train_matrix, dtype=int)
            res_alloc_matrix = (res_alloc_matrix + 1.0) / 2.0
            res_alloc_matrix = np.clip(res_alloc_matrix, 0.0, 1.0)
            res_alloc_matrix *= edge_train_matrix
        except Exception:
            res_alloc_matrix = np.zeros((self.N, self.M), dtype=float)


        
        # 使用成本计算器计算延迟和能耗
        delays, energies, costs, valid_flags, penalty_flag = self.cost_calculator.calculate_system_total_cost(
            train_local_decisions, edge_train_matrix, edge_agg_decisions, cloud_agg_decision, res_alloc_matrix
        )
        
        # 记录历史数据
        for i in range(self.N):
            if i < len(delays):
                self.delays_history[i, self.current_fl_round] = delays[i]
                self.energies_history[i, self.current_fl_round] = energies[i]
                self.costs_history[i, self.current_fl_round] = costs[i]
        
        # 计算总延迟、能耗和成本
        total_delay = sum(delays)
        total_energy = sum(energies)
        total_cost = sum(costs)
        # 更新队列
        self.queue_manager.update_all_queues(energies)
        
        # 使用成本计算器计算奖励
        total_reward = self.cost_calculator.calculate_lyapunov_reward(
            costs, energies, self.queue_manager, penalty_flag=penalty_flag
        )
        
        
        # 更新状态
        next_state = self._get_state()
        self.current_fl_round += 1 # 内部的episode轮次计数器仍然需要
        
        # 判断Episode是否结束
        converged = False
        if fl_loss is not None:
            # 如果当前损失与最优损失的绝对差值小于epsilon，则认为收敛
            if abs(fl_loss - self.best_loss) <= self.convergence_epsilon and self.best_loss != float('inf'):
                converged = True
                print(f"  [Convergence] Loss difference {abs(fl_loss - self.best_loss):.6f} <= {self.convergence_epsilon}. Episode ends.")
            
            # 更新最优损失
            self.best_loss = min(self.best_loss, fl_loss)

        max_rounds_reached = self.current_fl_round >= self.fl_rounds_per_episode
        episode_done = converged or max_rounds_reached
        
        if max_rounds_reached and not converged:
            print(f"  [Max Rounds] Reached {self.current_fl_round} FL rounds, but not converged. Episode ends.")

        # 记录数据
        self.episode_rewards.append(total_reward)
        self.episode_actions.append(action)
        self.episode_next_states.append(next_state)
        self.debug_info['reward_history'].append(total_reward)
        
        # 更新info字典
        info = {
            "total_delay": total_delay,
            "total_energy": total_energy, 
            "total_cost": total_cost,
            "valid_ratio": np.mean(valid_flags) if valid_flags else 0.0
        }
        
        # 将信息保存到self.info中，确保其他方法可以访问
        self.info = info.copy()
        
        return next_state, total_reward, episode_done, info
        
    def initialize_dataset(self, dataset_name, is_iid, non_iid_level=0.5, data_path='./data'):
        """初始化数据集，供客户端使用
        
        Args:
            dataset_name: 数据集名称 ('mnist', 'cifar10'等)
            is_iid: 是否使用IID数据分布
            non_iid_level: 非IID程度 (0-1之间)
            data_path: 数据存储路径
            
        Returns:
            (env_datasets, test_loader): 数据集字典和测试数据加载器
        """
        from torch.utils.data import DataLoader, Dataset
        from utils.data_utils import get_dataset
        
        # 加载数据集
        train_data, test_data = get_dataset(dataset_name, data_path)
        
        # 创建测试数据加载器
        if isinstance(test_data, Dataset):
            test_loader = DataLoader(test_data, batch_size=64, shuffle=False)
        else:
            # 如果已经是DataLoader，直接使用
            test_loader = test_data
        
        # 设置环境属性
        self.dataset_name = dataset_name
        self.is_iid = is_iid
        self.non_iid_level = non_iid_level
        self.train_data = train_data
        self.test_data = test_data
        
        # 返回数据集字典
        env_datasets = {
            'train_data': train_data,
            'test_data': test_data,
            'is_iid': is_iid,
            'non_iid_level': non_iid_level
        }
        
        return env_datasets, test_loader
        
    def set_clients_and_server(self, clients_manager, server):
        """设置客户端管理器和服务器引用
        
        Args:
            clients_manager: 客户端管理器实例
            server: 服务器实例
        """
        self.clients_manager = clients_manager
        self.server = server
        
        # 让服务器也能访问环境
        if hasattr(server, 'env'):
            server.env = self
                
        return self
        
    def get_episode_data(self):
        """获取当前Episode的所有历史数据，用于DRL智能体学习"""
        return {
            'states': self.episode_states,
            'actions': self.episode_actions,
            'rewards': self.episode_rewards,
            'next_states': self.episode_next_states
        }

    def _randomize_environment(self):
        """每个round随机变化无线信道"""
        self.comm_model.update_channel_gains()
        self.comm_model.update_transmission_rates()