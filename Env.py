import numpy as np
import torch
import math
import time
import gym
from gym import spaces
from utils.data_utils import get_dataset
import os

class LyapunovQueue:
    """李雅普诺夫能量队列类 - 实现能量队列的更新、李雅普诺夫函数、漂移计算及稳定性判定"""
    def __init__(self, e_avg):
        self.e_avg = e_avg  # 平均能量消耗阈值
        self.queue = 0.0    # 能量队列初始值
        
    def update_queue(self, e_u_t):
        """更新能量队列: Q(t+1) = max{Q(t) - E_avg, 0} + E(t)"""
        self.queue = max(0, self.queue - self.e_avg) + e_u_t
        return self.queue
    
    def q_u_compute(self, e_u_t):
        """计算队列更新后的值，但不实际更新队列"""
        return max(0, self.queue - self.e_avg) + e_u_t
    
    def lyapunov_function(self):
        """计算李雅普诺夫函数: L(Q) = 1/2 * Q²"""
        return 0.5 * self.queue**2
    
    def lyapunov_drift(self, e_u_t):
        """计算李雅普诺夫漂移: ΔL = L(Q(t+1)) - L(Q(t))"""
        q_next = self.q_u_compute(e_u_t)
        return 0.5 * q_next**2 - self.lyapunov_function()
    
    def is_stable(self, e_u_t):
        """判断队列是否稳定: 如果漂移小于等于0，则队列稳定"""
        return self.lyapunov_drift(e_u_t) <= 0

class Env:
    """环境类 - 提供状态、动作、奖励计算"""
    def __init__(self, num_devices, num_edges):
        # 系统参数
        self.N = num_devices        # 终端设备数量
        self.M = num_edges         # 边缘服务器数量
        self.K = 1         # 云服务器数量
        
        # 记录历史数据
        self.T = 100       # 最大时隙数量
        self.delays_history = np.zeros((self.N, self.T))
        self.energies_history = np.zeros((self.N, self.T))
        self.costs_history = np.zeros((self.N, self.T))
        
        # 成本计算相关变量
        self.total_cost = 0.0        # 累积的总成本
        self.round_costs = []        # 每轮的成本列表

        
        # 联邦学习和DRL的分层嵌套关系参数
        self.fl_rounds_per_episode = 100  # 每个DRL Episode包含的FL轮次数量
        self.current_fl_round = 0       # 当前Episode中的FL轮次计数
        self.current_episode = 0        # 当前DRL Episode计数
        self.episode_rewards = []       # 当前Episode累积的奖励
        self.episode_states = []        # 当前Episode的状态历史
        self.episode_actions = []       # 当前Episode的动作历史
        self.episode_next_states = []   # 当前Episode的下一状态历史
        
        # 数据集相关
        self.dataset_name = None       # 数据集名称
        self.is_iid = True             # 是否为IID数据分布
        self.non_iid_level = 0.5       # 非IID程度
        self.train_data = None         # 训练数据
        self.test_data = None          # 测试数据
        self.client_datasets = {}      # 客户端数据集字典
        
        # 数据模型
        self.data_sizes = np.ones(self.N) * 10 * 1024 * 1024  # 10MB

        # 通信模型
        self.B = 6  # 信道带宽 (MHz)
        self.Pt_UP = 24    # 上行传输功率 (dBM)
        self.Pt_down = 30  # 下行传输功率 (dBM)
        self.Pt_edge_transmit = 24 # 边缘节点传输功率 (dBM) - 用于任务成本计算
        self.Pt_cloud_down = 30 # 云到边缘下行传输功率(dBM) - 用于任务成本计算
        self.N0 = 10e-13   # 噪声功率
        self.rate_CU = 120  # 边缘到云上行速率 (Mbps)
        self.rate_CD = 150  # 边缘到云下行速率 (Mbps)
        self.R_up=np.zeros((self.N,self.M))
        self.R_down=np.zeros((self.N,self.M))
        # 计算模型
        # 终端设备计算资源 (异构)
        self.f_l = np.random.uniform(0.4e9, 2.9e9, size=self.N)  # 0.4GHz~2.9GHz
        # 终端设备计算资源范围
        self.f_l_min = 0.4e9  # 最小CPU频率 (0.4GHz)
        self.f_l_max = 2.9e9  # 最大CPU频率 (2.9GHz)
        
        # 计算复杂度 (cycles/bit)
        self.c = np.random.uniform(300, 500, size=self.N)  # 不同设备不同复杂度
        
        # 边缘节点计算资源
        self.f_e_min = 2.9e9  # 最小CPU频率 (2.9GHz)
        self.f_e_max = 4.3e9  # 最大CPU频率 (4.3GHz)
        self.f_e = np.full(self.M, self.f_e_min)  # 当前分配的计算资源 (Hz)
        self.F_e = np.full(self.M, self.f_e_max)  # 最大计算能力上限 (Hz)

        # 时间模型
        self.time_slot = 0.1

        # 状态空间
        # 每个终端设备状态: 5个基本状态 + M个上行速率 + M个下行速率
        # 每个边缘节点状态: 1个计算资源 + 2个与云的传输速率
        self.state_dim = 5*self.N + 2*self.N*self.M + 2*self.M  # 修正为正确的计算公式
        print(f"计算的状态维度: {self.state_dim} (N={self.N}, M={self.M})")

        # 动作空间：
        # 1. 训练决策: 
        #    - x_i^l(t): N个二元变量，表示设备i是否在本地训练
        #    - y_(i,m)^e(t): N*M个二元变量，表示设备i是否在边缘节点m上训练
        # 2. 聚合决策:
        #    - z_m^e(t): M个二元变量，表示是否在边缘节点m上聚合
        #    - w^c(t): 1个二元变量，表示是否在云端聚合
        # 3. 资源分配:
        #    - f_(i,m)^e(t): N*M个连续变量，表示为边缘节点m上的设备i分配的计算资源

        # 计算动作空间维度
        self.train_dim = self.N * (1 + self.M)  # x_i^l(t) + y_(i,m)^e(t)
        self.agg_dim = self.M + 1                # z_m^e(t) + w^c(t) - 修改为一个全局聚合决策
        self.res_dim = self.N * self.M             # f_(i,m)^e(t)
        self.action_dim = self.train_dim + self.agg_dim + self.res_dim

        # 修改为复合动作空间
        self.action_space = spaces.Tuple([
            # 离散部分：44维
            spaces.MultiDiscrete([2] * self.train_dim + [2] * self.agg_dim),
            # 连续部分：30维
            spaces.Box(low=0, high=1, shape=(self.N, self.M), dtype=np.float32)
        ])

        # 保留原始的扁平化维度信息用于调试
        discrete_dim = self.train_dim + self.agg_dim  # 44维
        continuous_dim = self.res_dim  # 30维
        print(f"环境动作空间: 总维度={discrete_dim + continuous_dim}, 离散部分={discrete_dim}, 连续部分={continuous_dim}")
        
        # Lyapunov参数
        self.energy_max = 2500  # 所有设备统一阈值
        self.e_avg = self.energy_max  # 移除平均计算
        self.alpha = 0.5  # 延迟权重
        self.beta = 0.5  # 能耗权重
        #self.reward_bate = 20.0  # 奖励系数 (显著提高以平衡奖励信号)
        self.convergence_epsilon = 1e-3  # 新的收敛阈值: F(ωt) - F(ω*) <= ϵ
        self.best_loss = float('inf')    # 记录当前episode的最优损失
        
        # 修改为每个设备独立的队列
        self.queues = [LyapunovQueue(e_avg=self.energy_max) for _ in range(self.N)]  # 统一阈值
        
        # 模型参数大小 (固定为2MB，用于聚合阶段计算)
        self.model_sizes = np.ones(self.N) * 2 * 1024 * 1024
        
        # 为PDQN算法添加标准gym空间格式
        # 观察空间 - 连续状态空间
        self.observation_space = spaces.Box(
            low=np.float32(-np.inf), 
            high=np.float32(np.inf),
            shape=(self.state_dim,),
            dtype=np.float32
        )
        print(f"设置observation_space.shape = {self.observation_space.shape}")
        
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
        
        # 环境重置
        # self.reset()
        
        # 显式声明env属性，避免类型检查器报错
        self.env = None

        # 新增：为约束违反创建一个日志文件
        self.invalid_log_file = "logs/invalid_actions.log"
        if not os.path.exists("logs"):
            os.makedirs("logs")
        with open(self.invalid_log_file, 'w') as f:
            f.write(f"===== Invalid Action Log - New Experiment Started at {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")


    #无线通道模型，计算传输速率
    def calculate_rate(self, transmit_power, gn, N0):
        """计算传输速率（单位：bps）"""
        rate = self.B * 1e6 * np.log2(1 + transmit_power * gn/N0)  # 香农公式，MHZ转换为Hz
        return rate  # 单位：bps
    
    def reset(self):
        """重置环境时初始化所有队列"""
        # 初始化每个设备的队列
        self.queues = [LyapunovQueue(e_avg=self.energy_max) for _ in range(self.N)]
        
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
        
        # 重置边缘节点资源分配
        self.f_e = np.full(self.M, self.f_e_min)  # 重置为最小值
        
        # 使用瑞利衰落模型生成信道增益
        # 瑞利分布参数scale=1.0表示平均信道增益为1
        # 上行信道增益 h_up: (N,M) 表示N个设备到M个边缘服务器的信道增益
        self.h_up = np.random.rayleigh(scale=1.0, size=(self.N, self.M))
        
        # 下行信道增益 h_down: (N,M) 表示M个边缘服务器到N个设备的信道增益
        self.h_down = np.random.rayleigh(scale=1.0, size=(self.N, self.M))
        
        # 重置李雅普诺夫能量队列
        for i in range(self.N):
            self.queues[i] = LyapunovQueue(e_avg=self.energy_max)
        
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
        """构建符合MEC联邦学习需求的状态空间"""
        global_state = []
            
        for i in range(self.N):
            # 初始化队列积压和剩余能量
            queue_backlogs = np.zeros(self.N)
            remaining_energies = np.zeros(self.N)
            
            # 计算上下行速率
            for j in range(self.M):
                self.R_up[i][j] = self.calculate_rate(
                    self.Pt_UP,
                    self.h_up[i][j],
                    self.N0
                )
                
                self.R_down[i][j] = self.calculate_rate(
                    self.Pt_down,
                    self.h_down[i][j],
                    self.N0
                )

            # 使用设备i的队列积压
            if i < len(self.queues):
                queue_backlogs[i] = self.queues[i].queue / self.energy_max
            
            # 计算设备i的剩余能量
            if i < len(self.queues):
                remaining_energies[i] = max(0, self.energy_max - self.queues[i].queue)
            
            # 基本状态 - 5个元素/设备，总共5*N个元素
            device_state = [
                float(self.data_sizes[i]) / (100.0 * 1024 * 1024),  # 数据大小(归一化)
                float(self.model_sizes[i]) / (10.0 * 1024 * 1024),  # 模型大小(归一化)
                float(queue_backlogs[i]),                           # 队列积压
                float(self.f_l[i]) / 3e9 if isinstance(self.f_l, np.ndarray) and i < len(self.f_l) else 0.0,  # 设备CPU频率(归一化)
                float(remaining_energies[i]) / self.energy_max      # 剩余能量比例
            ]
            
            global_state.extend(device_state)
            
            # 添加通信速率 - 每个设备2*M个元素
            for j in range(self.M):
                global_state.append(self.R_up[i][j] / 1e8)   # 上行速率(归一化)
                global_state.append(self.R_down[i][j] / 1e8) # 下行速率(归一化)
        
        # 添加边缘到云的速率 - M个元素
        for j in range(self.M):
            global_state.append(self.rate_CU / 200.0)  # 边缘到云速率(归一化)
            global_state.append(self.rate_CD / 200.0)  # 云到边缘速率(归一化)
        
        # 添加边缘节点CPU频率 - M个元素
        for j in range(self.M):
            # 添加边缘节点当前分配的计算资源
            global_state.append(self.f_e[j] / 4.5e9)   # 当前分配的计算资源(归一化)
        
        state = np.array(global_state, dtype=np.float32)
        print(f"状态向量长度: {len(state)}")
        
        return state
    
    def _calculate_train_communication(self, device_idx, edge_idx):
        """
        计算训练阶段的通信延迟和能耗 (从终端设备到训练节点)
        仅当训练在边缘节点时才有通信开销
        """
        # 使用数据大小 data_sizes 计算 - 数据集大小
        data_size = self.data_sizes[device_idx]
        
        # 调用calculate_rate函数计算上行传输速率
        rate_up = self.calculate_rate(self.Pt_UP, self.h_up[device_idx][edge_idx], self.N0)
        delay_up = data_size / rate_up
        energy_up = self.Pt_UP * delay_up
        
        return delay_up, energy_up

    def _calculate_train_computation(self, device_idx, edge_idx=None, is_local=True, freq=None):
        """
        计算训练阶段的计算延迟和能耗
        :param device_idx: 设备索引
        :param edge_idx: 边缘节点索引
        :param is_local: 是否本地计算
        :param freq: 指定的计算频率 (Hz)，如果为None则使用当前分配的频率
        :return: 延迟, 能耗
        """
        data_size = self.data_sizes[device_idx]
        cycles = float(self.c[device_idx]) if isinstance(self.c, np.ndarray) and device_idx < len(self.c) else 0.0  # cycles/bit
        
        if is_local:
            # 本地计算 - 使用正确的公式
            # 使用指定的频率，或者默认使用DRL分配的终端设备频率
            if freq is not None:
                compute_freq = float(freq)
            elif isinstance(self.f_l, np.ndarray) and device_idx < len(self.f_l):
                compute_freq = float(self.f_l[device_idx])
            else:
                compute_freq = self.f_l_min
                
            # 确保频率在有效范围内
            compute_freq = max(self.f_l_min, min(self.f_l_max, compute_freq))
            
            # 计算延迟和能耗
            delay = data_size * cycles / compute_freq
            # 修正：任务在边缘节点上的计算能耗，是整个任务成本的一部分，必须计入
            energy = 1e-27 * (compute_freq)**3 * delay
        else:
            # 边缘计算
            # 使用指定的频率，或者默认使用DRL分配的边缘节点频率
            if freq is not None:
                compute_freq = float(freq)
            elif edge_idx is not None and isinstance(self.f_e, np.ndarray) and edge_idx < len(self.f_e):
                compute_freq = float(self.f_e[edge_idx])
            else:
                compute_freq = self.f_e_min
                
            # 确保频率在有效范围内
            compute_freq = max(self.f_e_min, min(self.f_e_max, compute_freq))
            
            # 计算延迟和能耗
            delay = data_size * cycles / compute_freq
            # 修正：任务在边缘节点上的计算能耗，是整个任务成本的一部分，必须计入
            energy = 1e-27 * (compute_freq)**3 * delay
        
        return delay, energy
    
    def _calculate_aggregate_upload(self, device_idx, train_edge_idx, agg_edge_idx, is_local_training, is_cloud_agg):
        """
        计算聚合阶段的上传延迟和能耗 (从训练节点到聚合节点)
        
        Returns:
            (delay, energy): 上传延迟和能耗
        """
        # 使用模型参数大小计算 - 这是FL任务训练产生的模型参数大小
        model_size = self.model_sizes[device_idx]
        
        if is_local_training:
            # 本地训练 → 上传到聚合节点
            if is_cloud_agg:
                # 场景2: 本地 → 边缘 → 云
                # 步骤1: 终端设备到边缘节点
                rate_up = self.calculate_rate(self.Pt_UP, self.h_up[device_idx][agg_edge_idx], self.N0)
                delay_to_edge = model_size / rate_up
                energy_to_edge = self.Pt_UP * delay_to_edge
                
                # 步骤2: 边缘节点到云端
                delay_to_cloud = model_size / (self.rate_CU * 1e6)  # 转换为bps
                # 边缘到云的能耗不计入终端设备
                
                return delay_to_edge + delay_to_cloud, energy_to_edge
            else:
                # 场景1: 本地 → 边缘
                rate_up = self.calculate_rate(self.Pt_UP, self.h_up[device_idx][agg_edge_idx], self.N0)
                delay = model_size / rate_up
                energy = self.Pt_UP * delay
                
                return delay, energy
        else:
            # 边缘训练 → 上传到聚合节点
            if is_cloud_agg:
                # 场景4: 边缘 → 云
                delay = model_size / (self.rate_CU * 1e6)  # 转换为bps
                # 边缘到云的能耗计入任务总成本
                energy = self.Pt_edge_transmit * delay
                return delay, energy
            else:
                # 场景3: 边缘 → 边缘
                if train_edge_idx == agg_edge_idx:
                    # 场景3a: 同一个边缘节点，无需传输
                    return 0.0, 0.0
                else:
                    # 场景3b: 不同边缘节点之间的传输
                    edge_to_edge_rate = 1000 * 1e6  # 1000Mbps = 1Gbps
                    delay = model_size / edge_to_edge_rate
                    # 边缘间传输能耗计入任务总成本
                    energy = self.Pt_edge_transmit * delay
                    return delay, energy

    def _calculate_aggregate_feedback(self, device_idx, agg_edge_idx, is_cloud_agg):
        """
        计算聚合阶段的反馈延迟和能耗 (从聚合节点到终端设备)
        
        在三层架构中，全局模型的下发路径有以下几种情况：
        1. 边缘聚合 → 终端设备: 边缘节点直接将全局模型下发到终端设备
        2. 云端聚合 → 终端设备: 云端先将全局模型下发到边缘节点，再由边缘节点转发到终端设备
        
        Args:
            device_idx: 终端设备索引
            agg_edge_idx: 聚合的边缘节点索引（如果是边缘聚合或作为云聚合的中继）
            is_cloud_agg: 是否为云端聚合
            
        Returns:
            (delay, energy): 反馈延迟和能耗
        """
        # 使用模型参数大小计算 - 这是全局聚合后下发的模型
        model_size = self.model_sizes[device_idx]
        
        if is_cloud_agg:
            # 场景2: 云 → 边缘 → 终端
            # 步骤1: 云到边缘
            delay_cloud_to_edge = model_size / (self.rate_CD * 1e6)  # 转换为bps
            # 云端能耗，使用独立的下行功率
            energy_cloud_to_edge = self.Pt_cloud_down * delay_cloud_to_edge
            
            # 步骤2: 边缘到终端
            rate_down = self.calculate_rate(self.Pt_down, self.h_down[device_idx][agg_edge_idx], self.N0)
            delay_edge_to_device = model_size / rate_down
            # 边缘节点下行能耗
            energy_edge_to_device = self.Pt_down * delay_edge_to_device
            
            total_delay = delay_cloud_to_edge + delay_edge_to_device
            total_infra_energy = energy_cloud_to_edge + energy_edge_to_device
            return total_delay, total_infra_energy
        else:
            # 场景1: 边缘 → 终端
            rate_down = self.calculate_rate(self.Pt_down, self.h_down[device_idx][agg_edge_idx], self.N0)
            delay = model_size / rate_down
            # 边缘节点下行能耗
            energy = self.Pt_down * delay
            
            return delay, energy

    def _meets_constraints(self, train_local_decisions, edge_train_matrix, edge_agg_decision, 
                          cloud_agg_decision, res_alloc_matrix):
        
        penalty = 0.0
        invalid_reasons = []
        
        # 约束1: 训练位置唯一性 - 每个设备必须且只能在一个位置训练
        for i in range(self.N):
            local_decision = train_local_decisions[i]
            edge_sum = np.sum(edge_train_matrix[i])
            # 检查是否未分配训练位置
            if (local_decision + edge_sum) == 0:
                penalty += 800.0  # 未分配训练位置惩罚
                invalid_reasons.append(f"设备{i}未被分配训练位置")
            
            # 检查是否分配了多个训练位置
            if (local_decision + edge_sum) > 1:
                penalty += 300.0  # 相对较低的惩罚
                invalid_reasons.append(f"设备{i}被分配到多个训练位置")
           

        # 约束2: 聚合位置唯一性 - 必须且只能选择一个聚合位置
        edge_agg_sum = np.sum(edge_agg_decision)
        if (edge_agg_sum + cloud_agg_decision) != 1:
            penalty += 300.0  # 聚合位置冲突/缺失惩罚
            invalid_reasons.append("聚合位置约束违反")

        # 如果选择了多个边缘节点进行聚合
        if edge_agg_sum > 1:
            penalty += 150.0 * (edge_agg_sum - 1)  # 多选聚合节点惩罚

        # 约束3: 边缘节点资源约束 - 资源分配不能超过上限
        for j in range(self.M):
            total_alloc = 0.0
            for i in range(self.N):
                if edge_train_matrix[i, j] == 1:
                    # 安全地计算分配的资源
                    alloc_ratio = min(max(res_alloc_matrix[i, j], 0), 1)
                    actual_freq = self.f_e_min + alloc_ratio * (self.f_e_max - self.f_e_min)
                    total_alloc += actual_freq
            
            if total_alloc > self.F_e[j]:
                # 资源超限惩罚
                overuse_ratio = total_alloc / self.F_e[j] - 1
                penalty += 150.0 + 300.0 * overuse_ratio
                invalid_reasons.append(f"边缘节点{j}资源分配超限: {total_alloc/1e9:.2f}GHz > {self.F_e[j]/1e9:.2f}GHz")
        
        return penalty, invalid_reasons

    def parse_action(self, action):
        """
        解析扁平化的动作数组为离散和连续部分
        
        Args:
            action: 扁平化的动作数组
        
        Returns:
            discrete_action: 离散动作部分
            continuous_action: 连续动作部分
        """
        # 确保动作是numpy数组
        if not isinstance(action, np.ndarray):
            action = np.array(action)
        
        # 分割动作数组
        discrete_dim = self.train_dim + self.agg_dim
        continuous_dim = self.res_dim
        
        # 确保动作维度正确
        if len(action) < discrete_dim + continuous_dim:
            # 如果动作维度不足，使用默认值填充
            full_action = np.zeros(discrete_dim + continuous_dim)
            full_action[:len(action)] = action
            action = full_action
        
        # 分离离散和连续部分
        discrete_action = action[:discrete_dim]
        continuous_action = action[discrete_dim:discrete_dim + continuous_dim]
        
        return discrete_action, continuous_action

    def _parse_action_for_training(self, action, clients_manager=None, server=None):
        """
        将DRL动作解析为训练决策、聚合决策和资源分配
        这是一个内部方法，被step和get_fl_training_params共同使用
        
        Args:
            action: DRL智能体输出的动作向量
            clients_manager: 客户端管理器实例（可选）
            server: 服务器实例（可选）
            
        Returns:
            parsed_results: 包含解析结果的字典
        """
        # 定义常用常量
        DEFAULT_F_L_MIN = 4e8    # 最小CPU频率 (0.4GHz)
        DEFAULT_F_L_MAX = 2.9e9  # 最大CPU频率 (2.9GHz)
        DEFAULT_F_E_MIN = 2.9e9  # 最小CPU频率 (2.9GHz)
        DEFAULT_F_E_MAX = 4.3e9  # 最大CPU频率 (4.3GHz)
        
        # 获取可用的终端设备和边缘节点（如果提供了clients_manager）
        available_clients = []
        available_edges = []
        
        if clients_manager:
            available_clients = [
                cid for cid in clients_manager.clients 
                if clients_manager.clients[cid].available and cid.startswith("client")]
            
            available_edges = [
                cid for cid in clients_manager.clients
                if clients_manager.clients[cid].available and cid.startswith("edge")]
        
        # 解析动作
        discrete_action, continuous_action = self.parse_action(action)
        
        # 1. 解析训练决策
        # 1.1 解析本地训练决策
        train_local_decisions = discrete_action[:min(self.N, len(discrete_action))]  # x_i^l(t)
        
        # 1.2 解析边缘训练决策
        edge_train_offset = self.N
        edge_train_decisions = discrete_action[edge_train_offset:edge_train_offset + self.N * self.M]
        
        # 1.3 解析聚合决策
        edge_agg_offset = edge_train_offset + self.N * self.M
        edge_agg_decisions = discrete_action[edge_agg_offset:edge_agg_offset + self.M]
        
        cloud_agg_offset = edge_agg_offset + self.M
        cloud_agg_decisions = discrete_action[cloud_agg_offset:cloud_agg_offset + 1]
        cloud_agg_decision = cloud_agg_decisions[0] if len(cloud_agg_decisions) > 0 else 0
        
        # 1.4 解析资源分配决策
        res_alloc_flat = continuous_action
        
        # 返回解析结果
        return {
            'train_local_decisions': train_local_decisions,
            'edge_train_decisions': edge_train_decisions,
            'edge_agg_decisions': edge_agg_decisions,
            'cloud_agg_decision': cloud_agg_decision,
            'res_alloc_flat': res_alloc_flat,
            'discrete_action': discrete_action,
            'continuous_action': continuous_action,
            'available_clients': available_clients,
            'available_edges': available_edges
        }

    def step(self, action, fl_loss=None):
        """
        执行一步动作 - 对应一个FL轮次
        :param action: 扁平化的动作数组
        :param fl_loss: 从服务器传来的全局损失，用于判断收敛
        :return: 新状态，奖励，是否结束当前Episode，附加信息
        """
        # 初始化信息字典
        info = {
            "total_delay": 0.0,
            "total_energy": 0.0, 
            "total_cost": 0.0,
            "valid_ratio": 0.0
        }
        
        # 解析动作
        parsed_action = self._parse_action_for_training(action)
        
        # 提取解析结果
        train_local_decisions = parsed_action['train_local_decisions']
        edge_train_decisions = parsed_action['edge_train_decisions']
        edge_agg_decisions = parsed_action['edge_agg_decisions']
        cloud_agg_decision = parsed_action['cloud_agg_decision']
        res_alloc_flat = parsed_action['res_alloc_flat']
        
        # 环境随机变化
        self._randomize_environment()
        
        # 将训练决策转换为矩阵形式
        edge_train_matrix = np.zeros((self.N, self.M), dtype=int)
        valid_indices = min(len(edge_train_decisions), self.N * self.M)

        for idx in range(valid_indices):
            i = idx // self.M  # 行索引
            j = idx % self.M   # 列索引
            edge_train_matrix[i, j] = 1 if edge_train_decisions[idx] > 0.5 else 0
        
        # 将资源分配转换为矩阵形式
        res_alloc_matrix = np.zeros((self.N, self.M))
        flat_idx = 0
        for i in range(self.N):
            for j in range(self.M):
                if flat_idx < len(res_alloc_flat):
                    res_alloc_matrix[i, j] = res_alloc_flat[flat_idx]
                else:
                    res_alloc_matrix[i, j] = 0.5  # 默认值
                flat_idx += 1
        
        # 2. 检查约束
        # 将所有离散决策转换为0/1二元值再传入约束检查
        binary_train_local = (train_local_decisions > 0.5).astype(int)
        binary_edge_agg = (edge_agg_decisions > 0.5).astype(int)
        binary_cloud_agg = 1 if cloud_agg_decision > 0.5 else 0
        constraint_penalty, invalid_reasons = self._meets_constraints(
            binary_train_local, 
            edge_train_matrix,
            binary_edge_agg,
            binary_cloud_agg,
            res_alloc_matrix
        )
        
        # 新增：如果存在约束违反，则记录日志
        if invalid_reasons:
            with open(self.invalid_log_file, 'a') as f:
                log_entry = (
                    f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] "
                    f"Episode: {self.current_episode}, FL Round: {self.current_fl_round}\n"
                )
                log_entry += "  Reasons for invalid action:\n"
                for reason in invalid_reasons:
                    log_entry += f"    - {reason}\n"
                f.write(log_entry + "\n")
        


        # 3. 计算延迟和能耗
        delays, energies, costs = [], [], []
        device_energies_for_queue = np.zeros(self.N) # 新增：用于更新队列
        valid_flags = []  # 记录各任务是否满足时隙约束
        
        # 为每个终端设备计算延迟和能耗
        for i in range(self.N):
            # 初始化分层成本变量，以保持计算结构
            train_comm_delay, train_comm_energy = 0.0, 0.0
            train_comp_delay, train_comp_energy = 0.0, 0.0
            agg_upload_delay, agg_upload_energy = 0.0, 0.0
            agg_feedback_delay, agg_feedback_energy = 0.0, 0.0

            # 获取设备i的决策变量 (1或0)
            is_local_train = 1 if i < len(train_local_decisions) and train_local_decisions[i] > 0.5 else 0
            edge_train_selections = (edge_train_matrix[i] > 0.5).astype(int) if i < edge_train_matrix.shape[0] else np.zeros(self.M, dtype=int)
            
            # --- 1. 累加训练成本 (遵循条件计算原则) ---
            # a. 如果选择本地训练(is_local_train=1)，累加其成本
            if is_local_train == 1:
                comp_d, comp_e = self._calculate_train_computation(i, is_local=True)
                train_comp_delay += comp_d
                train_comp_energy += comp_e
                device_energies_for_queue[i] += comp_e # 本地计算能耗计入设备

            # b. 循环所有边缘节点，如果选择在某边缘训练(selections[m]=1)，累加其成本
            selected_train_edges = [] # 用于后续聚合阶段
            for m in range(self.M):
                if edge_train_selections[m] == 1:
                    selected_train_edges.append(m)
                    # 通信成本
                    comm_d, comm_e = self._calculate_train_communication(i, m)
                    train_comm_delay += comm_d
                    train_comm_energy += comm_e
                    device_energies_for_queue[i] += comm_e # 设备上传能耗计入设备
                    # 计算成本
                    alloc = min(max(res_alloc_matrix[i, m], 0), 1)
                    # 计算分配给该任务的CPU频率
                    freq = self.f_e_min + alloc * (self.f_e_max - self.f_e_min)
                    comp_d, comp_e = self._calculate_train_computation(i, edge_idx=m, is_local=False, freq=freq)
                    train_comp_delay += comp_d
                    train_comp_energy += comp_e # 边缘计算能耗只计入任务总成本

            # --- 2. 累加聚合成本 (遵循条件计算原则) ---
            is_cloud_agg = 1 if cloud_agg_decision > 0.5 else 0
            selected_agg_edges = np.where(edge_agg_decisions > 0.5)[0]

            # a. 累加所有被选中的边缘聚合方案的成本
            for agg_edge_idx in selected_agg_edges:
                # 聚合上传成本
                if is_local_train == 1:
                    up_d, up_e = self._calculate_aggregate_upload(i, None, agg_edge_idx, True, False)
                    agg_upload_delay += up_d
                    agg_upload_energy += up_e
                    device_energies_for_queue[i] += up_e # 本地上传计入设备
                for train_edge_idx in selected_train_edges:
                    up_d, up_e = self._calculate_aggregate_upload(i, train_edge_idx, agg_edge_idx, False, False)
                    agg_upload_delay += up_d
                    agg_upload_energy += up_e
                # 聚合反馈成本
                fb_d, fb_e = self._calculate_aggregate_feedback(i, agg_edge_idx, False)
                agg_feedback_delay += fb_d
                agg_feedback_energy += fb_e

            # b. 如果选择云聚合(is_cloud_agg=1)，累加其成本
            if is_cloud_agg == 1:
                relay_edge_up = np.argmax(self.h_up[i])
                relay_edge_down = np.argmax(self.h_down[i])
                # 聚合上传成本
                if is_local_train == 1:
                    up_d, up_e = self._calculate_aggregate_upload(i, None, relay_edge_up, True, True)
                    agg_upload_delay += up_d
                    agg_upload_energy += up_e
                    device_energies_for_queue[i] += up_e # 本地上传计入设备
                for train_edge_idx in selected_train_edges:
                    up_d, up_e = self._calculate_aggregate_upload(i, train_edge_idx, None, False, True)
                    agg_upload_delay += up_d
                    agg_upload_energy += up_e
                # 聚合反馈成本
                fb_d, fb_e = self._calculate_aggregate_feedback(i, relay_edge_down, True)
                agg_feedback_delay += fb_d
                agg_feedback_energy += fb_e
            
            # --- 3. 最终汇总（保持原有分层逻辑） ---
            total_delay = train_comm_delay + train_comp_delay + agg_upload_delay + agg_feedback_delay
            total_energy = train_comm_energy + train_comp_energy + agg_upload_energy + agg_feedback_energy
            task_cost = self.alpha * total_delay + self.beta * total_energy
            
            # 记录历史数据
            self.delays_history[i, self.current_fl_round] = total_delay
            self.energies_history[i, self.current_fl_round] = total_energy
            self.costs_history[i, self.current_fl_round] = task_cost
            
            # 添加到列表
            delays.append(total_delay)
            energies.append(total_energy)
            costs.append(task_cost)
            valid_flags.append(total_delay <= self.time_slot)  # 检查是否满足时隙约束
        
        # 4. 更新队列并计算奖励
        total_reward = 0.0
        
        # 计算总延迟、能耗和成本
        total_delay = sum(delays)
        total_energy = sum(energies)
        total_cost = sum(costs)
        
        # 计算每个设备的奖励（基于成本和队列的加权和）
        for i in range(self.N):
            if i < len(energies):  # 确保索引有效
                Q_i_t = self.queues[i].queue
                device_reward = - (
                    0.2 * Q_i_t * device_energies_for_queue[i]   # 队列项 (使用设备自身能耗)
                  + 0.8 * costs[i]              # 成本项 (基于任务总能耗)
                )
                total_reward += device_reward
                self.queues[i].update_queue(device_energies_for_queue[i]) # 更新队列使用设备自身能耗
        
        # 应用最终的约束惩罚
        total_reward -= constraint_penalty
        
        # 更新状态
        next_state = self._get_state()
        self.current_fl_round += 1
        
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
        
        # 更新边缘节点的实际资源分配状态（仅用于状态表示）
        self.f_e = np.full(self.M, self.f_e_min)  # 重置为最小值
        
        # 根据本轮实际决策更新边缘节点资源分配
        for j in range(self.M):
            total_alloc = self.f_e_min  # 初始值为最小资源
            for i in range(self.N):
                if i < edge_train_matrix.shape[0] and j < edge_train_matrix.shape[1] and edge_train_matrix[i, j] == 1:
                    # 该设备的任务被分配到此边缘节点
                    if i < res_alloc_matrix.shape[0] and j < res_alloc_matrix.shape[1]:
                        alloc_ratio = min(max(res_alloc_matrix[i, j], 0), 1)
                        # 计算分配的资源并累加
                        freq = self.f_e_min + alloc_ratio * (self.f_e_max - self.f_e_min)
                        total_alloc += freq - self.f_e_min  # 累加分配的额外资源
            
            # 确保不超过最大值
            self.f_e[j] = min(total_alloc, self.F_e[j])
        
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
            
        # 更新数据集大小
        # if hasattr(clients_manager, 'get_data_sizes'):
        #     data_sizes = clients_manager.get_data_sizes()
        #     if data_sizes is not None:
        #         self.set_data_sizes(data_sizes)
                
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
        # 信道增益变化(瑞利衰落)
        # 使用Jakes模型的思想，通过添加随机相位来模拟时变特性
        # 保持一定的相关性，避免信道增益突变
        correlation_factor = 0.7  # 时间相关性因子
        
        # 生成新的瑞利衰落信道增益
        new_h_up = np.random.rayleigh(scale=1.0, size=(self.N, self.M))
        new_h_down = np.random.rayleigh(scale=1.0, size=(self.N, self.M))
        
        # 更新信道增益，保持时间相关性
        self.h_up = correlation_factor * self.h_up + (1 - correlation_factor) * new_h_up
        self.h_down = correlation_factor * self.h_down + (1 - correlation_factor) * new_h_down
        
        # 确保信道增益非负
        self.h_up = np.maximum(0.1, self.h_up)
        self.h_down = np.maximum(0.1, self.h_down)
        
    def get_fl_training_params(self, action, clients_manager, server):
        """
        将DRL动作直接转换为联邦学习训练参数
        
        Args:
            action: DRL智能体输出的动作向量
            clients_manager: 客户端管理器实例
            server: 服务器实例
            
        Returns:
            training_args: 联邦学习训练参数字典，包含:
                - selected_nodes: 选中的训练节点列表
                - resource_allocation: 资源分配字典
                - aggregation_location: 聚合位置
                - drl_train_decisions: 训练决策列表
        """
        # 定义常用常量
        DEFAULT_F_L_MIN = 4e8    # 最小CPU频率 (0.4GHz)
        DEFAULT_F_L_MAX = 2.9e9  # 最大CPU频率 (2.9GHz)
        DEFAULT_F_E_MIN = 2.9e9  # 最小CPU频率 (2.9GHz)
        DEFAULT_F_E_MAX = 4.3e9  # 最大CPU频率 (4.3GHz)
        
        # 解析动作
        parsed_action = self._parse_action_for_training(action, clients_manager, server)
        
        # 提取解析结果
        train_local_decisions = parsed_action['train_local_decisions']
        edge_train_decisions = parsed_action['edge_train_decisions']
        edge_agg_decisions = parsed_action['edge_agg_decisions']
        cloud_agg_decision = parsed_action['cloud_agg_decision']
        res_alloc_flat = parsed_action['res_alloc_flat']
        available_clients = parsed_action['available_clients']
        available_edges = parsed_action['available_edges']
    

        
        # 1. 将决策向量转换为与step()方法中相同的二进制格式和矩阵格式
        train_local_binary = (train_local_decisions > 0.5).astype(int)
        
        edge_train_matrix = np.zeros((self.N, self.M), dtype=int)
        edge_decisions_flat = (edge_train_decisions > 0.5).astype(int)
        
        valid_indices = min(len(edge_decisions_flat), self.N * self.M)
        for idx in range(valid_indices):
            i = idx // self.M
            j = idx % self.M
            edge_train_matrix[i, j] = edge_decisions_flat[idx]
            # except (ValueError, IndexError):
            #     # 如果客户端ID格式不正确，则跳过
        # 2. 基于统一的决策矩阵构建训练参数
        drl_train_decisions = []
        client_edge_mapping = {}
        selected_nodes = []
        selected_edges_set = set()

        for i in range(self.N):
            client_id = f"client{i}"
            if client_id not in available_clients:
                continue
        
     
            is_local = i < len(train_local_binary) and train_local_binary[i] == 1
            edge_selections = edge_train_matrix[i]
            
            # 约束检查在step()中进行，这里只负责执行
            # 如果本地训练和边缘训练同时为1，则优先本地（或根据特定规则）
            if is_local:
                selected_nodes.append(client_id)
                drl_train_decisions.append(1) # 1=本地训练
            
            elif np.sum(edge_selections) > 0:
                # 选择第一个被标记的边缘节点进行训练
                edge_idx = np.where(edge_selections == 1)[0][0]
                edge_id = f"edge_{edge_idx}"
                if edge_id in available_edges:
                    selected_nodes.append(client_id)
                    drl_train_decisions.append(0) # 0=边缘训练
                    client_edge_mapping[client_id] = edge_id
                    selected_edges_set.add(edge_id)
        
        # 3. 确定聚合位置 (逻辑保持不变)
        if cloud_agg_decision > 0.5:
            aggregation_location = "cloud"
        else:
            # DRL决策选择在边缘节点聚合
            # 选择得分最高的边缘节点
            selected_edge = None
            max_score = -1
            
            for j, edge_id in enumerate(available_edges):
                if j < len(edge_agg_decisions) and edge_agg_decisions[j] > max_score:
                    max_score = edge_agg_decisions[j]
                    selected_edge = edge_id
            
            # 如果有选中的边缘节点且其分数大于阈值，则在该边缘节点聚合
            if selected_edge is not None and max_score > 0.5:
                aggregation_location = selected_edge
            else:
                # 默认在云端聚合
                aggregation_location = "cloud"
        
        # 记录聚合决策信息（用于日志和调试）
        edge_agg_counts = {edge_id: edge_agg_decisions[j] if j < len(edge_agg_decisions) else 0 
                          for j, edge_id in enumerate(available_edges)}
        
        # 4. 处理资源分配
        client_resources = {}
        
        # 为终端设备分配资源（随机分配）
        for i, client_id in enumerate(available_clients):
            if i >= self.N:
                continue
                
            # 获取终端设备的资源范围
            f_min = self.f_l_min if hasattr(self, 'f_l_min') else DEFAULT_F_L_MIN
            f_max = self.f_l_max if hasattr(self, 'f_l_max') else DEFAULT_F_L_MAX
            
            # 在最小值和最大值之间随机分配资源
            import random
            client_resources[client_id] = f_min + random.random() * (f_max - f_min)
        
     
        # 5. 添加所有选定的边缘节点到训练节点列表
        selected_nodes.extend(selected_edges_set)
        
        # 尊重智能体的决策，如果没有任何客户端被选中，则返回空列表
        if not any(cid.startswith("client") for cid in selected_nodes):
            print("警告: get_fl_training_params - DRL智能体未选择任何终端设备，本轮将不进行训练。")
            selected_nodes = []
            drl_train_decisions = []
            client_edge_mapping = {}
            aggregation_location = "cloud"
        
        # 将客户端到边缘节点的映射信息保存到服务器中
        if hasattr(server, 'client_edge_mapping'):
            server.client_edge_mapping = client_edge_mapping
            
        # 保存聚合决策信息到服务器
        if hasattr(server, 'edge_agg_counts'):
            server.edge_agg_counts = edge_agg_counts
        
        if hasattr(server, 'cloud_agg_decisions'):
            server.cloud_agg_decisions = [cloud_agg_decision]
        
        # 6. 构建训练参数
        training_args = {
            'selected_nodes': selected_nodes,
            'resource_allocation': client_resources,
            'aggregation_location': aggregation_location,
            'drl_train_decisions': drl_train_decisions,
        }
        
        # 7. 构建原始决策字典
        raw_decisions = {
            'local_train': (train_local_decisions > 0.5).astype(int),
            'edge_train': (edge_train_decisions > 0.5).astype(int),
            'edge_agg': (edge_agg_decisions > 0.5).astype(int),
            'cloud_agg': (cloud_agg_decision > 0.5).astype(int),
            'resource_alloc': res_alloc_flat # 资源分配保持原始浮点值
        }
        
        return training_args, raw_decisions
        
 
        
