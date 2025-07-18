import numpy as np
import torch
import math
import time
import gym
from gym import spaces
from utils.data_utils import get_dataset

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
    def __init__(self, C=10, gama=0.01, delta=1.0):
        # 环境参数
        self.C = C          # 全局迭代次数参数
        self.gama = gama    # 全局精确度
        self.delta = delta  # 模型参数大小比例
        
        # 系统参数
        self.N = 10        # 终端设备数量
        self.M = 3         # 边缘服务器数量
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
        self.fl_rounds_per_episode = 10  # 每个DRL Episode包含的FL轮次数量
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
        self.data_sizes = np.ones(self.N) * 50 * 1024 * 1024  # 50MB

        # 通信模型
        self.B = 6  # 信道带宽 (MHz)
        self.Pt_UP = 24    # 上行传输功率 (dBM)
        self.Pt_down = 30  # 下行传输功率 (dBM)
        self.power_mcU = 35 # 边缘到云上行功率
        self.power_mcD = 45 # 边缘到云下行功率
        self.N0 = 10e-13   # 噪声功率
        self.rate_CU = 120  # 边缘到云上行速率 (Mbps)
        self.rate_CD = 150  # 边缘到云下行速率 (Mbps)

        # 计算模型
        # 终端设备计算资源 - 随机分配固定值，不再由DRL决策
        self.f_l_min = 4e8    # 最小CPU频率 (0.4GHz)
        self.f_l_max = 2.9e9  # 最大CPU频率 (2.9GHz)
        # 随机分配终端设备的计算资源，且保持固定
        self.f_l = np.random.uniform(self.f_l_min, self.f_l_max, size=self.N)
        self.F_l = np.copy(self.f_l)  # 保存初始分配值
        
        # 边缘节点计算资源
        self.f_e_min = 2.9e9  # 最小CPU频率 (2.9GHz)
        self.f_e_max = 4.3e9  # 最大CPU频率 (4.3GHz)
        self.f_e = np.full(self.M, self.f_e_min)
        self.F_e = np.full(self.M, self.f_e_max)
        
        # 计算复杂度 (cycles/bit)
        self.c = np.random.uniform(300, 500, size=self.N)

        # 时间模型
        self.time_slot = 0.1

        # 状态空间
        # 每个终端设备状态: 5个基本状态 + M个上行速率 + M个下行速率
        # 每个边缘节点状态: 1个计算资源 + 2个与云的传输速率
        self.state_dim = self.N*(5 + 2*self.M) + self.M*3  # 设备状态 + 边缘状态
          

        # 动作空间：
        # 1. 训练决策: 
        #    - x_i^l(t): N个二元变量，表示设备i是否在本地训练
        #    - y_(i,m)^e(t): N*M个二元变量，表示设备i是否在边缘节点m上训练
        # 2. 聚合决策:
        #    - z_(i,m)^e(t): N*M个二元变量，表示设备i的更新是否在边缘节点m上聚合
        #    - w_i^c(t): N个二元变量，表示设备i的更新是否在云端聚合
        # 3. 资源分配:
        #    - f_(i,m)^e(t): N*M个连续变量，表示为边缘节点m上的设备i分配的计算资源

        # 计算动作空间维度
        self.train_dim = self.N + self.N * self.M  # x_i^l(t) + y_(i,m)^e(t)
        self.agg_dim = self.N * self.M + self.N    # z_(i,m)^e(t) + w_i^c(t)
        self.res_dim = self.N * self.M             # f_(i,m)^e(t)
        self.action_dim = self.train_dim + self.agg_dim + self.res_dim

        # Lyapunov参数
        self.energy_max = np.random.uniform(low=2500, high=3000, size=self.N)  # 每个设备的能耗阈值
        self.e_avg = np.mean(self.energy_max)  # 平均能量消耗阈值
        self.alpha = 0.5  # 延迟权重
        self.beta = 0.5  # 能耗权重
        self.reward_bate = 0.1  # 奖励系数
        
        # 修改为每个设备独立的队列
        self.queues = [LyapunovQueue(e_avg=self.energy_max[i]) for i in range(self.N)]  # 使用设备自身的能耗阈值
        
        # 模型参数大小 (通常比数据大小小，这里用delta作为比例因子)
        self.model_sizes = np.array([self.delta * ds for ds in self.data_sizes])
        
        # 为PDQN算法添加标准gym空间格式
        # 观察空间 - 连续状态空间
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf,
            shape=(self.state_dim,),
            dtype=np.float32
        )
        
        # 动作空间 - 混合动作空间
        # 离散动作部分：训练决策(N+N*M)和聚合决策(N*M+N)
        discrete_dim = self.train_dim + self.agg_dim
        
        # 连续动作部分：资源分配(N*M)
        continuous_shape = (self.res_dim,)
        
        # 将Dict格式改为Tuple格式
        self.action_space = spaces.Tuple([
            spaces.MultiDiscrete([2] * discrete_dim),  # 离散动作部分 - 训练和聚合决策
            spaces.Box(                                # 连续动作部分 - 资源分配
                low=0.0, 
                high=1.0, 
                shape=continuous_shape,
                dtype=np.float32
            )
        ])
        
        # 打印动作空间维度信息，方便调试
        print(f"环境动作空间: 离散部分维度={discrete_dim}, 连续部分维度={self.res_dim}")
        print(f"训练决策维度: N={self.N}, 边缘训练维度: N*M={self.N*self.M}")
        print(f"聚合决策维度: 边缘聚合={self.N*self.M}, 云聚合={self.N}")
        
        # 定义max_episodes属性供DRL算法使用
        self.max_episodes = 0
        
        # 添加调试信息属性
        self.debug_info = {
            'step_count': 0,
            'reward_history': []
        }
        
        # 环境重置
        self.reset()


    #无线通道模型，计算传输速率
    def calculate_rate(self, transmit_power, gn, N0):
        """计算传输速率（单位：bps）"""
        rate = self.B * 1e6 * np.log2(1 + transmit_power * gn/N0)  # 香农公式，MHZ转换为Hz
        return rate  # 单位：bps
    
    def set_data_sizes(self, data_sizes):
        """设置终端设备的数据集大小"""
        self.data_sizes = np.array(data_sizes)
        # 更新模型参数大小
        self.model_sizes = np.array([self.delta * ds for ds in self.data_sizes])

    def reset(self):
        """重置环境状态 - 开始新的DRL Episode"""
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
        
        # 重置计算资源
        self.f_l = np.random.uniform(self.f_l_min, self.f_l_max, size=self.N)
        self.f_e = np.full(self.M, self.f_e_min)
        
        # 使用瑞利衰落模型生成信道增益
        # 瑞利分布参数scale=1.0表示平均信道增益为1
        # 上行信道增益 h_up: (N,M) 表示N个设备到M个边缘服务器的信道增益
        self.h_up = np.random.rayleigh(scale=1.0, size=(self.N, self.M))
        
        # 下行信道增益 h_down: (N,M) 表示M个边缘服务器到N个设备的信道增益
        self.h_down = np.random.rayleigh(scale=1.0, size=(self.N, self.M))
        
        # 重置李雅普诺夫能量队列
        for i in range(self.N):
            self.queues[i] = LyapunovQueue(e_avg=self.energy_max[i])
        
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
        """构造状态向量: 包括终端设备的FL任务数据大小、更新大小、边缘节点的可用计算资源、
        不同层之间的传输速率、设备能量队列积压以及终端设备的剩余能量"""
        state = []
            
        # 终端设备状态: [数据大小, 更新大小, 能量队列积压, 可用计算资源, 剩余能量]
        for i in range(self.N):
            best_edge = np.argmax(self.h_up[i])  # 选择信道增益最大的边缘服务器
            
            # 计算当前更新大小 (与模型参数大小和训练轮次相关)
            # 随着训练轮次增加，更新大小会逐渐减小，反映模型收敛
            progress_factor = max(0.05, 0.15 - 0.001 * self.current_fl_round)  # 随训练轮次减小
            update_size = self.model_sizes[i] * (progress_factor + 0.01 * np.random.rand())
            
            # 剩余能量 (随时间衰减)
            remaining_energy = max(0, self.energy_max[i] - self.queues[i].queue)
            
            # 基本状态
            device_state = [
                self.data_sizes[i] / (100.0 * 1024 * 1024),  # 任务数据大小 (归一化)
                update_size / (10.0 * 1024 * 1024),          # 更新大小 (归一化)
                self.queues[i].queue / self.energy_max[i],              # 能量队列积压 (归一化)
                self.F_l[i] / 3e9,                           # 可用计算资源 (归一化)
                remaining_energy / self.energy_max[i]         # 剩余能量比例
            ]
            
            # 为了保持状态空间维度不变，添加占位值
            # 在实际决策中，我们会直接调用calculate_rate函数计算速率
            # 添加2*M个占位值，对应上行和下行速率
            device_state.extend([0.0] * (2 * self.M))
            state.extend(device_state)
            
        # 边缘节点状态: [可用计算资源, 与云节点之间的上行和下行传输速率]
        for m in range(self.M):
            # 使用已定义的速率常量，避免重复计算
            # 在实际使用时直接访问这些常量
            edge_state = [
                self.F_e[m] / 4.5e9,        # 可用计算资源 (归一化)
                self.rate_CU / 150.0,       # 边缘到云上行传输速率 (归一化)
                self.rate_CD / 200.0        # 云到边缘下行传输速率 (归一化)
            ]
            state.extend(edge_state)
            
        return np.array(state, dtype=np.float32)
    
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
        cycles = self.c[device_idx]  # cycles/bit
        
        if is_local:
            # 本地计算 - 使用正确的公式
            # 使用指定的频率，或者默认使用DRL分配的终端设备频率
            compute_freq = freq if freq is not None else self.f_l[device_idx]
            # 确保频率在有效范围内
            compute_freq = max(self.f_l_min, min(self.f_l_max, compute_freq))
            
            # 计算延迟和能耗
            delay = data_size * cycles / compute_freq
            energy = 1e-27 * (compute_freq)**3 * delay
        else:
            # 边缘计算
            # 使用指定的频率，或者默认使用DRL分配的边缘节点频率
            compute_freq = freq if freq is not None else self.f_e[edge_idx]
            # 确保频率在有效范围内
            compute_freq = max(self.f_e_min, min(self.f_e_max, compute_freq))
            
            # 计算延迟和能耗
            delay = data_size * cycles / compute_freq
            energy = 0.0  # 边缘节点能耗不计入终端设备
        
        return delay, energy
    
    def _calculate_aggregate_upload(self, device_idx, train_edge_idx, agg_edge_idx, is_local_training, is_cloud_agg):
        """
        计算聚合阶段的上传延迟和能耗 (从训练节点到聚合节点)
        
        在三层架构中，模型参数的上传路径有以下几种情况：
        1. 本地训练 → 边缘聚合: 终端设备直接将模型参数上传到指定的边缘节点
        2. 本地训练 → 云端聚合: 终端设备先将模型参数上传到就近边缘节点，再由边缘节点转发到云端
        3. 边缘训练 → 边缘聚合: 
           a. 如果训练和聚合是同一边缘节点，无需传输
           b. 如果是不同边缘节点，需要边缘节点间传输
        4. 边缘训练 → 云端聚合: 训练的边缘节点直接将模型参数上传到云端
        
        Args:
            device_idx: 终端设备索引
            train_edge_idx: 训练的边缘节点索引（如果是边缘训练）
            agg_edge_idx: 聚合的边缘节点索引（如果是边缘聚合或作为云聚合的中继）
            is_local_training: 是否为本地训练
            is_cloud_agg: 是否为云端聚合
            
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
                # 边缘到云的能耗不计入终端设备
                return delay, 0.0
            else:
                # 场景3: 边缘 → 边缘
                if train_edge_idx == agg_edge_idx:
                    # 场景3a: 同一个边缘节点，无需传输
                    return 0.0, 0.0
                else:
                    # 场景3b: 不同边缘节点之间的传输
                    # 假设边缘节点之间通过高速专用网络连接
                    edge_to_edge_rate = 1000 * 1e6  # 1000Mbps = 1Gbps
                    delay = model_size / edge_to_edge_rate
                    # 边缘间传输能耗不计入终端设备
                    return delay, 0.0

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
            
            # 步骤2: 边缘到终端
            rate_down = self.calculate_rate(self.Pt_down, self.h_down[device_idx][agg_edge_idx], self.N0)
            delay_edge_to_device = model_size / rate_down
            # 下行通信能耗由基站/边缘节点承担，不计入终端设备能耗
            energy_edge_to_device = 0.0
            
            return delay_cloud_to_edge + delay_edge_to_device, energy_edge_to_device
        else:
            # 场景1: 边缘 → 终端
            rate_down = self.calculate_rate(self.Pt_down, self.h_down[device_idx][agg_edge_idx], self.N0)
            delay = model_size / rate_down
            # 下行通信能耗由基站/边缘节点承担，不计入终端设备能耗
            energy = 0.0
            
            return delay, energy

    def step(self, action):
        """
        执行一步动作 - 对应一个FL轮次
        :param action: 动作向量，Tuple格式[离散动作, 连续动作]
        :return: 新状态，奖励，是否结束当前Episode，附加信息
        """
        # 更新调试信息
        self.debug_info['step_count'] += 1
        
        # 解析动作，预期为Tuple格式(离散部分, 连续部分)
        try:
            if isinstance(action, tuple) and len(action) == 2:
                discrete_action, continuous_action = action
            else:
                # 如果不是Tuple格式，使用默认分割
                discrete_dim = self.train_dim + self.agg_dim
                discrete_action = action[:discrete_dim]
                continuous_action = action[discrete_dim:discrete_dim + self.res_dim]
            
            # 1. 解析动作
            # 1.1 解析训练决策
            train_local_decisions = (discrete_action[:self.N] > 0.5).astype(int)  # x_i^l(t)
            
            # 1.2 解析边缘训练决策
            edge_train_offset = self.N
            edge_train_decisions = discrete_action[edge_train_offset:edge_train_offset + self.N * self.M]
            # 安全地reshape边缘训练决策
            edge_train_matrix = np.zeros((self.N, self.M), dtype=int)
            for i in range(self.N):
                for j in range(self.M):
                    idx = i * self.M + j
                    if idx < len(edge_train_decisions):
                        edge_train_matrix[i, j] = 1 if edge_train_decisions[idx] > 0.5 else 0
            
            # 1.3 解析聚合决策
            edge_agg_offset = edge_train_offset + self.N * self.M
            edge_agg_decisions = discrete_action[edge_agg_offset:edge_agg_offset + self.N * self.M]
            # 安全地reshape边缘聚合决策
            edge_agg_matrix = np.zeros((self.N, self.M), dtype=int)
            for i in range(self.N):
                for j in range(self.M):
                    idx = i * self.M + j
                    if idx < len(edge_agg_decisions):
                        edge_agg_matrix[i, j] = 1 if edge_agg_decisions[idx] > 0.5 else 0
            
            cloud_agg_offset = edge_agg_offset + self.N * self.M
            # 确保索引不超出范围
            cloud_agg_end = min(cloud_agg_offset + self.N, len(discrete_action))
            cloud_agg_decisions = np.zeros(self.N, dtype=int)
            for i in range(self.N):
                if cloud_agg_offset + i < cloud_agg_end:
                    cloud_agg_decisions[i] = 1 if discrete_action[cloud_agg_offset + i] > 0.5 else 0
            
            # 1.4 解析资源分配 - 使用连续动作部分
            # 创建正确形状的资源分配矩阵
            res_alloc_matrix = np.zeros((self.N, self.M))
            # 安全填充连续动作值
            flat_idx = 0
            for i in range(self.N):
                for j in range(self.M):
                    if flat_idx < len(continuous_action):
                        res_alloc_matrix[i, j] = continuous_action[flat_idx]
                    else:
                        # 如果索引超出范围，使用默认值0.5
                        res_alloc_matrix[i, j] = 0.5
                    flat_idx += 1
            
            # 更新边缘节点计算资源
            edge_resources = np.mean(res_alloc_matrix, axis=0)  # M维数组
            self.f_e = self.f_e_min + edge_resources * (self.f_e_max - self.f_e_min)
            
            # 2. 检查动作有效性
            action_valid, invalid_reasons = self._check_action_validity(
                train_local_decisions, edge_train_matrix, cloud_agg_decisions, edge_agg_matrix)
            
            if not action_valid:
                # 如果动作无效，返回负奖励，但不终止Episode
                next_state = self._get_state()
                info = {
                    "error": "无效动作: " + "; ".join(invalid_reasons),
                    "valid_ratio": 0
                }
                return next_state, -10.0, False, info
            
            # 3. 计算延迟和能耗
            delays, energies, costs = [], [], []
            valid_flags = []  # 记录各任务是否满足最大延迟约束
            
            for i in range(self.N):
                # 3.1 确定训练节点
                train_comm_delay, train_comm_energy = 0.0, 0.0
                
                # 本地训练决策
                is_local_training = train_local_decisions[i] == 1
                
                # 边缘训练决策
                train_edge_idx = None
                if not is_local_training:
                    train_edge_indices = np.where(edge_train_matrix[i])[0]
                    train_edge_idx = train_edge_indices[0] if len(train_edge_indices) > 0 else 0
                
                # 3.2 确定聚合节点
                is_cloud_agg = cloud_agg_decisions[i] == 1
                selected_edge_indices = np.where(edge_agg_matrix[i])[0]
                agg_edge_idx = selected_edge_indices[0] if len(selected_edge_indices) > 0 else 0
                
                # 3.3 计算训练阶段延迟和能耗
                if is_local_training:
                    # 本地训练
                    train_comp_delay, train_comp_energy = self._calculate_train_computation(
                        i, is_local=True
                    )
                else:
                    # 边缘训练
                    train_comm_delay, train_comm_energy = self._calculate_train_communication(
                        i, train_edge_idx
                    )
                    train_comp_delay, train_comp_energy = self._calculate_train_computation(
                        i, train_edge_idx, is_local=False
                    )
                
                # 3.4 计算聚合阶段延迟和能耗
                agg_upload_delay, agg_upload_energy = self._calculate_aggregate_upload(
                    i, train_edge_idx, agg_edge_idx, 
                    is_local_training=is_local_training, 
                    is_cloud_agg=is_cloud_agg
                )
                
                agg_feedback_delay, agg_feedback_energy = self._calculate_aggregate_feedback(
                    i, agg_edge_idx, is_cloud_agg=is_cloud_agg
                )
                
                # 3.5 计算总延迟和能耗
                total_delay = train_comm_delay + train_comp_delay + agg_upload_delay + agg_feedback_delay
                total_energy = train_comm_energy + train_comp_energy + agg_upload_energy + agg_feedback_energy
                
                # 计算任务执行成本
                task_cost = self.alpha * total_delay + self.beta * total_energy
                
                # 记录当前FL轮次的延迟、能耗和成本
                self.delays_history[i, self.current_fl_round] = total_delay
                self.energies_history[i, self.current_fl_round] = total_energy
                self.costs_history[i, self.current_fl_round] = task_cost
                
                delays.append(total_delay)
                energies.append(total_energy)
                costs.append(task_cost)
                valid_flags.append(total_delay <= self.time_slot)  # 检查是否满足时隙约束
            
            # 4. 更新每个设备的队列并计算奖励
            total_reward = 0.0
            for i in range(self.N):
                # 获取当前设备的队列积压（t时刻）
                Q_i_t = self.queues[i].queue
                
                # 更新队列（变为t+1时刻）
                self.queues[i].update_queue(energies[i])
                
                # 计算该设备的奖励项
                device_reward = -(Q_i_t * energies[i] + self.reward_bate * costs[i])
                total_reward += device_reward
            
            # 平均奖励
            avg_reward = total_reward / self.N if self.N > 0 else 0
            
            # 5. 更新FL轮次计数
            self.current_fl_round += 1
            
            # 6. 随机变化环境状态
            self._randomize_environment()
            
            # 获取新的环境状态
            next_state = self._get_state()
            
            # 判断当前Episode是否结束
            episode_done = self.current_fl_round >= self.fl_rounds_per_episode
            
            # 记录本FL轮次的数据
            self.episode_rewards.append(avg_reward)
            self.episode_actions.append(action)
            self.episode_next_states.append(next_state)
            self.debug_info['reward_history'].append(avg_reward)
            
            # 构建info字典
            info = {
                "avg_delay": np.mean(delays),
                "avg_energy": np.mean(energies),
                "avg_cost": np.mean(costs),
                "valid_ratio": np.mean(valid_flags),
                "energy_queues": [q.queue for q in self.queues]
            }
            
            return next_state, avg_reward, episode_done, info
            
        except Exception as e:
            # 出错时返回一个安全的状态和负奖励
            next_state = self._get_state()
            info = {"error": f"环境step方法执行错误: {e}"}
            return next_state, -10.0, False, info
        
     def get_episode_data(self):
        """获取当前Episode的所有历史数据，用于DRL智能体学习"""
        return {
            'states': self.episode_states,
            'actions': self.episode_actions,
            'rewards': self.episode_rewards,
            'next_states': self.episode_next_states
        }
    
    def set_fl_rounds_per_episode(self, rounds):
        """设置每个DRL Episode包含的FL轮次数量"""
        self.fl_rounds_per_episode = max(1, rounds)
        print(f"每个DRL Episode现在包含 {self.fl_rounds_per_episode} 个FL轮次")
        
    def get_current_episode_info(self):
        """获取当前Episode的信息"""
        return {
            'episode': self.current_episode,
            'fl_round': self.current_fl_round,
            'total_fl_rounds': self.fl_rounds_per_episode,
            'rewards_so_far': self.episode_rewards,
            'average_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0,
            'total_reward': np.sum(self.episode_rewards) if self.episode_rewards else 0
        }
    
    def _randomize_environment(self):
        """每个round随机变化无线信道"""
        # 信道增益变化(瑞利衰落)
        # 使用Jakes模型的思想，通过添加随机相位来模拟时变特性
        # 保持一定的相关性，避免信道增益突变
        correlation_factor = 0.95  # 时间相关性因子
        
        # 生成新的瑞利衰落信道增益
        new_h_up = np.random.rayleigh(scale=1.0, size=(self.N, self.M))
        new_h_down = np.random.rayleigh(scale=1.0, size=(self.N, self.M))
        
        # 更新信道增益，保持时间相关性
        self.h_up = correlation_factor * self.h_up + (1 - correlation_factor) * new_h_up
        self.h_down = correlation_factor * self.h_down + (1 - correlation_factor) * new_h_down
        
        # 确保信道增益非负
        self.h_up = np.maximum(0.1, self.h_up)
        self.h_down = np.maximum(0.1, self.h_down)
        

    def _check_action_validity(self, train_local_decisions, edge_train_matrix, cloud_agg_decisions, edge_agg_matrix):
        """最小化有效性检查，以适应DRL智能体初始探索阶段"""
        # 在训练初期，智能体可能会生成各种不合理的动作
        # 我们需要更宽容的检查，让智能体能够通过探索学习正确的动作
        
        # 始终返回有效，让系统继续运行
        # 即使没有设备参与训练或聚合，也不要中断
        return True, []

    def get_stats(self):
        """获取当前环境的统计信息"""
        # 计算平均延迟、能耗和成本
        current_delays = self.delays_history[:, :self.current_fl_round].mean() if self.current_fl_round > 0 else 0
        current_energies = self.energies_history[:, :self.current_fl_round].mean() if self.current_fl_round > 0 else 0
        current_costs = self.costs_history[:, :self.current_fl_round].mean() if self.current_fl_round > 0 else 0
        
        # 使用LyapunovQueue类的方法获取队列统计信息
        avg_queue = np.mean(self.queue.queue)
        max_queue = np.max(self.queue.queue)
        
        # 统计信息字典
        stats = {
            "episode": self.current_episode,
            "time_slot": self.current_fl_round,
            "avg_delay": current_delays,
            "avg_energy": current_energies,
            "avg_cost": current_costs,
            "avg_energy_queue": avg_queue,
            "max_energy_queue": max_queue,
            "data_sizes": np.mean(self.data_sizes) / (1024 * 1024),  # 转换为MB
            "channel_quality": np.mean(self.h_up),
        }
        
        return stats
        
    def initialize_dataset(self, dataset_name, is_iid=True, non_iid_level=0.5, data_path='./data'):
        """初始化数据集并分配给各个终端设备
        
        Args:
            dataset_name: 数据集名称
            is_iid: 是否为IID数据分布
            non_iid_level: 非IID程度
            data_path: 数据集存储路径
            
        Returns:
            client_datasets: 客户端数据集字典
            test_data: 测试数据集
        """
        print(f"环境初始化数据集: {dataset_name}, {'IID' if is_iid else f'非IID(level={non_iid_level})'}")
        
        # 保存数据集配置
        self.dataset_name = dataset_name
        self.is_iid = is_iid
        self.non_iid_level = non_iid_level
        
        # 加载数据集
        train_data, test_data = get_dataset(dataset_name, data_path)
        self.train_data = train_data
        self.test_data = test_data
        
        # 准备客户端数据集字典，稍后由ClientsGroup使用
        self.client_datasets = {
            'train_data': train_data,
            'test_data': test_data,
            'is_iid': is_iid,
            'non_iid_level': non_iid_level
        }
        
        # 设置数据大小 - 根据数据集类型估计大小
        if dataset_name.lower() == 'mnist':
            # MNIST: 60,000张28x28的图像，每个客户端约有60000/N张
            avg_samples = 60000 / self.N
            # 每个样本大约是28*28*1 = 784字节
            sample_size = 784
        elif dataset_name.lower() in ['cifar10', 'cifar']:
            # CIFAR10: 50,000张32x32x3的图像，每个客户端约有50000/N张
            avg_samples = 50000 / self.N
            # 每个样本大约是32*32*3 = 3072字节
            sample_size = 3072
        else:
            # 默认估计
            avg_samples = 50000 / self.N
            sample_size = 1000
        
        # 更新数据大小估计
        self.data_sizes = np.ones(self.N) * avg_samples * sample_size
        
        # 更新模型参数大小
        self.model_sizes = np.array([self.delta * ds for ds in self.data_sizes])
        
        return self.client_datasets, test_data
    
    def set_clients_and_server(self, clients_manager, server):
        """设置客户端和服务器实例，用于环境与FL系统交互"""
        self.clients_manager = clients_manager
        self.server = server
        
        # 同步环境参数
        if hasattr(clients_manager, 'num_clients'):
            self.N = clients_manager.num_clients
        if hasattr(clients_manager, 'num_edges'):
            self.M = clients_manager.num_edges
            
        # 如果已经初始化了数据集，将数据集信息传递给客户端管理器
        if self.dataset_name is not None and hasattr(clients_manager, 'update_datasets'):
            clients_manager.update_datasets(self.client_datasets)
        
        # 尝试从客户端管理器获取数据集大小
        if hasattr(clients_manager, 'get_data_sizes'):
            data_sizes = clients_manager.get_data_sizes()
            if data_sizes is not None and len(data_sizes) == self.N:
                self.set_data_sizes(data_sizes)
        
        return self
        
    def _check_constraints(self, train_decisions, aggreg_decision):
        """
        旧的约束检查方法，重定向到新方法以保持向后兼容性
        """
        # 创建兼容的输入
        train_local_decisions = train_decisions
        edge_train_matrix = np.zeros((self.N, self.M))
        for i in range(self.N):
            if train_decisions[i] == 0:  # 如果是边缘训练
                edge_train_matrix[i, 0] = 1  # 默认选第一个边缘节点
        
        cloud_agg_decisions = np.zeros(self.N)
        edge_agg_matrix = np.zeros((self.N, self.M))
        if aggreg_decision == 1:  # 边缘聚合
            edge_agg_matrix[:, 0] = 1  # 默认全部选第一个边缘节点
        else:  # 云端聚合
            cloud_agg_decisions[:] = 1
        
        is_valid, _ = self._check_action_validity(train_local_decisions, edge_train_matrix, 
                                                cloud_agg_decisions, edge_agg_matrix)
        return is_valid
        