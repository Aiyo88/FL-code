"""
成本计算器模块

实现系统成本计算，包括：
- 训练阶段成本计算（通信+计算）
- 聚合阶段成本计算（上传+下载）
- 总体成本汇总
- 奖励函数计算
"""

import numpy as np
from config import TIME_SLOT, MAX_COST_PER_ROUND, MAX_Q_ENERGY_PER_ROUND, W_COST, W_Q, W_LOSS


class CostCalculator:
    """系统成本计算器类"""
    
    def __init__(self, communication_model, computation_model, alpha=0.5, beta=0.5):
        """
        初始化成本计算器
        
        Args:
            communication_model: 通信模型实例
            computation_model: 计算模型实例
            alpha: 延迟权重
            beta: 能耗权重
        """
        self.comm_model = communication_model
        self.comp_model = computation_model
        self.alpha = alpha  # 延迟权重
        self.beta = beta   # 能耗权重
        
        # 数据和模型大小
        self.data_sizes = None
        self.model_sizes = None
        
    def set_data_model_sizes(self, data_sizes, model_sizes):
        """
        设置数据和模型大小
        
        Args:
            data_sizes: 各设备的数据大小数组
            model_sizes: 各设备的模型大小数组
        """
        self.data_sizes = data_sizes
        self.model_sizes = model_sizes
        
    def calculate_train_communication_cost(self, device_idx, edge_idx):
        """
        计算训练阶段的通信延迟和能耗 (从终端设备到训练节点)
        仅当训练在边缘节点时才有通信开销
        
        Args:
            device_idx: 设备索引
            edge_idx: 边缘节点索引
            
        Returns:
            (延迟, 能耗): 通信延迟和能耗
        """
        if self.data_sizes is None:
            return 0.0, 0.0
            
        data_size = self.data_sizes[device_idx]
        return self.comm_model.calculate_transmission_delay_energy(
            data_size, device_idx, edge_idx, is_uplink=True
        )

    def calculate_train_computation_cost(self, device_idx, edge_idx=None, is_local=True, allocation_ratio=1.0):
        """
        计算训练阶段的计算延迟和能耗
        
        Args:
            device_idx: 设备索引
            edge_idx: 边缘节点索引
            is_local: 是否本地计算
            allocation_ratio: 资源分配比例
            
        Returns:
            (延迟, 能耗): 计算延迟和能耗
        """
        if self.data_sizes is None:
            return 0.0, 0.0
            
        data_size = self.data_sizes[device_idx]
        return self.comp_model.calculate_computation_with_allocation(
            device_idx, edge_idx, data_size, is_local, allocation_ratio
        )

    def calculate_aggregate_upload_cost(self, device_idx, train_edge_idx, agg_edge_idx, 
                                      is_local_training, is_cloud_agg):
        """
        计算聚合阶段的上传延迟和能耗 (从训练节点到聚合节点)
        
        Args:
            device_idx: 设备索引
            train_edge_idx: 训练边缘节点索引
            agg_edge_idx: 聚合边缘节点索引
            is_local_training: 是否本地训练
            is_cloud_agg: 是否云端聚合
            
        Returns:
            (延迟, 能耗): 上传延迟和能耗
        """
        if self.model_sizes is None:
            return 0.0, 0.0
            
        model_size = self.model_sizes[device_idx]
        
        if is_local_training:
            # 本地训练 → 上传到聚合节点
            if is_cloud_agg:
                # 场景2: 本地 → 边缘 → 云
                # 步骤1: 终端设备到边缘节点
                delay_to_edge, energy_to_edge = self.comm_model.calculate_transmission_delay_energy(
                    model_size, device_idx, agg_edge_idx, is_uplink=True
                )
                
                # 步骤2: 边缘节点到云端
                delay_to_cloud, _ = self.comm_model.calculate_edge_to_cloud_delay_energy(
                    model_size, is_uplink=True
                )
                
                return delay_to_edge + delay_to_cloud, energy_to_edge
            else:
                # 场景1: 本地 → 边缘
                return self.comm_model.calculate_transmission_delay_energy(
                    model_size, device_idx, agg_edge_idx, is_uplink=True
                )
        else:
            # 边缘训练 → 上传到聚合节点
            if is_cloud_agg:
                # 场景4: 边缘 → 云
                return self.comm_model.calculate_edge_to_cloud_delay_energy(
                    model_size, is_uplink=True
                )
            else:
                # 场景3: 边缘 → 边缘
                if train_edge_idx == agg_edge_idx:
                    # 场景3a: 同一个边缘节点，无需传输
                    return 0.0, 0.0
                else:
                    # 场景3b: 不同边缘节点之间的传输
                    return self.comm_model.calculate_edge_to_edge_delay_energy(model_size)

    def calculate_aggregate_feedback_cost(self, device_idx, agg_edge_idx, is_cloud_agg):
        """
        计算聚合阶段的反馈延迟和能耗 (从聚合节点到终端设备)
        
        Args:
            device_idx: 设备索引
            agg_edge_idx: 聚合边缘节点索引
            is_cloud_agg: 是否云端聚合
            
        Returns:
            (延迟, 能耗): 反馈延迟和能耗
        """
        if self.model_sizes is None:
            return 0.0, 0.0
            
        model_size = self.model_sizes[device_idx]
        
        if is_cloud_agg:
            # 场景2: 云 → 边缘 → 终端
            # 步骤1: 云到边缘
            delay_cloud_to_edge, energy_cloud_to_edge = self.comm_model.calculate_edge_to_cloud_delay_energy(
                model_size, is_uplink=False
            )
            
            # 步骤2: 边缘到终端
            delay_edge_to_device, energy_edge_to_device = self.comm_model.calculate_transmission_delay_energy(
                model_size, device_idx, agg_edge_idx, is_uplink=False
            )
            
            total_delay = delay_cloud_to_edge + delay_edge_to_device
            total_infra_energy = energy_cloud_to_edge + energy_edge_to_device
            return total_delay, total_infra_energy
        else:
            # 场景1: 边缘 → 终端
            return self.comm_model.calculate_transmission_delay_energy(
                model_size, device_idx, agg_edge_idx, is_uplink=False
            )

    def calculate_device_total_cost(self, device_idx, train_local_decisions, edge_train_matrix, 
                                  edge_agg_decisions, cloud_agg_decision, res_alloc_matrix):
        """
        计算单个设备的总成本
        
        Args:
            device_idx: 设备索引
            train_local_decisions: 本地训练决策数组
            edge_train_matrix: 边缘训练矩阵
            edge_agg_decisions: 边缘聚合决策数组
            cloud_agg_decision: 云聚合决策
            res_alloc_matrix: 资源分配矩阵
            
        Returns:
            (总延迟, 总能耗, 总成本, 设备能耗): 各项成本指标
        """
        # 初始化分层成本变量
        train_comm_delay, train_comm_energy = 0.0, 0.0
        train_comp_delay, train_comp_energy = 0.0, 0.0
        agg_upload_delay, agg_upload_energy = 0.0, 0.0
        agg_feedback_delay, agg_feedback_energy = 0.0, 0.0

        # 获取设备决策变量
        is_local_train = int(train_local_decisions[device_idx]) if device_idx < len(train_local_decisions) else 0
        edge_train_selections = edge_train_matrix[device_idx].astype(int) if device_idx < edge_train_matrix.shape[0] else np.zeros(self.comm_model.M, dtype=int)
        
        device_energy_consumption = 0.0  # 设备自身的能耗（用于队列更新）
        
        # --- 1. 累加训练成本 ---
        # a. 如果选择本地训练，累加其成本
        if is_local_train == 1:
            comp_d, comp_e = self.calculate_train_computation_cost(device_idx, is_local=True)
            train_comp_delay += comp_d
            train_comp_energy += comp_e
            device_energy_consumption += comp_e  # 本地计算能耗计入设备

        # b. 循环所有边缘节点，如果选择在某边缘训练，累加其成本
        selected_train_edges = []  # 用于后续聚合阶段
        for m in range(self.comm_model.M):
            if edge_train_selections[m] == 1:
                selected_train_edges.append(m)
                # 通信成本
                comm_d, comm_e = self.calculate_train_communication_cost(device_idx, m)
                train_comm_delay += comm_d
                train_comm_energy += comm_e
                device_energy_consumption += comm_e  # 设备上传能耗计入设备
                # 计算成本
                alloc_ratio = min(max(res_alloc_matrix[device_idx, m], 0), 1)
                comp_d, comp_e = self.calculate_train_computation_cost(
                    device_idx, edge_idx=m, is_local=False, allocation_ratio=alloc_ratio
                )
                train_comp_delay += comp_d
                train_comp_energy += comp_e  # 边缘计算能耗只计入任务总成本

        # --- 2. 累加聚合成本 ---
        is_cloud_agg = int(cloud_agg_decision)
        selected_agg_edges = np.where(edge_agg_decisions > 0)[0]

        # a. 累加所有被选中的边缘聚合方案的成本
        for agg_edge_idx in selected_agg_edges:
            # 聚合上传成本
            if is_local_train == 1:
                up_d, up_e = self.calculate_aggregate_upload_cost(
                    device_idx, None, agg_edge_idx, True, False
                )
                agg_upload_delay += up_d
                agg_upload_energy += up_e
                device_energy_consumption += up_e  # 本地上传计入设备
            for train_edge_idx in selected_train_edges:
                up_d, up_e = self.calculate_aggregate_upload_cost(
                    device_idx, train_edge_idx, agg_edge_idx, False, False
                )
                agg_upload_delay += up_d
                agg_upload_energy += up_e
            # 聚合反馈成本
            fb_d, fb_e = self.calculate_aggregate_feedback_cost(device_idx, agg_edge_idx, False)
            agg_feedback_delay += fb_d
            agg_feedback_energy += fb_e

        # b. 如果选择云聚合，累加其成本
        if is_cloud_agg == 1:
            relay_edge_up = np.argmax(self.comm_model.h_up[device_idx])
            relay_edge_down = np.argmax(self.comm_model.h_down[device_idx])
            # 聚合上传成本
            if is_local_train == 1:
                up_d, up_e = self.calculate_aggregate_upload_cost(
                    device_idx, None, relay_edge_up, True, True
                )
                agg_upload_delay += up_d
                agg_upload_energy += up_e
                device_energy_consumption += up_e  # 本地上传计入设备
            for train_edge_idx in selected_train_edges:
                up_d, up_e = self.calculate_aggregate_upload_cost(
                    device_idx, train_edge_idx, None, False, True
                )
                agg_upload_delay += up_d
                agg_upload_energy += up_e
            # 聚合反馈成本
            fb_d, fb_e = self.calculate_aggregate_feedback_cost(device_idx, relay_edge_down, True)
            agg_feedback_delay += fb_d
            agg_feedback_energy += fb_e
        
        # --- 3. 最终汇总 ---
        total_delay = train_comm_delay + train_comp_delay + agg_upload_delay + agg_feedback_delay
        total_energy = train_comm_energy + train_comp_energy + agg_upload_energy + agg_feedback_energy
        task_cost = self.alpha * total_delay + self.beta * total_energy
        
        return total_delay, total_energy, task_cost, device_energy_consumption

    def calculate_system_total_cost(self, train_local_decisions, edge_train_matrix, 
                                  edge_agg_decisions, cloud_agg_decision, res_alloc_matrix):
        """
        计算系统总成本
        
        Args:
            train_local_decisions: 本地训练决策数组
            edge_train_matrix: 边缘训练矩阵
            edge_agg_decisions: 边缘聚合决策数组
            cloud_agg_decision: 云聚合决策
            res_alloc_matrix: 资源分配矩阵
            
        Returns:
            (总延迟列表, 总能耗列表, 总成本列表, 设备能耗列表, 有效标志列表): 各项成本指标
        """
        delays, energies, costs = [], [], []
        device_energies_for_queue = np.zeros(self.comm_model.N)
        valid_flags = []  # 记录各任务是否满足时隙约束
        
        # 为每个终端设备计算延迟和能耗
        for i in range(self.comm_model.N):
            total_delay, total_energy, task_cost, device_energy = self.calculate_device_total_cost(
                i, train_local_decisions, edge_train_matrix, 
                edge_agg_decisions, cloud_agg_decision, res_alloc_matrix
            )
            
            delays.append(total_delay)
            energies.append(total_energy)
            costs.append(task_cost)
            device_energies_for_queue[i] = device_energy
            valid_flags.append(total_delay <= TIME_SLOT)  # 检查是否满足时隙约束
        
        return delays, energies, costs, device_energies_for_queue, valid_flags

    def calculate_lyapunov_reward(self, costs, device_energies, queue_manager, fl_loss=None, 
                                w_cost=W_COST, w_q=W_Q, w_loss=W_LOSS):
        """
        计算基于李雅普诺夫理论的奖励
        
        Args:
            costs: 各设备的成本列表
            device_energies: 各设备的能耗数组
            queue_manager: 李雅普诺夫队列管理器
            fl_loss: 联邦学习损失
            w_cost: 成本权重
            w_q: 队列权重
            w_loss: 损失权重
            
        Returns:
            总奖励
        """
        # 归一化参数
        max_cost_per_round = MAX_COST_PER_ROUND
        max_q_energy_per_round = MAX_Q_ENERGY_PER_ROUND
        
        total_reward = 0.0
        
        # 计算每个设备的奖励
        for i in range(min(len(costs), len(queue_manager.queues))):
            Q_i_t = queue_manager.queues[i].queue
            
            # 1. 成本惩罚项 (归一化)
            normalized_cost = costs[i] / max_cost_per_round
            cost_penalty = w_cost * normalized_cost
            
            # 2. 队列惩罚项 (归一化)
            q_energy = Q_i_t * device_energies[i]
            normalized_q_energy = q_energy / max_q_energy_per_round
            q_penalty = w_q * normalized_q_energy

            # 3. 性能奖励项
            loss_reward = 0.0
            if fl_loss is not None and 0 < fl_loss < 1:
                loss_reward = w_loss * (1.0 - fl_loss)

            # 组合奖励
            device_reward = loss_reward - cost_penalty - q_penalty
            total_reward += device_reward
            
        return total_reward 