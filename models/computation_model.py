"""
计算模型模块

实现计算任务的物理模型，包括：
- 本地计算的延迟和能耗
- 边缘计算的延迟和能耗
- CPU频率管理
- 计算复杂度处理
"""

import numpy as np
from config import F_L_MIN, F_L_MAX, F_E_MIN, F_E_MAX, COMPUTE_COMPLEXITY_MIN, COMPUTE_COMPLEXITY_MAX


class ComputationModel:
    """计算模型类"""
    
    def __init__(self, num_devices, num_edges):
        """
        初始化计算模型
        
        Args:
            num_devices: 终端设备数量
            num_edges: 边缘服务器数量
        """
        self.N = num_devices
        self.M = num_edges
        
        # 终端设备计算资源参数
        self.f_l_min = F_L_MIN  # 最小CPU频率
        self.f_l_max = F_L_MAX  # 最大CPU频率
        
        # 边缘节点计算资源参数
        self.f_e_min = F_E_MIN  # 最小CPU频率
        self.f_e_max = F_E_MAX  # 最大CPU频率
        
        # 初始化终端设备计算资源 (异构)
        self.f_l = np.random.uniform(self.f_l_min, self.f_l_max, size=self.N)
        
        # 初始化边缘节点计算资源
        self.f_e = np.full(self.M, self.f_e_min)  # 当前分配的计算资源
        self.F_e = np.full(self.M, self.f_e_max)  # 最大计算能力上限
        
        # 计算复杂度 (cycles/bit) - 不同设备不同复杂度
        self.c = np.random.uniform(COMPUTE_COMPLEXITY_MIN, COMPUTE_COMPLEXITY_MAX, size=self.N)
        
    def calculate_local_computation(self, device_idx, data_size, freq=None):
        """
        计算本地计算的延迟和能耗
        
        Args:
            device_idx: 设备索引
            data_size: 数据大小 (bytes)
            freq: 指定的计算频率 (Hz)，如果为None则使用当前分配的频率
            
        Returns:
            (延迟, 能耗): 计算延迟和能耗
        """
        if device_idx >= self.N:
            return 0.0, 0.0
            
        # 获取计算复杂度
        cycles_per_bit = float(self.c[device_idx]) if isinstance(self.c, np.ndarray) else 0.0
        
        # 确定计算频率
        if freq is not None:
            compute_freq = float(freq)
        elif isinstance(self.f_l, np.ndarray) and device_idx < len(self.f_l):
            compute_freq = float(self.f_l[device_idx])
        else:
            compute_freq = self.f_l_min
            
        # 确保频率在有效范围内
        compute_freq = max(self.f_l_min, min(self.f_l_max, compute_freq))
        
        # 计算延迟和能耗
        delay = data_size * cycles_per_bit / compute_freq
        energy = 1e-27 * (compute_freq)**3 * delay
        
        return delay, energy
        
    def calculate_edge_computation(self, edge_idx, data_size, freq=None):
        """
        计算边缘计算的延迟和能耗
        
        Args:
            edge_idx: 边缘节点索引
            data_size: 数据大小 (bytes)
            freq: 指定的计算频率 (Hz)，如果为None则使用当前分配的频率
            
        Returns:
            (延迟, 能耗): 计算延迟和能耗
        """
        if edge_idx >= self.M:
            return 0.0, 0.0
            
        # 使用平均计算复杂度（可以根据任务类型调整）
        cycles_per_bit = np.mean(self.c)
        
        # 确定计算频率
        if freq is not None:
            compute_freq = float(freq)
        elif isinstance(self.f_e, np.ndarray) and edge_idx < len(self.f_e):
            compute_freq = float(self.f_e[edge_idx])
        else:
            compute_freq = self.f_e_min
            
        # 确保频率在有效范围内
        compute_freq = max(self.f_e_min, min(self.f_e_max, compute_freq))
        
        # 计算延迟和能耗
        delay = data_size * cycles_per_bit / compute_freq
        energy = 1e-27 * (compute_freq)**3 * delay
        
        return delay, energy
        
    def calculate_computation_with_allocation(self, device_idx, edge_idx, data_size, 
                                            is_local, allocation_ratio):
        """
        根据资源分配比例计算计算延迟和能耗
        
        Args:
            device_idx: 设备索引
            edge_idx: 边缘节点索引
            data_size: 数据大小 (bytes)
            is_local: 是否本地计算
            allocation_ratio: 资源分配比例 (0-1)
            
        Returns:
            (延迟, 能耗): 计算延迟和能耗
        """
        if is_local:
            return self.calculate_local_computation(device_idx, data_size)
        else:
            # 计算分配给该任务的CPU频率
            allocated_freq = allocation_ratio * self.F_e[edge_idx]
            return self.calculate_edge_computation(edge_idx, data_size, freq=allocated_freq)
            
    def update_edge_resource_allocation(self, edge_train_matrix, res_alloc_matrix):
        """
        更新边缘节点的资源分配状态
        
        Args:
            edge_train_matrix: 边缘训练决策矩阵 (N×M)
            res_alloc_matrix: 资源分配矩阵 (N×M)
        """
        # 重置边缘节点资源分配
        self.f_e = np.full(self.M, 0.0)
        
        # 根据决策矩阵更新边缘节点资源分配
        for j in range(self.M):
            total_alloc_freq = 0.0  # 累加分配的绝对频率
            for i in range(self.N):
                if (i < edge_train_matrix.shape[0] and j < edge_train_matrix.shape[1] and 
                    edge_train_matrix[i, j] == 1):
                    # 该设备的任务被分配到此边缘节点
                    if (i < res_alloc_matrix.shape[0] and j < res_alloc_matrix.shape[1]):
                        alloc_ratio = min(max(res_alloc_matrix[i, j], 0), 1)
                        # 计算分配的绝对频率并累加
                        total_alloc_freq += alloc_ratio * self.F_e[j]
            
            # 确保不超过最大值
            self.f_e[j] = min(total_alloc_freq, self.F_e[j])
            
    def get_computation_state_vector(self):
        """
        获取计算状态向量（归一化后）
        
        Returns:
            计算状态向量
        """
        state_vector = []
        
        # 添加终端设备状态
        for i in range(self.N):
            # 设备CPU频率(归一化)
            if isinstance(self.f_l, np.ndarray) and i < len(self.f_l):
                normalized_freq = float(self.f_l[i]) / 3e9
            else:
                normalized_freq = 0.0
            state_vector.append(normalized_freq)
            
        # 添加边缘节点CPU频率
        for j in range(self.M):
            # 当前分配的计算资源(归一化)
            state_vector.append(self.f_e[j] / 4.5e9)
            
        return state_vector
        
    def validate_resource_constraints(self, edge_train_matrix, res_alloc_matrix):
        """
        验证资源约束是否满足
        
        Args:
            edge_train_matrix: 边缘训练决策矩阵 (N×M)
            res_alloc_matrix: 资源分配矩阵 (N×M)
            
        Returns:
            (is_valid, violations): 是否满足约束和违反详情
        """
        violations = []
        
        # 检查边缘节点资源约束
        for j in range(self.M):
            total_ratio = 0.0
            for i in range(self.N):
                if (i < edge_train_matrix.shape[0] and j < edge_train_matrix.shape[1] and
                    edge_train_matrix[i, j] == 1):
                    # 累加分配给边缘节点j的资源比例
                    if (i < res_alloc_matrix.shape[0] and j < res_alloc_matrix.shape[1]):
                        total_ratio += min(max(res_alloc_matrix[i, j], 0), 1)
            
            if total_ratio > 1.0 + 1e-6:  # 添加浮点数容差
                overuse_amount = total_ratio - 1.0
                violations.append(f"边缘节点{j}资源分配超限: 分配比例总和 {total_ratio:.6f} > 1.0")
        
        return len(violations) == 0, violations
        
    def reset_resources(self):
        """重置所有计算资源"""
        # 重新随机化终端设备计算资源
        self.f_l = np.random.uniform(self.f_l_min, self.f_l_max, size=self.N)
        
        # 重置边缘节点资源分配
        self.f_e = np.full(self.M, self.f_e_min)
        
        # 重新随机化计算复杂度
        self.c = np.random.uniform(300, 500, size=self.N) 