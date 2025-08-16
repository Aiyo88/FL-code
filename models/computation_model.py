"""
计算模型模块

实现计算任务的物理模型，包括：
- 本地计算的延迟和能耗
- 边缘计算的延迟和能耗
- CPU频率管理
- 计算复杂度处理
"""

import numpy as np
from config import F_L_MIN, F_L_MAX, F_E_MIN, F_E_MAX, COMPUTE_COMPLEXITY_MIN, COMPUTE_COMPLEXITY_MAX, COMPUTATION_ENERGY_SCALE


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
            freq: 保留参数，不再使用。终端设备本地计算频率固定为初始化的 self.f_l[device_idx]
            
        Returns:
            (延迟, 能耗): 计算延迟和能耗
        """
        if device_idx >= self.N:
            return 0.0, 0.0
            
        # 获取计算复杂度
        cycles_per_bit = float(self.c[device_idx]) if isinstance(self.c, np.ndarray) else 0.0
        
        # 固定使用设备的本地计算频率（初始化时随机分配）
        if isinstance(self.f_l, np.ndarray) and device_idx < len(self.f_l):
            compute_freq = float(self.f_l[device_idx])
        else:
            compute_freq = self.f_l_min
        
        # 确保频率在有效范围内
        compute_freq = max(self.f_l_min, min(self.f_l_max, compute_freq))
        
        # 计算延迟和能耗 - 修正单位
        # data_size (bytes) × 8 (bits/byte) × cycles_per_bit (cycles/bit) / compute_freq (cycles/sec) = seconds
        delay = (data_size * 8 * cycles_per_bit) / compute_freq
        energy = 1e-27 * (compute_freq)**3 * delay * COMPUTATION_ENERGY_SCALE
        
        return delay, energy
        
    def calculate_edge_computation(self, device_idx, edge_idx, data_size, freq):
        """
        计算边缘计算的延迟和能耗 (已简化)
        
        Args:
            device_idx: 任务所属设备索引（用于采用该设备的计算复杂度）
            edge_idx: 边缘节点索引
            data_size: 数据大小 (bytes)
            freq: 为该任务分配的特定计算频率 (Hz)
            
        Returns:
            (延迟, 能耗): 计算延迟和能耗
        """
        if edge_idx >= self.M or freq <= 0:
            return 0.0, 0.0
            
        # 使用该任务所属设备的计算复杂度
        cycles_per_bit = float(self.c[device_idx])
        
        # 使用明确传入的、已为该任务分配好的频率
        compute_freq = float(freq)
        
        # 计算延迟和能耗 - 修正单位  
        # data_size (bytes) × 8 (bits/byte) × cycles_per_bit (cycles/bit) / compute_freq (cycles/sec) = seconds
        delay = (data_size * 8 * cycles_per_bit) / compute_freq
        energy = 1e-27 * (compute_freq)**3 * delay * COMPUTATION_ENERGY_SCALE
        
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
            return self.calculate_edge_computation(device_idx, edge_idx, data_size, freq=allocated_freq)
            
    def get_computation_state_vector(self):
        pass
        
    def reset_resources(self):
        """重置所有计算资源"""
        # 重新随机化终端设备计算资源
        self.f_l = np.random.uniform(self.f_l_min, self.f_l_max, size=self.N)
        
        # 重置边缘节点资源分配
        self.f_e = np.full(self.M, self.f_e_min)
        
        # 重新随机化计算复杂度
        self.c = np.random.uniform(300, 500, size=self.N) 