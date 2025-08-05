"""
李雅普诺夫能量队列模块

实现基于李雅普诺夫理论的能量队列管理，包括：
- 能量队列的更新
- 李雅普诺夫函数计算
- 漂移计算
- 稳定性判定
"""

import numpy as np
from config import ENERGY_REPLENISH_RATE


class LyapunovQueue:
    """李雅普诺夫能量队列类 - 实现能量队列的更新、李雅普诺夫函数、漂移计算及稳定性判定"""
    
    def __init__(self):
        """
        初始化李雅普诺夫队列
        """
        self.queue = 0.0    # 能量队列初始值 (代表能量赤字)
        
    def update_queue(self, energy_consumed, energy_replenished=ENERGY_REPLENISH_RATE):
        """
        更新能量队列: Q(t+1) = max{0, Q(t) + E_consumed(t) - E_replenished(t)}
        
        Args:
            energy_consumed: 当前时隙消耗的能量
            energy_replenished: 当前时隙补充的能量
            
        Returns:
            更新后的队列值
        """
        self.queue = max(0, self.queue + energy_consumed - energy_replenished)
        return self.queue
    
    def q_u_compute(self, energy_consumed, energy_replenished=ENERGY_REPLENISH_RATE):
        """
        计算队列更新后的值，但不实际更新队列
        """
        return max(0, self.queue + energy_consumed - energy_replenished)
    
    def lyapunov_function(self):
        """
        计算李雅普诺夫函数: L(Q) = 1/2 * Q²
        
        Returns:
            李雅普诺夫函数值
        """
        return 0.5 * self.queue**2
    
    def lyapunov_drift(self, energy_consumed):
        """
        计算李雅普诺夫漂移: ΔL = L(Q(t+1)) - L(Q(t))
        
        Args:
            energy_consumed: 当前时隙的能量消耗
            
        Returns:
            李雅普诺夫漂移值
        """
        q_next = self.q_u_compute(energy_consumed)
        return 0.5 * q_next**2 - self.lyapunov_function()
    
    def is_stable(self, energy_consumed):
        """
        判断队列是否稳定: 如果漂移小于等于0，则队列稳定
        
        Args:
            energy_consumed: 当前时隙的能量消耗
            
        Returns:
            是否稳定
        """
        return self.lyapunov_drift(energy_consumed) <= 0


class LyapunovQueueManager:
    """李雅普诺夫队列管理器 - 管理多个设备的队列"""
    
    def __init__(self, num_devices, energy_threshold):
        """
        初始化队列管理器
        
        Args:
            num_devices: 设备数量
            energy_threshold: 能量阈值
        """
        self.num_devices = num_devices
        self.energy_threshold = energy_threshold
        self.queues = [LyapunovQueue() for _ in range(num_devices)]
    
    def reset_queues(self):
        """重置所有队列"""
        self.queues = [LyapunovQueue() for _ in range(self.num_devices)]
    
    def update_all_queues(self, energy_consumptions):
        """
        更新所有设备的队列
        
        Args:
            energy_consumptions: 各设备的能量消耗数组
            
        Returns:
            更新后的队列值数组
        """
        queue_values = []
        for i, energy in enumerate(energy_consumptions):
            if i < len(self.queues):
                queue_value = self.queues[i].update_queue(energy)
                queue_values.append(queue_value)
        return np.array(queue_values)
    
    def get_queue_states(self):
        """
        获取所有队列的当前状态
        
        Returns:
            队列状态数组
        """
        return np.array([queue.queue for queue in self.queues])
    
    def get_lyapunov_values(self):
        """
        获取所有队列的李雅普诺夫函数值
        
        Returns:
            李雅普诺夫函数值数组
        """
        return np.array([queue.lyapunov_function() for queue in self.queues])
    
    def calculate_q_energy_penalty(self, device_energies):
        """
        计算所有设备的队列能量惩罚项
        
        Args:
            device_energies: 各设备的能量消耗
            
        Returns:
            队列能量惩罚项数组
        """
        penalties = []
        for i, energy in enumerate(device_energies):
            if i < len(self.queues):
                q_energy = self.queues[i].queue * energy
                penalties.append(q_energy)
        return np.array(penalties) 