"""
通信模型模块

实现无线通信的物理层模型，包括：
- 信道增益计算（瑞利衰落）
- 传输速率计算（香农公式）
- 信道环境的随机变化
- 通信延迟和能耗计算
"""

import numpy as np
from config import RAYLEIGH_SCALE, CORRELATION_FACTOR, MIN_CHANNEL_GAIN, EDGE_TO_EDGE_RATE


class WirelessCommunicationModel:
    """无线通信模型类"""
    
    def __init__(self, num_devices, num_edges, bandwidth=6, noise_power=10e-13):
        """
        初始化通信模型
        
        Args:
            num_devices: 终端设备数量
            num_edges: 边缘服务器数量  
            bandwidth: 信道带宽 (MHz)
            noise_power: 噪声功率
        """
        self.N = num_devices
        self.M = num_edges
        self.B = bandwidth  # 信道带宽 (MHz)
        self.N0 = noise_power  # 噪声功率
        
        # 传输功率设置 (dBm)
        self.Pt_UP = 24    # 上行传输功率 (dBm)
        self.Pt_down = 30  # 下行传输功率 (dBm)
        self.Pt_edge_transmit = 24  # 边缘节点传输功率 (dBm)
        self.Pt_cloud_down = 30     # 云到边缘下行传输功率(dBm)
        
        # 边缘到云的固定速率 (Mbps)
        self.rate_CU = 120  # 边缘到云上行速率 (Mbps)
        self.rate_CD = 150  # 边缘到云下行速率 (Mbps)
        
        # 初始化信道增益矩阵
        self.h_up = np.random.rayleigh(scale=RAYLEIGH_SCALE, size=(self.N, self.M))
        self.h_down = np.random.rayleigh(scale=RAYLEIGH_SCALE, size=(self.N, self.M))
        
        # 传输速率矩阵
        self.R_up = np.zeros((self.N, self.M))
        self.R_down = np.zeros((self.N, self.M))
        
    def calculate_rate(self, transmit_power, channel_gain, noise_power=None):
        """
        计算传输速率（单位：bps）- 基于香农公式
        
        Args:
            transmit_power: 传输功率 (dBm)
            channel_gain: 信道增益
            noise_power: 噪声功率，如果为None则使用默认值
            
        Returns:
            传输速率 (bps)
        """
        if noise_power is None:
            noise_power = self.N0
            
        # 将dBm转换为线性功率
        power_linear = 10**(transmit_power/10) * 1e-3  # 转换为瓦特
        
        # 香农公式计算速率
        rate = self.B * 1e6 * np.log2(1 + power_linear * channel_gain / noise_power)  # MHz转换为Hz
        return rate  # 单位：bps
    
    def update_channel_gains(self, correlation_factor=CORRELATION_FACTOR):
        """
        更新信道增益 - 模拟瑞利衰落的时变特性
        
        Args:
            correlation_factor: 时间相关性因子 (0-1之间)
        """
        # 生成新的瑞利衰落信道增益
        new_h_up = np.random.rayleigh(scale=RAYLEIGH_SCALE, size=(self.N, self.M))
        new_h_down = np.random.rayleigh(scale=RAYLEIGH_SCALE, size=(self.N, self.M))
        
        # 更新信道增益，保持时间相关性
        self.h_up = correlation_factor * self.h_up + (1 - correlation_factor) * new_h_up
        self.h_down = correlation_factor * self.h_down + (1 - correlation_factor) * new_h_down
        
        # 确保信道增益非负
        self.h_up = np.maximum(MIN_CHANNEL_GAIN, self.h_up)
        self.h_down = np.maximum(MIN_CHANNEL_GAIN, self.h_down)
    
    def update_transmission_rates(self):
        """更新所有设备到边缘节点的传输速率"""
        for i in range(self.N):
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
    
    def get_uplink_rate(self, device_idx, edge_idx):
        """
        获取指定设备到边缘节点的上行速率
        
        Args:
            device_idx: 设备索引
            edge_idx: 边缘节点索引
            
        Returns:
            上行传输速率 (bps)
        """
        if device_idx < self.N and edge_idx < self.M:
            return self.R_up[device_idx][edge_idx]
        return 0.0
    
    def get_downlink_rate(self, device_idx, edge_idx):
        """
        获取指定边缘节点到设备的下行速率
        
        Args:
            device_idx: 设备索引
            edge_idx: 边缘节点索引
            
        Returns:
            下行传输速率 (bps)
        """
        if device_idx < self.N and edge_idx < self.M:
            return self.R_down[device_idx][edge_idx]
        return 0.0
    
    def calculate_transmission_delay_energy(self, data_size, device_idx, edge_idx, is_uplink=True):
        """
        计算传输延迟和能耗
        
        Args:
            data_size: 数据大小 (bytes)
            device_idx: 设备索引
            edge_idx: 边缘节点索引
            is_uplink: 是否为上行传输
            
        Returns:
            (延迟, 能耗): 传输延迟和能耗
        """
        if is_uplink:
            rate = self.get_uplink_rate(device_idx, edge_idx)
            power = self.Pt_UP
        else:
            rate = self.get_downlink_rate(device_idx, edge_idx)
            power = self.Pt_down
        
        if rate > 0:
            delay = data_size / rate
            energy = power * delay
            return delay, energy
        else:
            return 0.0, 0.0
    
    def calculate_edge_to_cloud_delay_energy(self, data_size, is_uplink=True):
        """
        计算边缘到云的传输延迟和能耗
        
        Args:
            data_size: 数据大小 (bytes)
            is_uplink: 是否为上行传输（边缘到云）
            
        Returns:
            (延迟, 能耗): 传输延迟和能耗
        """
        if is_uplink:
            rate = self.rate_CU * 1e6  # 转换为bps
            power = self.Pt_edge_transmit
        else:
            rate = self.rate_CD * 1e6  # 转换为bps
            power = self.Pt_cloud_down
        
        delay = data_size / rate
        energy = power * delay
        return delay, energy
    
    def calculate_edge_to_edge_delay_energy(self, data_size, edge_rate=EDGE_TO_EDGE_RATE):
        """
        计算边缘节点间的传输延迟和能耗
        
        Args:
            data_size: 数据大小 (bytes)
            edge_rate: 边缘间传输速率 (bps)，默认1Gbps
            
        Returns:
            (延迟, 能耗): 传输延迟和能耗
        """
        delay = data_size / edge_rate
        energy = self.Pt_edge_transmit * delay
        return delay, energy
    
    def get_communication_state_vector(self):
        """
        获取通信状态向量（归一化后）
        
        Returns:
            通信状态向量
        """
        state_vector = []
        
        # 添加设备到边缘的通信速率
        for i in range(self.N):
            for j in range(self.M):
                state_vector.append(self.R_up[i][j] / 1e8)   # 上行速率(归一化)
                state_vector.append(self.R_down[i][j] / 1e8) # 下行速率(归一化)
        
        # 添加边缘到云的速率
        for j in range(self.M):
            state_vector.append(self.rate_CU / 200.0)  # 边缘到云速率(归一化)
            state_vector.append(self.rate_CD / 200.0)  # 云到边缘速率(归一化)
        
        return state_vector 