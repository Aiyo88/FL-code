"""
动作解析器模块

实现DRL动作的解析和处理，包括：
- 动作空间的解析
- 约束检查
- 决策转换
- 动作验证
"""

import numpy as np
import time
import os
from config import INVALID_ACTION_LOG, CONSTRAINT_TOLERANCE, F_L_MIN, F_L_MAX


class ActionParser:
    """DRL动作解析器类"""
    
    def __init__(self, num_devices, num_edges, log_file=INVALID_ACTION_LOG):
        """
        初始化动作解析器
        
        Args:
            num_devices: 终端设备数量
            num_edges: 边缘服务器数量
            log_file: 无效动作日志文件路径
        """
        self.N = num_devices
        self.M = num_edges
        self.log_file = log_file
        
        # 确保日志目录存在
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        # 从环境中获取决策维度信息
        self.aggregation_choices = self.N + 1
        self.training_choices_per_client = self.M + 1

    def parse_action_for_training(self, action_integer, clients_manager, server):
        """
        [整数解码] 解析来自DQN智能体的单一整数动作。
        
        Args:
            action_integer: int, 代表所有决策的编码整数。
            
        Returns:
            一个包含解析后决策的字典
        """
        temp_action = action_integer
        
        # 1. 解码聚合决策
        agg_choice = temp_action % self.aggregation_choices
        agg_location = "cloud" if agg_choice >= self.M else f"edge_{agg_choice}"
        temp_action //= self.aggregation_choices

        # 2. 解码N个客户端的训练决策
        local_train_decisions = np.zeros(self.N, dtype=int)
        edge_train_decisions = np.zeros((self.N, self.M), dtype=int)
        client_edge_mapping = {}
        edge_load_counter = {f"edge_{j}": 0 for j in range(self.M)}

        for i in range(self.N):
            decision = temp_action % self.training_choices_per_client
            temp_action //= self.training_choices_per_client
            
            if decision == 0:
                local_train_decisions[i] = 1
            else:
                edge_idx = decision - 1
                edge_id = f"edge_{edge_idx}"
                edge_train_decisions[i, edge_idx] = 1
                client_id = f"client{i}"
                client_edge_mapping[client_id] = edge_id
                edge_load_counter[edge_id] += 1

        # 3. 生成资源分配矩阵 (均分策略)
        resource_alloc_matrix = np.zeros((self.N, self.M))
        for client_id, edge_id in client_edge_mapping.items():
            client_idx = int(client_id.replace("client", ""))
            edge_idx = int(edge_id.replace("edge_", ""))
            
            if edge_load_counter[edge_id] > 0:
                resource_alloc_matrix[client_idx, edge_idx] = 1.0 / edge_load_counter[edge_id]

        return {
            "aggregation_location": agg_location,
            "local_train_decisions": local_train_decisions,
            "edge_train_matrix": edge_train_decisions,
            "resource_alloc_matrix": resource_alloc_matrix,
            "client_edge_mapping": client_edge_mapping,
        }

    def convert_to_fl_training_params(self, parsed_action):
        """
        将解析后的动作转换为联邦学习训练所需的参数
        
        Args:
            parsed_action: `parse_action_for_training` 返回的字典
            
        Returns:
            (training_args, raw_decisions)
        """
        agg_location = parsed_action["aggregation_location"]
        local_train = parsed_action["local_train_decisions"]
        edge_train_matrix = parsed_action["edge_train_matrix"]
        res_alloc = parsed_action["resource_alloc_matrix"]
        client_edge_mapping = parsed_action["client_edge_mapping"]

        # 确定参与训练的节点
        selected_nodes = []
        resource_allocation = {}
        
        # 本地训练的客户端
        for i in range(self.N):
            if local_train[i] == 1:
                client_id = f"client{i}"
                selected_nodes.append(client_id)
                # 任务标识符: (执行节点, 代理客户端) -> 资源比例
                resource_allocation[(client_id, client_id)] = 1.0 # 本地训练独占资源
        
        # 卸载训练的客户端 (由边缘节点代理)
        for client_id, edge_id in client_edge_mapping.items():
            if edge_id not in selected_nodes:
                selected_nodes.append(edge_id)
            
            client_idx = int(client_id.replace("client", ""))
            edge_idx = int(edge_id.replace("edge_", ""))
            
            # 任务标识符: (执行节点, 代理客户端) -> 资源比例
            ratio = res_alloc[client_idx, edge_idx]
            resource_allocation[(edge_id, client_id)] = ratio

        # 准备 training_args
        training_args = {
            'selected_nodes': selected_nodes,
            'resource_allocation': resource_allocation,
            'aggregation_location': agg_location,
            'drl_train_decisions': local_train.tolist(), # 保持与旧格式的兼容
        }

        # 准备 raw_decisions (用于环境步进)
        edge_agg = np.zeros(self.M)
        cloud_agg = 0
        if agg_location == "cloud":
            cloud_agg = 1
        else:
            try:
                edge_idx = int(agg_location.split('_')[1])
                if edge_idx < self.M:
                    edge_agg[edge_idx] = 1
            except (IndexError, ValueError):
                cloud_agg = 1 # 安全回退

        raw_decisions = {
            'local_train': local_train,
            'edge_train': edge_train_matrix.flatten(),
            'edge_agg': edge_agg,
            'cloud_agg': np.array(cloud_agg),
            'resource_alloc': res_alloc.flatten(),
            'edge_client_mapping': client_edge_mapping
        }
        
        return training_args, raw_decisions

    def is_action_valid(self, raw_decisions, episode_idx, global_round_idx):
        """
        在新动作空间下，大部分约束在解析时已强制满足，
        主要检查资源分配是否超限。
        """
        if raw_decisions is None:
            return False
            
        res_alloc_flat = raw_decisions.get('resource_alloc', [])
        res_alloc_matrix = np.array(res_alloc_flat).reshape((self.N, self.M))
        
        violations = []
        for j in range(self.M):
            total_ratio = np.sum(res_alloc_matrix[:, j])
            if total_ratio > 1.0 + CONSTRAINT_TOLERANCE:
                violations.append(f"Edge {j} resource overuse: {total_ratio:.4f} > 1.0")
        
        if violations:
            self.log_invalid_action(episode_idx, global_round_idx, violations)
            return False
        return True

    def log_invalid_action(self, episode, round_num, violations):
        """记录无效动作到日志文件"""
        log_message = f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Episode {episode}, Round {round_num}: Invalid action detected.\n"
        log_message += "Violations:\n" + "\n".join(f"  - {v}" for v in violations) + "\n\n"
        try:
            with open(self.log_file, "a") as f:
                f.write(log_message)
        except IOError as e:
            print(f"Error writing to invalid action log: {e}") 