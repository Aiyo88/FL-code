import os
import torch
import numpy as np
from typing import Any, Dict, List, Tuple, Union
from gym import spaces

class DRLAdapter:
    """DRL算法统一适配器基类 - 为不同DRL算法提供统一接口"""
    
    def __init__(self, agent, env):
        """初始化适配器
        
        Args:
            agent: DRL智能体实例
            env: 环境实例
        """
        self.agent = agent
        self.env = env
        self.device = getattr(agent, 'device', torch.device("cpu"))
        
        # 存储当前Episode的完整经验轨迹
        self.episode_buffer = []
        
        # 获取环境参数
        self.N = getattr(env, 'N', 5)  # 终端设备数量
        self.M = getattr(env, 'M', 2)   # 边缘节点数量
        
        # 记录训练数据
        self.current_episode_buffer = []
        self.episode_rewards = []
        self.episode_counter = 0
        
        # 用于实现DRL Episode和FL轮次的分层嵌套关系
        self.fl_rounds_per_episode = getattr(env, 'fl_rounds_per_episode', 100)
        self.current_fl_round = 0
    
    def get_action(self, state):
        """获取动作 - 子类必须实现此方法
        
        Args:
            state: 环境状态
            
        Returns:
            动作向量，符合环境要求的格式
        """
        raise NotImplementedError("需要由子类实现")
    
    def learn(self, state, action, reward, next_state, done):
        """在每个时间步后调用此方法来存储经验。"""
        # 只存储经验，不立即学习
        self.episode_buffer.append((state, action, reward, next_state, done))

    def learn_from_episode(self):
        """在Episode结束后，调用此方法进行批量学习。"""
        raise NotImplementedError("需要由子类实现")
    
    def save_model(self, path):
        """保存模型 - 子类必须实现此方法"""
        raise NotImplementedError("需要由子类实现")
    
    def load_model(self, path):
        """加载模型 - 子类必须实现此方法"""
        raise NotImplementedError("需要由子类实现")


class PDQNAdapter(DRLAdapter):
    """PDQN算法适配器"""
    
    def __init__(self, agent, env):
        """初始化PDQN智能体适配器
        
        Args:
            agent: PDQN智能体实例
            env: 环境实例
        """
        super().__init__(agent, env)
    
    def get_action(self, state):
        """获取动作
        
        Args:
            state: 环境状态
            
        Returns:
            动作元组(discrete_action, continuous_action)，其中:
            - discrete_action是一个多维离散动作向量，代表并行的二元决策
            - continuous_action是一个连续动作矩阵，形状为(N,M)
        """
        # 确保状态是numpy数组
        if isinstance(state, list):
            state = np.array(state, dtype=np.float32)
        
        # 使用PDQN智能体获取动作 - 返回(discrete_action, continuous_action)元组
        # discrete_action 是一个向量
        return self.agent.select_action(state)
    
    def learn(self, state, action, reward, next_state, done):
        """
        在每个时间步仅存储经验到临时缓冲区。
        """
        # (state, action, reward, next_state, done)
        self.episode_buffer.append((state, action, reward, next_state, done))

    def learn_from_episode(self):
        """
        在Episode结束后，将整个Episode的数据传递给Agent进行批量学习。
        """
        if not self.episode_buffer:
            return

        # 1. 从缓冲区解构数据
        states = [e[0] for e in self.episode_buffer]
        actions = [e[1] for e in self.episode_buffer]
        rewards = np.array([e[2] for e in self.episode_buffer])
        next_states = [e[3] for e in self.episode_buffer]
        terminals = np.array([e[4] for e in self.episode_buffer])

        episode_data = {
            'states': states,
            'actions': actions,
            'rewards': rewards,
            'next_states': next_states,
            'terminals': terminals
        }

        # 2. 调用Agent的核心回合制学习方法
        if hasattr(self.agent, 'learn_from_episode_data'):
            self.agent.learn_from_episode_data(episode_data)

        # 3. 清空缓冲区，为下一个Episode做准备
        self.episode_buffer = []
    
    def save_model(self, path):
        """保存模型"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # 使用PDQN的save_models方法
        if hasattr(self.agent, 'save_models'):
            self.agent.save_models(path)
        elif hasattr(self.agent, 'save'):
            self.agent.save(path)
        print(f"模型已保存到 {path}")
    
    def load_model(self, path):
        """加载模型"""
        # 检查第一个Q网络模型文件是否存在以判断模型是否可用
        actor1_path = path + '_q_actor1.pt'
        if not os.path.exists(actor1_path):
            print(f"警告：模型路径 {actor1_path} 不存在")
            return
        
        # 使用PDQN的load_models方法
        if hasattr(self.agent, 'load_models'):
            self.agent.load_models(path)
        elif hasattr(self.agent, 'load'):
            self.agent.load(path)
        print(f"模型已从 {path} 加载")


class DDPGAdapter(DRLAdapter):
    """DDPG智能体适配器"""
    
    def get_action(self, state):
        """获取动作"""
        # 获取DDPG原始动作
        return self.agent.act(state) if hasattr(self.agent, 'act') else self.agent.select_action(state)
    
    def learn(self, state, action, reward, next_state, done):
        """训练智能体"""
        # 存储经验并学习
        self.agent.store_transition(state, action, reward, next_state, done)
        self.agent.learn()
    
    def save_model(self, path):
        """保存模型"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.agent.save(path)
    
    def load_model(self, path):
        """加载模型"""
        if not os.path.exists(path):
            print(f"警告：模型路径 {path} 不存在")
            return
        self.agent.load(path)


class TD3Adapter(DRLAdapter):
    """TD3智能体适配器"""
    
    def get_action(self, state):
        """获取动作"""
        return self.agent.act(state) if hasattr(self.agent, 'act') else self.agent.select_action(state)
    
    def learn(self, state, action, reward, next_state, done):
        """训练智能体"""
        # 存储经验并学习
        self.agent.store_transition(state, action, reward, next_state, done)
        self.agent.learn()
    
    def save_model(self, path):
        """保存模型"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.agent.save(path)
    
    def load_model(self, path):
        """加载模型"""
        if not os.path.exists(path):
            print(f"警告：模型路径 {path} 不存在")
            return
        self.agent.load(path)


class SACAdapter(DRLAdapter):
    """SAC智能体适配器"""
    
    def get_action(self, state):
        """获取动作"""
        return self.agent.act(state) if hasattr(self.agent, 'act') else self.agent.select_action(state)
    
    def learn(self, state, action, reward, next_state, done):
        """训练智能体"""
        # 存储经验并学习
        self.agent.store_transition(state, action, reward, next_state, done)
        self.agent.learn()
    
    def save_model(self, path):
        """保存模型"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.agent.save(path)
    
    def load_model(self, path):
        """加载模型"""
        if not os.path.exists(path):
            print(f"警告：模型路径 {path} 不存在")
            return
        self.agent.load(path)


class PPOAdapter(DRLAdapter):
    """PPO智能体适配器"""
    
    def get_action(self, state):
        """获取动作"""
        return self.agent.act(state) if hasattr(self.agent, 'act') else self.agent.select_action(state)
    
    def learn(self, state, action, reward, next_state, done):
        """训练智能体"""
        # 存储经验并学习
        self.agent.store_transition(state, action, reward, next_state, done)
        self.agent.learn()
    
    def save_model(self, path):
        """保存模型"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.agent.save(path)
    
    def load_model(self, path):
        """加载模型"""
        if not os.path.exists(path):
            print(f"警告：模型路径 {path} 不存在")
            return
        self.agent.load(path)


class A3CAdapter(DRLAdapter):
    """A3C智能体适配器"""
    
    def get_action(self, state):
        """获取动作"""
        return self.agent.choose_action(state)
    
    def learn(self, state, action, reward, next_state, done):
        """训练智能体"""
        # 直接学习
        self.agent.learn(state, action, reward, next_state, done)
    
    def save_model(self, path):
        """保存模型"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.agent.save(path)
    
    def load_model(self, path):
        """加载模型"""
        if not os.path.exists(path):
            print(f"警告：模型路径 {path} 不存在")
            return
        self.agent.load(path)


def create_drl_agent(args, env, **kwargs):
    """创建DRL智能体，根据指定的算法类型
    
    Args:
        args: 命令行传入的参数，包含DRL超参数
        env: 环境实例
        
    Returns:
        DRL智能体实例
    """
    algorithm = args.drl_algo.lower()
    
    if algorithm == 'pdqn':
        # 导入PDQN实现
        from PDQN.agents.pdqn import PDQNAgent, QActor, ParamActor
        
        # 将命令行参数映射到PDQN的参数字典
        pdqn_params = {
            'batch_size': min(args.drl_batch_size, 256),  # 限制批次大小，提高学习稳定性
            'gamma': args.drl_gamma,
            'replay_memory_size': args.drl_memory_size,
            'learning_rate_actor': args.drl_lr * 0.1,    # 大幅降低学习率
            'learning_rate_actor_param': args.drl_lr * 0.1, # 大幅降低学习率

            # 保留部分固定参数
            'inverting_gradients': True, # 使用梯度反转方案
            'initial_memory_threshold': 32,   # 进一步降低，更早开始学习
            'use_ornstein_noise': True,  # 使用Ornstein噪声
            'epsilon_steps': 30,         # 更快衰减，在30个episode内从探索转向利用
            'epsilon_final': 0.02,       # 最终保持很小的探索率
            'tau_actor': 0.01,       # 降低软更新率，提高稳定性
            'tau_actor_param': 0.005, # 进一步降低参数网络更新率
            'clip_grad': 1.0,        # 梯度裁剪，防止梯度爆炸
            'zero_index_gradients': False,        # 是否将不对应所选动作的动作参数的所有梯度归零
        }
        
        # 新增：为 GNN 架构准备参数
        actor_kwargs = {
            'hidden_dim': args.resnet_hidden_size, # 复用参数
            'num_heads': 4
        }
        actor_param_kwargs = {
            'hidden_dim': args.resnet_hidden_size, # 复用参数
            'num_heads': 4
        }

        # 创建PDQN智能体
        print(f"PDQN智能体使用复合动作空间: {env.action_space}")
        agent = PDQNAgent(env.observation_space, env.action_space, 
                actor_class=QActor, 
                actor_kwargs=actor_kwargs,
                actor_param_class=ParamActor,
                actor_param_kwargs=actor_param_kwargs,
                **pdqn_params)
        return PDQNAdapter(agent, env)
    
    elif algorithm in ['ddpg', 'td3', 'sac', 'ppo', 'a3c']:
        # 这些算法不是本项目的重点，提示使用PDQN
        raise ValueError(f"请使用PDQN算法代替{algorithm.upper()}，其它算法尚未完全支持")
    
    else:
        raise ValueError(f"不支持的DRL算法: {algorithm}") 