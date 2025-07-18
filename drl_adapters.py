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
        """在每个时间步后调用此方法来训练智能体。
        
        此方法封装了"存储经验"和"执行一步学习"两个过程。
        子类必须实现此方法。
        """
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
            动作向量，扁平化的一维数组
        """
        # 确保状态是numpy数组
        if isinstance(state, list):
            state = np.array(state, dtype=np.float32)
        
        # 使用PDQN智能体获取动作
        action = self.agent.select_action(state)
        
        # 确保返回的是一维numpy数组
        if not isinstance(action, np.ndarray):
            action = np.array(action, dtype=np.float32)
        
        # 确保动作在正确的范围内
        action = np.clip(action, 0.0, 1.0)
        
        return action
    
    def learn(self, state, action, reward, next_state, done):
        """训练智能体"""
        # 确保所有输入都是numpy数组
        if isinstance(state, list):
            state = np.array(state, dtype=np.float32)
        if isinstance(action, list):
            action = np.array(action, dtype=np.float32)
        if isinstance(next_state, list):
            next_state = np.array(next_state, dtype=np.float32)
        
        # 直接存储经验，不需要修改动作向量
        # PDQNAgent的store方法内部会调用_optimize_td_loss进行学习
        self.agent.store(state, action, reward, next_state, done)
    
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
        if not os.path.exists(path + '_actor.pt'):
            print(f"警告：模型路径 {path}_actor.pt 不存在")
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
        action = self.agent.act(state) if hasattr(self.agent, 'act') else self.agent.select_action(state)
        
        # 确保动作是numpy数组
        if not isinstance(action, np.ndarray):
            action = np.array(action, dtype=np.float32)
        
        # 确保动作在正确的范围内
        action = np.clip(action, 0.0, 1.0)
        
        return action
    
    def learn(self, state, action, reward, next_state, done):
        """训练智能体"""
        # 存储经验
        self.agent.store_transition(state, action, reward, next_state, done)
            
        # 如果有足够的样本，就进行更新
        if hasattr(self.agent, 'memory_counter') and self.agent.memory_counter > self.agent.batch_size:
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
        action = self.agent.act(state) if hasattr(self.agent, 'act') else self.agent.select_action(state)
        return np.clip(action, 0.0, 1.0)
    
    def learn(self, state, action, reward, next_state, done):
        """训练智能体"""
        # 存储经验
        self.agent.store_transition(state, action, reward, next_state, done)
            
        # 如果有足够的样本，就进行更新
        if hasattr(self.agent, 'memory_counter') and self.agent.memory_counter > self.agent.batch_size:
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
        action = self.agent.act(state) if hasattr(self.agent, 'act') else self.agent.select_action(state)
        return np.clip(action, 0.0, 1.0)
    
    def learn(self, state, action, reward, next_state, done):
        """训练智能体"""
        # 存储经验
        self.agent.store_transition(state, action, reward, next_state, done)
            
        # 如果有足够的样本，就进行更新
        if hasattr(self.agent, 'memory_counter') and self.agent.memory_counter > self.agent.batch_size:
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
        action = self.agent.act(state) if hasattr(self.agent, 'act') else self.agent.select_action(state)
        return np.clip(action, 0.0, 1.0)
    
    def learn(self, state, action, reward, next_state, done):
        """训练智能体"""
        # PPO是on-policy，每个时间步都存储经验
        self.agent.store_transition(state, action, reward, next_state, done)
        
        # PPO agent内部有自己的逻辑决定何时学习 (例如，当缓冲区满时)
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
        # A3C是on-policy且通常是异步的，每个worker直接使用经验学习
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
            'batch_size': args.drl_batch_size,
            'gamma': args.drl_gamma,
            'replay_memory_size': args.drl_memory_size,
            'learning_rate_actor': args.drl_lr,
            
            # 保留部分固定参数
            'inverting_gradients': True, # 使用梯度反转方案
            'initial_memory_threshold': 128,  # 开始学习所需的转换数量
            'use_ornstein_noise': True,  # 使用Ornstein噪声
            'epsilon_steps': 100,        # 线性衰减epsilon的episode数量
            'epsilon_final': 0.01,      # 最终epsilon值
            'tau_actor': 0.05,       # 软目标网络更新因子
            'tau_actor_param': 0.01, # 参数Actor网络软更新因子
            'learning_rate_actor_param': 0.0005,  # 参数Actor网络学习率
            'clip_grad': 1.0,                     # 参数梯度裁剪限制
            'zero_index_gradients': False,        # 是否将不对应所选动作的动作参数的所有梯度归零
        }
        
        # 创建PDQN智能体
        agent = PDQNAgent(env.observation_space, env.action_space, 
                actor_class=QActor, 
                actor_param_class=ParamActor,
                actor_param_kwargs={
                    'hidden_layers': (256, 128, 64)  # 隐藏层结构
                },
                **pdqn_params)
        return PDQNAdapter(agent, env)
    
    elif algorithm == 'ddpg':
        # 导入DDPG实现
        from elsedrl import DDPG
        
        # DDPG优化参数配置
        ddpg_params = {
            'batch_size': 256,              # 批次大小
            'gamma': 0.99,                  # 折扣因子
            'tau': 0.005,                   # 软更新系数
            'buffer_size': 20000,           # 经验回放缓冲区大小
            'learning_rate_actor': 0.0001,  # Actor学习率
            'learning_rate_critic': 0.001,  # Critic学习率
            'hidden_sizes': [256, 256],     # 隐藏层大小
        }
        
        # 更新默认参数
        for key, value in ddpg_params.items():
            if key not in kwargs:
                kwargs[key] = value
        
        # 创建DDPG智能体
        agent = DDPG(env.observation_space, env.action_space, **kwargs)
        return DDPGAdapter(agent, env)
    
    elif algorithm == 'td3':
        # 导入TD3实现
        from elsedrl import TD3
        
        # TD3优化参数配置
        td3_params = {
            'batch_size': 256,              # 批次大小
            'gamma': 0.99,                  # 折扣因子
            'tau': 0.005,                   # 软更新系数
            'buffer_size': 20000,           # 经验回放缓冲区大小
            'learning_rate_actor': 0.0001,  # Actor学习率
            'learning_rate_critic': 0.001,  # Critic学习率
            'policy_noise': 0.2,            # 策略噪声
            'noise_clip': 0.5,              # 噪声裁剪
            'policy_delay': 2,              # 策略更新延迟
            'hidden_sizes': [256, 256],     # 隐藏层大小
        }
        
        # 更新默认参数
        for key, value in td3_params.items():
            if key not in kwargs:
                kwargs[key] = value
        
        # 创建TD3智能体
        agent = TD3(env.observation_space, env.action_space, **kwargs)
        return TD3Adapter(agent, env)
    
    elif algorithm == 'sac':
        # 导入SAC实现
        from elsedrl import SAC
        
        # SAC优化参数配置
        sac_params = {
            'batch_size': 256,              # 批次大小
            'gamma': 0.99,                  # 折扣因子
            'tau': 0.005,                   # 软更新系数
            'buffer_size': 20000,           # 经验回放缓冲区大小
            'learning_rate_actor': 0.0003,  # Actor学习率
            'learning_rate_critic': 0.0003, # Critic学习率
            'learning_rate_alpha': 0.0003,  # 温度参数学习率
            'hidden_sizes': [256, 256],     # 隐藏层大小
            'alpha': 0.2,                   # 初始温度参数
            'auto_entropy': True,           # 自动调整熵正则化系数
        }
        
        # 更新默认参数
        for key, value in sac_params.items():
            if key not in kwargs:
                kwargs[key] = value
        
        # 创建SAC智能体
        agent = SAC(env.observation_space, env.action_space, **kwargs)
        return SACAdapter(agent, env)
    
    elif algorithm == 'ppo':
        # 导入PPO实现
        from elsedrl import PPO
        
        # PPO优化参数配置
        ppo_params = {
            'batch_size': 64,               # 批次大小
            'gamma': 0.99,                  # 折扣因子
            'clip_param': 0.2,              # 裁剪参数
            'gae_lambda': 0.95,             # GAE lambda参数
            'learning_rate': 0.0003,        # 学习率
            'value_loss_coef': 0.5,         # 价值损失系数
            'entropy_coef': 0.01,           # 熵正则化系数
            'max_grad_norm': 0.5,           # 梯度裁剪
            'num_mini_batch': 4,            # mini-batch数量
            'ppo_epoch': 10,                # PPO更新轮数
            'hidden_sizes': [64, 64],       # 隐藏层大小
        }
        
        # 更新默认参数
        for key, value in ppo_params.items():
            if key not in kwargs:
                kwargs[key] = value
        
        # 创建PPO智能体
        agent = PPO(env.observation_space, env.action_space, **kwargs)
        return PPOAdapter(agent, env)
    
    elif algorithm == 'a3c':
        # 导入A3C实现
        from elsedrl import A3C
        
        # A3C优化参数配置
        a3c_params = {
            'gamma': 0.99,                  # 折扣因子
            'learning_rate': 0.0001,        # 学习率
            'value_loss_coef': 0.5,         # 价值损失系数
            'entropy_coef': 0.01,           # 熵正则化系数
            'max_grad_norm': 40,            # 梯度裁剪
            'hidden_sizes': [128, 128],     # 隐藏层大小
        }
        
        # 更新默认参数
        for key, value in a3c_params.items():
            if key not in kwargs:
                kwargs[key] = value
        
        # 创建A3C智能体
        agent = A3C(env.observation_space, env.action_space, **kwargs)
        return A3CAdapter(agent, env)
    
    else:
        raise ValueError(f"不支持的DRL算法: {algorithm}") 