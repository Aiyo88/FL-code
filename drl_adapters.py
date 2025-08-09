import os
import torch
import numpy as np
from typing import Any, Dict, List, Tuple, Union
from gym import spaces

# --- 核心算法与模型导入 ---
from PDQN.agents.pdqn import PDQNAgent, QActor, ParamActor


class DRLAdapter:
    """DRL算法统一适配器基类 - 为不同DRL算法提供统一接口"""
    
    def __init__(self, agent, env):
        self.agent = agent
        self.env = env
        self.device = getattr(agent, 'device', torch.device("cpu"))
        self.episode_buffer = [] # 用于回合制学习的数据暂存
    
    def get_action(self, state):
        raise NotImplementedError

    def learn(self, state, action, reward, next_state, done):
        # 在每个时间步，仅将经验暂存到回合缓冲区
        self.episode_buffer.append((state, action, reward, next_state, done))

    def learn_from_episode(self):
        raise NotImplementedError

    def save_model(self, path):
        raise NotImplementedError
    
    def load_model(self, path):
        raise NotImplementedError


class PDQNAdapter(DRLAdapter):
    """(魔改) PDQN算法适配器"""
    
    def __init__(self, agent, env):
        super().__init__(agent, env)
    
    def get_action(self, state):
        """获取单个整数动作"""
        return self.agent.select_action(state)
    
    def learn(self, state, action, reward, next_state, done):
        """在每个时间步，暂存经验到缓冲区。"""
        self.episode_buffer.append((state, action, reward, next_state, done))

    def learn_from_episode(self):
        """在Episode结束后，将整个Episode的数据传递给Agent进行批量学习。"""
        if not self.episode_buffer:
            return

        # 1. 解构数据
        episode_data = {
            'states': [e[0] for e in self.episode_buffer],
            'actions': [e[1] for e in self.episode_buffer],
            'rewards': np.array([e[2] for e in self.episode_buffer]),
            'next_states': [e[3] for e in self.episode_buffer],
            'terminals': np.array([e[4] for e in self.episode_buffer])
        }

        # 2. 调用Agent的核心学习方法
        if hasattr(self.agent, 'learn_from_episode_data'):
            self.agent.learn_from_episode_data(episode_data)

        # 3. 清空缓冲区
        self.episode_buffer = []

    def save_model(self, path):
        self.agent.save_models(path)
    
    def load_model(self, path):
        actor1_path = path + '_q_actor1.pt'
        if not os.path.exists(actor1_path):
            print(f"警告：模型路径 {actor1_path} 不存在")
            return
        self.agent.load_models(path)


def create_drl_agent(args, env, **kwargs):
    """
    创建 (魔改) PDQN 智能体。
    """
    algorithm = args.drl_algo.lower()
    
    if algorithm == 'pdqn':
        # 定义上下文向量的大小
        context_vector_size = 128

        pdqn_params = {
            'batch_size': args.drl_batch_size,
            'gamma': args.drl_gamma,
            'replay_memory_size': args.drl_memory_size,
            'learning_rate_actor': args.critic_lr,
            'learning_rate_actor_param': args.actor_lr,
            'initial_memory_threshold': args.drl_batch_size # 确保至少有一个batch的数据才开始学习
        }
        
        actor_kwargs = {
            'hidden_dim': args.resnet_hidden_size,
            'num_heads': 4,
            'context_vector_size': context_vector_size
        }
        actor_param_kwargs = {
            'hidden_dim': args.resnet_hidden_size,
            'num_heads': 4,
            'context_vector_size': context_vector_size
        }

        agent = PDQNAgent(
            observation_space=env.observation_space,
            action_space=env.action_space,
            actor_class=QActor,
            actor_kwargs=actor_kwargs,
            actor_param_class=ParamActor,
            actor_param_kwargs=actor_param_kwargs,
            **pdqn_params
        )
        return PDQNAdapter(agent, env)
    
    else:
        raise ValueError(f"当前配置仅支持 'pdqn' 算法, 而不是 {algorithm}")


def add_drl_args(parser):
    """为DRL算法添加特定的命令行参数"""
    from config import CRITIC_LR, ACTOR_LR, DRL_BATCH_SIZE, DRL_GAMMA, DRL_MEMORY_SIZE
    
    parser.add_argument('--critic_lr', type=float, default=CRITIC_LR, help='DRL Critic (QActor) 网络学习率')
    parser.add_argument('--actor_lr', type=float, default=ACTOR_LR, help='DRL Actor (ParamActor) 网络学习率')
    parser.add_argument('--drl_batch_size', type=int, default=DRL_BATCH_SIZE, help='DRL 批次大小')
    parser.add_argument('--drl_gamma', type=float, default=DRL_GAMMA, help='DRL 折扣因子')
    parser.add_argument('--drl_memory_size', type=int, default=DRL_MEMORY_SIZE, help='DRL 回放缓存区大小')
    return parser 