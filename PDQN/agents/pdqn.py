import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import math
from collections import Counter
from torch.autograd import Variable
import gym
import os
from gym import spaces
from PDQN.agents.basis.gnn_basis import GNNModel
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader as GraphDataLoader
from config import EPSILON_INITIAL, EPSILON_FINAL, EPSILON_DECAY_STEPS, TAU

# 修正导入路径问题
# from agent import Agent
# from agents.memory.memory import Memory
from PDQN.agents.agent import Agent  # 使用完整的包路径
from PDQN.agents.memory.memory import Memory  # 使用完整的包路径
# from agents.utils import soft_update_target_network, hard_update_target_network
# from agents.utils.noise import OrnsteinUhlenbeckActionNoise

# 噪声
class OrnsteinUhlenbeckActionNoise(object):
    """
    Based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
    Source: https://github.com/vy007vikas/PyTorch-ActorCriticRL/blob/master/utils.py
    """

    def __init__(self, action_dim, mu=0.0, theta=0.15, sigma=0.2, random_machine=None):
        super(OrnsteinUhlenbeckActionNoise, self).__init__()
        self.random = random_machine if random_machine is not None else np.random
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.X = np.ones(self.action_dim) * self.mu

    def reset(self):
        self.X = np.ones(self.action_dim) * self.mu

    def sample(self):
        dx = self.theta * (self.mu - self.X)
        dx = dx + self.sigma * self.random.randn(len(self.X))
        self.X = self.X + dx
        return self.X

# 网络软更新
def soft_update_target_network(source_network, target_network, tau):
    for target_param, param in zip(target_network.parameters(), source_network.parameters()):
        target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)


def hard_update_target_network(source_network, target_network):
    for target_param, param in zip(target_network.parameters(), source_network.parameters()):
        target_param.data.copy_(param.data)


class QActor(nn.Module):
    """ QActor网络，使用GNN架构 """
    def __init__(self, observation_space, action_size, action_parameter_size, hidden_dim=128, num_heads=4, **kwargs):
        super(QActor, self).__init__()
        node_feature_dim = observation_space['x'].shape[1]
        
        # GNN处理图状态
        self.gnn = GNNModel(
            input_dim=node_feature_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim, # GNN输出图嵌入
            num_heads=num_heads
        )
        
        # MLP处理动作参数和图嵌入
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim + action_parameter_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_size)
        )

    def forward(self, state, action_parameters):
        # state现在是一个图数据对象
        graph_embedding = self.gnn(state)
        
        # 拼接图嵌入和动作参数
        combined = torch.cat([graph_embedding, action_parameters], dim=1)
        
        return self.mlp(combined)


class ParamActor(nn.Module):
    """ ParamActor网络，使用GNN架构 """
    def __init__(self, observation_space, action_size, action_parameter_size, hidden_dim=128, num_heads=4, **kwargs):
        super(ParamActor, self).__init__()
        node_feature_dim = observation_space['x'].shape[1]
        
        self.gnn = GNNModel(
            input_dim=node_feature_dim,
            hidden_dim=hidden_dim,
            output_dim=action_parameter_size,
            num_heads=num_heads
        )
        
    def forward(self, state):
        # state现在是一个图数据对象
        action_params = self.gnn(state)
        return torch.sigmoid(action_params)


class PDQNAgent(Agent):
    """
    DDPG actor-critic agent for parameterised action spaces
    [Hausknecht and Stone 2016]
    """

    NAME = "P-DQN Agent"

    def __init__(self,
                 observation_space,
                 action_space,
                 actor_class=QActor,
                 actor_kwargs={},
                 actor_param_class=ParamActor,
                 actor_param_kwargs={},
                 epsilon_initial=EPSILON_INITIAL,
                 epsilon_final=EPSILON_FINAL,
                 epsilon_steps=EPSILON_DECAY_STEPS,
                 batch_size=64,
                 gamma=0.9,
                 tau_actor=TAU,
                 tau_actor_param=TAU,
                 replay_memory_size=50000,
                 learning_rate_actor=0.0001,
                 learning_rate_actor_param=0.00001,
                 initial_memory_threshold=0,
                 use_ornstein_noise=False,  # if false, uses epsilon-greedy with uniform-random action-parameter exploration
                 loss_func=F.mse_loss, # F.mse_loss
                 clip_grad=1.0, # 梯度裁剪上限
                 inverting_gradients=False,
                 zero_index_gradients=False,
                 indexed=False,
                 weighted=False,
                 average=False,
                 random_weighted=False,
                 device="cuda" if torch.cuda.is_available() else "cpu",
                 seed=None):
        super(PDQNAgent, self).__init__(observation_space, action_space)
        self.device = torch.device(device)
        
        # 动作空间维度
        if isinstance(action_space, spaces.Tuple) and len(action_space.spaces) == 2:
            self.discrete_action_space = action_space.spaces[0]
            self.continuous_action_space = action_space.spaces[1]
            
            if not isinstance(self.discrete_action_space, spaces.Discrete):
                raise TypeError("Discrete action space must be of type gym.spaces.Discrete")
            if not isinstance(self.continuous_action_space, spaces.Box):
                raise TypeError("Continuous action space must be of type gym.spaces.Box")
                
            self.num_actions_discrete = self.discrete_action_space.n
            self.num_actions_continuous = int(np.prod(self.continuous_action_space.shape))
        else:
            raise TypeError("Action space must be a gym.spaces.Tuple of (Discrete, Box)")

        self.action_parameter_size = self.num_actions_continuous

        self.action_max = torch.from_numpy(self.continuous_action_space.high).float().to(device)
        self.action_min = torch.from_numpy(self.continuous_action_space.low).float().to(device)
        self.action_range = (self.action_max - self.action_min).detach()

        self.action_parameter_max_numpy = self.continuous_action_space.high.flatten()
        self.action_parameter_min_numpy = self.continuous_action_space.low.flatten()
        self.action_parameter_range_numpy = (self.action_parameter_max_numpy - self.action_parameter_min_numpy)
        
        self.actions_count = 0
        
        self.action_parameter_max = torch.from_numpy(self.action_parameter_max_numpy).float().to(self.device)
        self.action_parameter_min = torch.from_numpy(self.action_parameter_min_numpy).float().to(self.device)
        self.action_parameter_range = torch.from_numpy(self.action_parameter_range_numpy).float().to(self.device)

        self.epsilon = epsilon_initial
        self.epsilon_initial = epsilon_initial
        self.epsilon_final = epsilon_final
        self.epsilon_steps = epsilon_steps
        
        # 动作参数偏移量 (现在包含离散和连续部分)
        self.action_parameter_offsets = np.array([0, self.num_actions_discrete, self.action_parameter_size])

        self.batch_size = batch_size
        self.gamma = gamma
        self.replay_memory_size = replay_memory_size
        self.initial_memory_threshold = initial_memory_threshold
        self.learning_rate_actor = learning_rate_actor
        self.learning_rate_actor_param = learning_rate_actor_param
        self.inverting_gradients = inverting_gradients
        self.tau_actor = tau_actor
        self.tau_actor_param = tau_actor_param
        self._step = 0
        self._episode = 0
        self.action_epsilon = 0
        self.updates = 0
        self.clip_grad = clip_grad
        self.zero_index_gradients = zero_index_gradients

        self.np_random = np.random.RandomState()
        self.seed = seed
        self._seed(seed)
        
        # 新增：用于控制冷启动学习的标志
        self.memory_preheated = False

        self.use_ornstein_noise = use_ornstein_noise
        # 噪声应该加到总的动作参数向量上去
        self.noise = OrnsteinUhlenbeckActionNoise(self.action_parameter_size, mu=0.0, theta=0.15, sigma=0.2, random_machine=self.np_random)

        self.replay_memory = Memory(replay_memory_size)
        
        # --- Q-Network 1 ---
        self.q_actor1 = actor_class(self.observation_space, self.num_actions_discrete, self.action_parameter_size, **actor_kwargs).to(device)
        self.q_actor_target1 = actor_class(self.observation_space, self.num_actions_discrete, self.action_parameter_size, **actor_kwargs).to(device)
        hard_update_target_network(self.q_actor1, self.q_actor_target1)
        self.q_actor_target1.eval()

        # --- Q-Network 2 ---
        self.q_actor2 = actor_class(self.observation_space, self.num_actions_discrete, self.action_parameter_size, **actor_kwargs).to(device)
        self.q_actor_target2 = actor_class(self.observation_space, self.num_actions_discrete, self.action_parameter_size, **actor_kwargs).to(device)
        hard_update_target_network(self.q_actor2, self.q_actor_target2)
        self.q_actor_target2.eval()

        self.actor_param = actor_param_class(
            self.observation_space, 
            self.num_actions_discrete, 
            self.action_parameter_size, 
            **actor_param_kwargs
        ).to(device)

        self.actor_param_target = actor_param_class(
            self.observation_space, 
            self.num_actions_discrete, 
            self.action_parameter_size, 
            **actor_param_kwargs
        ).to(device)
        hard_update_target_network(self.actor_param, self.actor_param_target)
        self.actor_param_target.eval()

        self.loss_func = loss_func  # l1_smooth_loss performs better but original paper used MSE

        # Original DDPG paper [Lillicrap et al. 2016] used a weight decay of 0.01 for Q (critic)
        # but setting weight_decay=0.01 on the critic_optimiser seems to perform worse...
        # using AMSgrad ("fixed" version of Adam, amsgrad=True) doesn't seem to help either...
        self.q_actor1_optimiser = optim.Adam(self.q_actor1.parameters(), lr=self.learning_rate_actor)
        self.q_actor2_optimiser = optim.Adam(self.q_actor2.parameters(), lr=self.learning_rate_actor)
        self.actor_param_optimiser = optim.Adam(self.actor_param.parameters(), lr=self.learning_rate_actor_param) #, betas=(0.95, 0.999)) #, weight_decay=critic_l2_reg)

        # 添加学习率调度器
        # self.actor_lr_scheduler = optim.lr_scheduler.ExponentialLR(self.actor_optimiser, gamma=0.99)
        # self.param_lr_scheduler = optim.lr_scheduler.ExponentialLR(self.actor_param_optimiser, gamma=0.99)

    def __str__(self):
        desc = super().__str__() + "\n"
        desc += "Actor Network 1 {}\n".format(self.q_actor1) + \
                "Actor Network 2 {}\n".format(self.q_actor2) + \
                "Param Network {}\n".format(self.actor_param) + \
                "Actor Alpha: {}\n".format(self.learning_rate_actor) + \
                "Actor Param Alpha: {}\n".format(self.learning_rate_actor_param) + \
                "Gamma: {}\n".format(self.gamma) + \
                "Tau (actor): {}\n".format(self.tau_actor) + \
                "Tau (actor-params): {}\n".format(self.tau_actor_param) + \
                "Inverting Gradients: {}\n".format(self.inverting_gradients) + \
                "Replay Memory: {}\n".format(self.replay_memory_size) + \
                "Batch Size: {}\n".format(self.batch_size) + \
                "Initial memory: {}\n".format(self.initial_memory_threshold) + \
                "epsilon_initial: {}\n".format(self.epsilon_initial) + \
                "epsilon_final: {}\n".format(self.epsilon_final) + \
                "epsilon_steps: {}\n".format(self.epsilon_steps) + \
                "Clip Grad: {}\n".format(self.clip_grad) + \
                "Ornstein Noise?: {}\n".format(self.use_ornstein_noise) + \
                "Zero Index Grads?: {}\n".format(self.zero_index_gradients) + \
                "Seed: {}\n".format(self.seed)
        return desc

    # 人为调整神经网络的权重
    def set_action_parameter_passthrough_weights(self, initial_weights, initial_bias=None):
        passthrough_layer = self.actor_param.action_parameters_passthrough_layer
        print(initial_weights.shape)
        print(passthrough_layer.weight.data.size())
        assert initial_weights.shape == passthrough_layer.weight.data.size()
        passthrough_layer.weight.data = torch.Tensor(initial_weights).float().to(self.device)
        if initial_bias is not None:
            print(initial_bias.shape)
            print(passthrough_layer.bias.data.size())
            assert initial_bias.shape == passthrough_layer.bias.data.size()
            passthrough_layer.bias.data = torch.Tensor(initial_bias).float().to(self.device)
        
        # 修复requires_grad设置
        for param in passthrough_layer.parameters():
            param.requires_grad_(False)

        hard_update_target_network(self.actor_param, self.actor_param_target)

    def _seed(self, seed=None):
        """
        NOTE: this will not reset the randomly initialised weights; use the seed parameter in the constructor instead.

        :param seed:
        :return:
        """
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        
        # 确保np_random被正确初始化
        if seed is not None:
            self.np_random = np.random.RandomState(seed=seed)
        else:
            self.np_random = np.random.RandomState()
            
        if seed is not None:
            torch.manual_seed(seed)
            if self.device == torch.device("cuda"):
                torch.cuda.manual_seed(seed)

    def _ornstein_uhlenbeck_noise(self, all_action_parameters):
        """ Continuous action exploration using an Ornstein–Uhlenbeck process. """
        return all_action_parameters.data.numpy() + (self.noise.sample() * self.action_parameter_range_numpy)

    def start_episode(self):
        pass

    def end_episode(self):
        self._episode += 1

        # 在每个 episode 结束时，步进学习率调度器 (如果使用)
        # if self._episode > 0:
        #     self.actor_lr_scheduler.step()
        #     self.param_lr_scheduler.step()

        ep = self._episode
        if ep < self.epsilon_steps:
            self.epsilon = self.epsilon_initial - (self.epsilon_initial - self.epsilon_final) * (
                    ep / self.epsilon_steps)
        else:
            self.epsilon = self.epsilon_final

    def select_action(self, state):
        with torch.no_grad():
            # 将单个图数据对象放到设备上
            state.to(self.device)

            # 步骤 1: 始终通过策略网络计算确定性动作
            all_action_parameters = self.actor_param(state)
            # 在决策时，只使用第一个Q网络来选择动作
            q_values = self.q_actor1(state, all_action_parameters)
            
            discrete_action = torch.argmax(q_values).item()
            continuous_params = all_action_parameters[0].cpu().numpy()

            # 步骤 2: 在连续参数空间添加噪声以进行探索
            if self.use_ornstein_noise:
                 # Epsilon控制噪声的大小，实现从探索到利用的平滑过渡
                 noise = self.epsilon * self.noise.sample()
                 continuous_params += noise

            # 步骤 3: 裁剪参数确保在有效范围内
            continuous_params = np.clip(continuous_params, self.action_parameter_min_numpy, self.action_parameter_max_numpy)
            
            # 步骤 4: 恢复矩阵形状
            continuous_params = continuous_params.reshape(self.continuous_action_space.shape)

        return discrete_action, continuous_params

    def _invert_gradients(self, grad, vals, grad_type, inplace=True):
        if grad_type == "actions":
            max_p = self.action_max
            min_p = self.action_min
            rnge = self.action_range
        elif grad_type == "action_parameters":
            max_p = self.action_parameter_max
            min_p = self.action_parameter_min
            rnge = self.action_parameter_range
        else:
            raise ValueError("Unhandled grad_type: '"+str(grad_type) + "'")

        max_p = max_p.cpu()
        min_p = min_p.cpu()
        rnge = rnge.cpu()
        grad = grad.cpu()
        vals = vals.cpu()

        assert grad.shape == vals.shape

        if not inplace:
            grad = grad.clone()
        with torch.no_grad():
            index = grad > 0
            grad[index] *= (index.float() * (max_p - vals) / rnge)[index]
            grad[~index] *= ((~index).float() * (vals - min_p) / rnge)[~index]

        return grad

    def _zero_index_gradients(self, grad, batch_action_indices, inplace=True):
        assert grad.shape[0] == batch_action_indices.shape[0]
        grad = grad.cpu()

        if not inplace:
            grad = grad.clone()
        with torch.no_grad():
            ind = torch.zeros(self.action_parameter_size, dtype=torch.long)
            for a in range(self.num_actions_discrete):
                ind[self.action_parameter_offsets[a]:self.action_parameter_offsets[a+1]] = a
            ind_tile = ind.repeat(self.batch_size, 1).to(self.device)
            actual_index = ind_tile != batch_action_indices[:, np.newaxis]
            grad[actual_index] = 0.
        return grad

    def store(self, state, action, reward, next_state, terminals):
        # 确保图数据在正确的设备上
        state.to(self.device)
        next_state.to(self.device)

        # 处理元组格式的动作输入
        if isinstance(action, tuple) and len(action) == 2:
            discrete_action, continuous_params = action
        else:
            raise TypeError("Action must be a tuple: (discrete_action, continuous_parameters)")
            
        self._step += 1

        # 将动作扁平化存储
        action_flat = np.concatenate(([discrete_action], continuous_params.flatten()))
        
        # 存储样本，但不立即学习
        self._add_sample(state, action_flat, reward, next_state, terminals)

    def learn_from_episode_data(self, episode_data, num_updates=100):
        """
        在Episode结束后，使用修正后的奖励进行批量学习
        
        Args:
            episode_data: 包含 'states', 'actions', 'rewards', 'next_states', 'terminals' 的字典
            num_updates: 在一个Episode结束后执行的梯度更新次数
        """
        # 1. 计算未来累积折扣回报
        rewards = episode_data['rewards']
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(0, len(rewards))):
            running_add = running_add * self.gamma + rewards[t]
            discounted_rewards[t] = running_add

        # 2. 将修正后的经验存入回放池 (只调用_add_sample以避免_step重复增加)
        for i in range(len(rewards)):
            state = episode_data['states'][i]
            action = episode_data['actions'][i]
            reward = discounted_rewards[i]
            next_state = episode_data['next_states'][i]
            terminal = episode_data['terminals'][i]
            
            # 确保图数据在设备上
            state.to(self.device)
            next_state.to(self.device)
            
            # 扁平化动作
            discrete_action, continuous_params = action
            action_flat = np.concatenate(([discrete_action], continuous_params.flatten()))

            self._add_sample(state, action_flat, reward, next_state, terminal)

        # 3. 检查是否可以开始学习
        if not self.memory_preheated:
            # 检查是否达到了冷启动的经验阈值
            if self._step >= self.initial_memory_threshold:
                print(f"--- 经验池预热完成 (共 {self._step} 步)，开始学习 ---")
                self.memory_preheated = True
            else:
                print(f"--- 正在预热经验池 ({self._step}/{self.initial_memory_threshold} 步)... ---")
                return # 如果未达到阈值，则跳过本轮学习

        # 4. 一旦预热完成，每个Episode结束后都进行批量学习
        if self.memory_preheated:
            print(f"--- Episode结束，开始进行 {num_updates} 次批量学习 ---")
            for _ in range(num_updates):
                # 确保有足够的样本进行采样
                if self._step >= self.batch_size:
                    self._optimize_td_loss()
                    self.updates += 1
            print(f"--- 批量学习完成 ---")

    def _add_sample(self, state, action, reward, next_state, terminals):
        # 存储到经验回放缓冲区
        self.replay_memory.append(state, action, reward, next_state, terminals)

    def _optimize_td_loss(self):
        if self._step < self.batch_size or self.initial_memory_threshold > self._step:
            return
        # Sample a batch from replay memory
        states, actions_combined, rewards, next_states, terminals = self.replay_memory.sample(self.batch_size)
        
        # 使用GraphDataLoader来处理图数据的批处理
        states = Batch.from_data_list(list(states)).to(self.device)
        next_states = Batch.from_data_list(list(next_states)).to(self.device)

        actions_combined = torch.from_numpy(actions_combined).float().to(self.device)
        rewards = torch.from_numpy(rewards).float().to(self.device).squeeze()
        terminals = torch.from_numpy(terminals).to(self.device).squeeze()

        # 精确分离离散动作和连续参数
        actions = actions_combined[:, 0].long()
        continuous_parameters = actions_combined[:, 1:]
        
        # ---------------------- optimize Q-network ----------------------
        with torch.no_grad():
            # 获取下一状态的动作参数
            pred_next_continuous_parameters = self.actor_param_target(next_states)
            
            # 使用两个目标网络分别评估下一状态的Q值
            pred_Q_a1 = self.q_actor_target1(next_states, pred_next_continuous_parameters)
            pred_Q_a2 = self.q_actor_target2(next_states, pred_next_continuous_parameters)
            
            # 选择两个评估中较小的那个Q值 (Clipped Double Q-learning)
            Qprime, _ = torch.max(torch.min(pred_Q_a1, pred_Q_a2), 1)

            # 计算TD目标
            target = rewards + (1 - terminals) * self.gamma * Qprime

        # 计算当前策略的Q值 (两个Q网络都需要计算)
        q_values1 = self.q_actor1(states, continuous_parameters)
        q_values2 = self.q_actor2(states, continuous_parameters)
        
        # 收集与所采取的离散动作对应的Q值
        y_predicted1 = q_values1.gather(1, actions.unsqueeze(1)).squeeze(1)
        y_predicted2 = q_values2.gather(1, actions.unsqueeze(1)).squeeze(1)

        y_expected = target
        loss_Q1 = self.loss_func(y_predicted1, y_expected)
        loss_Q2 = self.loss_func(y_predicted2, y_expected)
        loss_Q = loss_Q1 + loss_Q2

        # 优化两个Q网络
        self.q_actor1_optimiser.zero_grad()
        self.q_actor2_optimiser.zero_grad()
        loss_Q.backward()

        # --- 调试代码：检查损失和梯度 ---
        if torch.isnan(loss_Q) or torch.isinf(loss_Q):
            print("!!! FATAL: Loss is NaN or Inf !!!")
            print(f"Loss value: {loss_Q.item()}")

        # for name, param in self.actor.named_parameters():
        #     if param.grad is not None:
        #         if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
        #             print(f"!!! FATAL: Gradient for {name} is NaN or Inf !!!")
        #             print(param.grad)
        # --- 调试代码结束 ---

        # Clip gradients
        if self.clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(self.q_actor1.parameters(), self.clip_grad)
            torch.nn.utils.clip_grad_norm_(self.q_actor2.parameters(), self.clip_grad)
        self.q_actor1_optimiser.step()
        self.q_actor2_optimiser.step()

        # ---------------------- optimize actor ----------------------
        with torch.no_grad():
            action_params = self.actor_param(states)
        action_params.requires_grad = True
        
        # 在更新Actor时，我们只使用第一个Q网络的评估值
        Q = self.q_actor1(states, action_params)
        Q_val = Q.gather(1, torch.argmax(Q, dim=1).unsqueeze(1)).squeeze(1)
            
        # 计算策略网络的损失
        Q_loss = -torch.mean(Q_val) # 我们希望最大化Q值，所以最小化其负值
            
        # 更新策略网络
        # self.actor.zero_grad() # 此行错误且多余，q_actor的梯度已在前面处理
        Q_loss.backward()
        from copy import deepcopy
        delta_a = deepcopy(action_params.grad.data)
        
        # 步骤2：使用actor_param网络生成连续动作参数
        action_params = self.actor_param(states)
        
        # 应用梯度反转和零梯度索引
        delta_a[:] = self._invert_gradients(delta_a, action_params, grad_type="action_parameters", inplace=True)
        if self.zero_index_gradients:
            delta_a[self.actor.get_action_indices(actions) == 0] = 0.

        out = -torch.mul(delta_a, action_params)
        self.actor_param.zero_grad()
        out.backward(torch.ones(out.shape).to(self.device))

        # 对 ParamActor 的梯度也进行裁剪
        if self.clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(self.actor_param.parameters(), self.clip_grad)

        self.actor_param_optimiser.step()

        soft_update_target_network(self.q_actor1, self.q_actor_target1, self.tau_actor)
        soft_update_target_network(self.q_actor2, self.q_actor_target2, self.tau_actor)
        soft_update_target_network(self.actor_param, self.actor_param_target, self.tau_actor_param)

    def save_models(self, prefix):
        """
        saves the target actor and critic models
        :param prefix: the count of episodes iterated
        :return:
        """
        torch.save(self.q_actor1.state_dict(), prefix + '_q_actor1.pt')
        torch.save(self.q_actor2.state_dict(), prefix + '_q_actor2.pt')
        torch.save(self.actor_param.state_dict(), prefix + '_actor_param.pt')
        print('Models saved successfully')

    def load_models(self, prefix):
        """
        loads the target actor and critic models, and copies them onto actor and critic models
        :param prefix: the count of episodes iterated (used to find the file name)
        :param target: whether to load the target newtwork too (not necessary for evaluation)
        :return:
        """
        # also try load on CPU if no GPU available?
        self.q_actor1.load_state_dict(torch.load(prefix + '_q_actor1.pt', map_location='cpu'))
        self.q_actor2.load_state_dict(torch.load(prefix + '_q_actor2.pt', map_location='cpu'))
        self.actor_param.load_state_dict(torch.load(prefix + '_actor_param.pt', map_location='cpu'))
        print('Models loaded successfully')
    def greedy_action(self, state):
        """
        执行贪心动作，始终选择最大Q值的动作
        """
        with torch.no_grad():
            # state_tensor = torch.from_numpy(state).to(self.device)
            all_action_parameters = self.actor_param.forward(state)
            
            # 使用actor选择最优动作
            q_values = self.q_actor1.forward(state.unsqueeze(0), all_action_parameters.unsqueeze(0))
            
            # 将扁平化的Q值重塑为 (决策数量, 每个决策的选项数=2)
            q_values_reshaped = q_values.view(self.num_actions_discrete, 2)
            
            # 为每个独立决策选择最优动作
            discrete_action_flat = torch.argmax(q_values_reshaped, dim=1).cpu().numpy()

            # 将扁平化的动作向量转换回元组结构
            actions_list = []
            current_pos = 0
            for dim in self.discrete_component_dims:
                if dim == 1:
                    actions_list.append(discrete_action_flat[current_pos])
                else:
                    actions_list.append(discrete_action_flat[current_pos : current_pos + dim])
                current_pos += dim
            discrete_action = tuple(actions_list)
            
            # 转换连续参数为numpy数组
            all_action_parameters = all_action_parameters.cpu().data.numpy()
            
            # 确保参数在正确范围内
            all_action_parameters = np.clip(all_action_parameters, 
                                    self.action_parameter_min_numpy[:len(all_action_parameters)],
                                    self.action_parameter_max_numpy[:len(all_action_parameters)])
            
            continuous_params_only = all_action_parameters[self.num_actions_discrete:]

            return discrete_action, continuous_params_only

