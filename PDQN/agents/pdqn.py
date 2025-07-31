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
from PDQN.agents.basis.resnet_basis import ResNetMLP

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
    """
    QActor网络，使用ResNet架构
    """
    def __init__(self, state_size, action_size, action_parameter_size, hidden_size=256, num_blocks=2, **kwargs):
        super(QActor, self).__init__()
        self.resnet = ResNetMLP(
            input_dim=state_size + action_parameter_size,
            output_dim=action_size,
            hidden_size=hidden_size,
            num_blocks=num_blocks
        )

    def forward(self, state, action_parameters):
        state = state.to(torch.float32)
        action_parameters = action_parameters.to(torch.float32)
        x = torch.cat((state, action_parameters), dim=1)
        return self.resnet(x)


class ParamActor(nn.Module):
    """
    ParamActor网络，使用ResNet架构
    """
    def __init__(self, state_size, action_size, action_parameter_size, hidden_size=256, num_blocks=2, **kwargs):
        super(ParamActor, self).__init__()
        self.resnet = ResNetMLP(
            input_dim=state_size,
            output_dim=action_parameter_size,
            hidden_size=hidden_size,
            num_blocks=num_blocks
        )
        
    def forward(self, state):
        state = state.to(torch.float32)
        action_params = self.resnet(state)
        # 对连续动作部分应用sigmoid激活函数
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
                 epsilon_initial=0.5,
                 epsilon_final=0.05,
                 epsilon_steps=1000,
                 batch_size=64,
                 gamma=0.9,
                 tau_actor=0.1,  # Polyak averaging factor for copying target weights
                 tau_actor_param=0.001,
                 replay_memory_size=50000,
                 learning_rate_actor=0.0001,
                 learning_rate_actor_param=0.00001,
                 initial_memory_threshold=0,
                 use_ornstein_noise=False,  # if false, uses epsilon-greedy with uniform-random action-parameter exploration
                 loss_func=F.mse_loss, # F.mse_loss
                 clip_grad=10,
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

        self.use_ornstein_noise = use_ornstein_noise
        # 噪声应该加到总的动作参数向量上去
        self.noise = OrnsteinUhlenbeckActionNoise(self.action_parameter_size, mu=0.0, theta=0.15, sigma=0.2, random_machine=self.np_random)

        print("self.observation_space.shape[0]",self.observation_space.shape[0])

        # 经验回放缓冲区存储总的动作参数向量
        # 确保状态维度与环境提供的维度匹配
        self.replay_memory = Memory(replay_memory_size, self.observation_space.shape, (1 + self.action_parameter_size,), next_actions=False)
        
        self.actor = actor_class(self.observation_space.shape[0], self.num_actions_discrete, self.action_parameter_size, **actor_kwargs).to(device)
        self.actor_target = actor_class(self.observation_space.shape[0], self.num_actions_discrete, self.action_parameter_size, **actor_kwargs).to(device)
        hard_update_target_network(self.actor, self.actor_target)
        self.actor_target.eval()

        self.actor_param = actor_param_class(
            self.observation_space.shape[0], 
            self.num_actions_discrete, 
            self.action_parameter_size, 
            **actor_param_kwargs
        ).to(device)

        self.actor_param_target = actor_param_class(
            self.observation_space.shape[0], 
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
        self.actor_optimiser = optim.Adam(self.actor.parameters(), lr=self.learning_rate_actor) #, betas=(0.95, 0.999))
        self.actor_param_optimiser = optim.Adam(self.actor_param.parameters(), lr=self.learning_rate_actor_param) #, betas=(0.95, 0.999)) #, weight_decay=critic_l2_reg)

    def __str__(self):
        desc = super().__str__() + "\n"
        desc += "Actor Network {}\n".format(self.actor) + \
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

        ep = self._episode
        if ep < self.epsilon_steps:
            self.epsilon = self.epsilon_initial - (self.epsilon_initial - self.epsilon_final) * (
                    ep / self.epsilon_steps)
        else:
            self.epsilon = self.epsilon_final

    def select_action(self, state):
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)

            # 探索 vs 利用
            if self.np_random.uniform() < self.epsilon:
                # 探索: 随机选择动作
                discrete_action = self.discrete_action_space.sample()
                # 生成随机参数并立即扁平化
                continuous_params = self.continuous_action_space.sample().flatten()
            else:
                # 利用: 从网络预测
                all_action_parameters = self.actor_param(state_tensor)
                q_values = self.actor(state_tensor, all_action_parameters)
                
                discrete_action = torch.argmax(q_values).item()
                # 网络输出已经是扁平的
                continuous_params = all_action_parameters[0].cpu().numpy()

            # 对扁平化的连续参数应用噪声
            if self.use_ornstein_noise:
                 continuous_params += self.noise.sample()

            # 对扁平化的参数进行裁剪
            continuous_params = np.clip(continuous_params, self.action_parameter_min_numpy, self.action_parameter_max_numpy)
            
            # 恢复矩阵形状
            continuous_params = continuous_params.reshape(self.continuous_action_space.shape)

        return discrete_action, continuous_params

    def store(self, state, action, reward, next_state, terminals):
        # 处理元组格式的动作输入
        if isinstance(action, tuple) and len(action) == 2:
            discrete_action, continuous_params = action
        else:
            raise TypeError("Action must be a tuple: (discrete_action, continuous_parameters)")
            
        self._step += 1

        # 将动作扁平化存储
        action_flat = np.concatenate(([discrete_action], continuous_params.flatten()))
        
        # 存储样本
        self._add_sample(state, action_flat, reward, next_state, terminals)
        
        if self._step >= self.batch_size and self._step >= self.initial_memory_threshold:
            self._optimize_td_loss()
            self.updates += 1

    def _add_sample(self, state, action, reward, next_state, terminals):
        # 存储到经验回放缓冲区
        self.replay_memory.append(state, action, reward, next_state, terminals)

    def _optimize_td_loss(self):
        if self._step < self.batch_size or self.initial_memory_threshold > self._step:
            return
        # Sample a batch from replay memory
        sample_batch = self.replay_memory.sample(self.batch_size)
        
        if len(sample_batch) == 6:
            states, actions_combined, rewards, next_states, _, terminals = sample_batch
        else:
            states, actions_combined, rewards, next_states, terminals = sample_batch

        states = torch.from_numpy(states).to(self.device)
        actions_combined = torch.from_numpy(actions_combined).to(self.device)
        rewards = torch.from_numpy(rewards).to(self.device).squeeze()
        next_states = torch.from_numpy(next_states).to(self.device)
        terminals = torch.from_numpy(terminals).to(self.device).squeeze()

        # 精确分离离散动作和连续参数
        actions = actions_combined[:, 0].long()
        continuous_parameters = actions_combined[:, 1:]
        
        # ---------------------- optimize Q-network ----------------------
        with torch.no_grad():
            # 获取下一状态的动作参数
            pred_next_continuous_parameters = self.actor_param_target.forward(next_states)
            
            # 使用目标网络评估下一状态的Q值
            pred_Q_a = self.actor_target(next_states, pred_next_continuous_parameters)
            
            # 选择最大Q值作为下一状态的价值
            Qprime, _ = torch.max(pred_Q_a, 1)

            # 计算TD目标
            target = rewards + (1 - terminals) * self.gamma * Qprime

        # 计算当前策略的Q值
        q_values_all = self.actor(states, continuous_parameters)
        
        # 收集与所采取的离散动作对应的Q值
        y_predicted = q_values_all.gather(1, actions.unsqueeze(1)).squeeze(1)

        y_expected = target
        loss_Q = self.loss_func(y_predicted, y_expected)

        # 优化Q网络
        self.actor_optimiser.zero_grad()
        loss_Q.backward()

        # --- 调试代码：检查损失和梯度 ---
        if torch.isnan(loss_Q) or torch.isinf(loss_Q):
            print("!!! FATAL: Loss is NaN or Inf !!!")
            print(f"Loss value: {loss_Q.item()}")

        for name, param in self.actor.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    print(f"!!! FATAL: Gradient for {name} is NaN or Inf !!!")
                    print(param.grad)
        # --- 调试代码结束 ---

        # Clip gradients
        if self.clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.clip_grad)
        self.actor_optimiser.step()

        # ---------------------- optimize actor ----------------------
        with torch.no_grad():
            action_params = self.actor_param(states)
        action_params.requires_grad = True
        
        Q = self.actor(states, action_params)
        Q_val = Q.gather(1, torch.argmax(Q, dim=1).unsqueeze(1)).squeeze(1)
            
        # 计算策略网络的损失
        Q_loss = -torch.mean(Q_val) # 我们希望最大化Q值，所以最小化其负值
            
        # 更新策略网络
        self.actor.zero_grad()
        Q_loss.backward()
        from copy import deepcopy
        delta_a = deepcopy(action_params.grad.data)
        
        # 步骤2：使用actor_param网络生成连续动作参数
        action_params = self.actor_param(Variable(states))
        
        # 应用梯度反转和零梯度索引
        delta_a[:] = self._invert_gradients(delta_a, action_params, grad_type="action_parameters", inplace=True)
        if self.zero_index_gradients:
            delta_a[self.actor.get_action_indices(actions) == 0] = 0.

        out = -torch.mul(delta_a, action_params)
        self.actor_param.zero_grad()
        out.backward(torch.ones(out.shape).to(self.device))
        if self.clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(self.actor_param.parameters(), self.clip_grad)

        self.actor_param_optimiser.step()

        soft_update_target_network(self.actor, self.actor_target, self.tau_actor)
        soft_update_target_network(self.actor_param, self.actor_param_target, self.tau_actor_param)

    def save_models(self, prefix):
        """
        saves the target actor and critic models
        :param prefix: the count of episodes iterated
        :return:
        """
        torch.save(self.actor.state_dict(), prefix + '_actor.pt')
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
        self.actor.load_state_dict(torch.load(prefix + '_actor.pt', map_location='cpu'))
        self.actor_param.load_state_dict(torch.load(prefix + '_actor_param.pt', map_location='cpu'))
        print('Models loaded successfully')
    def greedy_action(self, state):
        """
        执行贪心动作，始终选择最大Q值的动作
        """
        with torch.no_grad():
            state_tensor = torch.from_numpy(state).to(self.device)
            all_action_parameters = self.actor_param.forward(state_tensor)
            
            # 使用actor选择最优动作
            q_values = self.actor.forward(state_tensor.unsqueeze(0), all_action_parameters.unsqueeze(0))
            
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

