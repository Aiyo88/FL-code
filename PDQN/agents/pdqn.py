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

    def __init__(self, action_dim, mu=0, theta=0.15, sigma=0.2, random_machine=np.random):
        super(OrnsteinUhlenbeckActionNoise, self).__init__()
        self.random = random_machine
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

    def __init__(self, state_size, action_size, action_parameter_size, hidden_layers=(100,), action_input_layer=0,
                 output_layer_init_std=None, activation="relu", **kwargs):
        super(QActor, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.action_parameter_size = action_parameter_size
        self.activation = activation

        # create layers
        self.layers = nn.ModuleList()
        inputSize = self.state_size + self.action_parameter_size
        lastHiddenLayerSize = inputSize
        if hidden_layers is not None:
            nh = len(hidden_layers)
            self.layers.append(nn.Linear(inputSize, hidden_layers[0]))
            for i in range(1, nh):
                self.layers.append(nn.Linear(hidden_layers[i - 1], hidden_layers[i]))
            lastHiddenLayerSize = hidden_layers[nh - 1]
        self.layers.append(nn.Linear(lastHiddenLayerSize, self.action_size))

        # initialise layer weights
        for i in range(0, len(self.layers) - 1):
            nn.init.kaiming_normal_(self.layers[i].weight, nonlinearity=activation)
            nn.init.zeros_(self.layers[i].bias)
        if output_layer_init_std is not None:
            nn.init.normal_(self.layers[-1].weight, mean=0., std=output_layer_init_std)
        # else:
        #     nn.init.zeros_(self.layers[-1].weight)
        nn.init.zeros_(self.layers[-1].bias)


        

    def forward(self, state, action_parameters):
        # 确保输入为float32
        state = state.to(torch.float32)
        action_parameters = action_parameters.to(torch.float32)
        
        negative_slope = 0.01
        x = torch.cat((state, action_parameters), dim=1)
        
        num_layers = len(self.layers)
        for i in range(0, num_layers - 1):
            if self.activation == "relu":
                x = F.relu(self.layers[i](x))
            elif self.activation == "leaky_relu":
                x = F.leaky_relu(self.layers[i](x), negative_slope)
            else:
                raise ValueError("Unknown activation function "+str(self.activation))
        Q = self.layers[-1](x)
        return Q


class ParamActor(nn.Module):

    def __init__(self, state_size, action_size, action_parameter_size, hidden_layers, squashing_function=False,
                 output_layer_init_std=None, init_type="kaiming", activation="relu", init_std=None):
        super(ParamActor, self).__init__()

        self.state_size = state_size
        self.action_size = action_size
        self.action_parameter_size = action_parameter_size
        self.squashing_function = squashing_function
        self.activation = activation
        if init_type == "normal":
            assert init_std is not None and init_std > 0
        assert self.squashing_function is False  # unsupported, cannot get scaling right yet

        # create layers
        self.layers = nn.ModuleList()
        inputSize = self.state_size
        lastHiddenLayerSize = inputSize
        if hidden_layers is not None:
            nh = len(hidden_layers)
            self.layers.append(nn.Linear(inputSize, hidden_layers[0]))
            for i in range(1, nh):
                self.layers.append(nn.Linear(hidden_layers[i - 1], hidden_layers[i]))
            lastHiddenLayerSize = hidden_layers[nh - 1]
        self.action_parameters_output_layer = nn.Linear(lastHiddenLayerSize, self.action_parameter_size)
        self.action_parameters_passthrough_layer = nn.Linear(self.state_size, self.action_parameter_size)

        # initialise layer weights
        for i in range(0, len(self.layers)):
            if init_type == "kaiming":
                nn.init.kaiming_normal_(self.layers[i].weight, nonlinearity=activation)
            elif init_type == "normal":
                nn.init.normal_(self.layers[i].weight, std=init_std)
            else:
                raise ValueError("Unknown init_type "+str(init_type))
            nn.init.zeros_(self.layers[i].bias)
        if output_layer_init_std is not None:
            nn.init.normal_(self.action_parameters_output_layer.weight, std=output_layer_init_std)
        else:
            nn.init.zeros_(self.action_parameters_output_layer.weight)
        nn.init.zeros_(self.action_parameters_output_layer.bias)

        nn.init.zeros_(self.action_parameters_passthrough_layer.weight)
        nn.init.zeros_(self.action_parameters_passthrough_layer.bias)

        # fix passthrough layer to avoid instability, rest of network can compensate
        self.action_parameters_passthrough_layer.requires_grad = False
        self.action_parameters_passthrough_layer.weight.requires_grad = False
        self.action_parameters_passthrough_layer.bias.requires_grad = False

    def forward(self, state):
        # 确保输入为float32
        state = state.to(torch.float32)
        
        x = state
        negative_slope = 0.01
        num_hidden_layers = len(self.layers)
        for i in range(0, num_hidden_layers):
            if self.activation == "relu":
                x = F.relu(self.layers[i](x))
            elif self.activation == "leaky_relu":
                x = F.leaky_relu(self.layers[i](x), negative_slope)
            else:
                raise ValueError("Unknown activation function "+str(self.activation))
        action_params = self.action_parameters_output_layer(x)
        action_params += self.action_parameters_passthrough_layer(state)

        return action_params


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
        
        # 检查动作空间类型
        if hasattr(action_space, 'spaces'):
            # 原有代码处理复合动作空间
            self.num_actions_discrete = action_space.spaces[0].nvec.shape[0]
            self.action_parameter_sizes = np.array([(action_space.spaces[1].shape[0] * action_space.spaces[1].shape[1])])
            self.num_actions_continuous_1 = self.action_parameter_sizes[0]  # 连续动作空间的数量
        else:
            # 处理Box类型动作空间 - 假设前44维是离散部分，后30维是连续部分
            self.num_actions_discrete = 44
            self.action_parameter_sizes = np.array([30])
            self.num_actions_continuous_1 = 30  # 连续动作空间的数量
        
        # 计算总动作参数大小
        self.action_parameter_size = self.num_actions_discrete + int(self.action_parameter_sizes.sum())
        self.num_actions = self.num_actions_discrete + self.num_actions_continuous_1  # 离散的动作+连续的动作 总的数量
        
        print("self.num_actions_discrete", self.num_actions_discrete)
        print("self.num_actions_continuous_1", self.num_actions_continuous_1)
        print("self.action_parameter_size", self.action_parameter_size)
        print("self.num_actions", self.num_actions)
        
        # 设置动作空间的最大最小值和范围
        discrete_action_max = 1
        discrete_action_min = 0
        continuous_action_max = 1  # 资源分配最大值
        continuous_action_min = 0  # 资源分配最小值
        
        # 创建动作边界向量
        self.action_max = torch.from_numpy(np.concatenate([
            np.full(self.num_actions_discrete, discrete_action_max),
            np.full(self.num_actions_continuous_1, continuous_action_max)
        ])).float().to(device)
        
        self.action_min = torch.from_numpy(np.concatenate([
            np.full(self.num_actions_discrete, discrete_action_min),
            np.full(self.num_actions_continuous_1, continuous_action_min)
        ])).float().to(device)
        
        self.action_range = (self.action_max - self.action_min).detach()
        
        # 为numpy数组创建相同的边界
        self.action_parameter_max_numpy = np.concatenate([
            np.full(self.num_actions_discrete, discrete_action_max),
            np.full(self.num_actions_continuous_1, continuous_action_max)
        ]).ravel()
        
        self.action_parameter_min_numpy = np.concatenate([
            np.full(self.num_actions_discrete, discrete_action_min),
            np.full(self.num_actions_continuous_1, continuous_action_min)
        ]).ravel()
        
        self.action_parameter_range_numpy = (self.action_parameter_max_numpy - self.action_parameter_min_numpy)
        
        self.actions_count = 0
        
        self.action_parameter_max = torch.from_numpy(self.action_parameter_max_numpy).float().to(device)
        self.action_parameter_min = torch.from_numpy(self.action_parameter_min_numpy).float().to(device)
        self.action_parameter_range = torch.from_numpy(self.action_parameter_range_numpy).float().to(device)
        
        self.epsilon = epsilon_initial
        self.epsilon_initial = epsilon_initial
        self.epsilon_final = epsilon_final
        self.epsilon_steps = epsilon_steps
        self.indexed = indexed
        self.weighted = weighted
        self.average = average
        self.random_weighted = random_weighted
        assert (weighted ^ average ^ random_weighted) or not (weighted or average or random_weighted)

        # 设置动作参数偏移
        self.action_parameter_offsets = self.action_parameter_sizes.cumsum()
        self.action_parameter_offsets = np.insert(self.action_parameter_offsets, 0, 0)

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



        self.np_random = None
        self.seed = seed
        self._seed(seed)

        self.use_ornstein_noise = use_ornstein_noise
        # 噪声应该加到连续的动作空间向量上去
        self.noise = OrnsteinUhlenbeckActionNoise(self.action_parameter_size, random_machine=self.np_random, mu=0., theta=0.15, sigma=0.0001) #, theta=0.01, sigma=0.01)

        print("self.num_actions+self.action_parameter_size",self.num_actions+self.action_parameter_size)

        print("self.observation_space.shape[0]",self.observation_space.shape[0])

        self.replay_memory = Memory(replay_memory_size, observation_space.shape, (self.action_parameter_size,), next_actions=False)
        # self.replay_memory = Memory(replay_memory_size, observation_space.shape, (1+self.action_parameter_size,), next_actions=False)
        
        self.actor = actor_class(self.observation_space.shape[0], self.num_actions, self.action_parameter_size, **actor_kwargs).to(device)
        self.actor_target = actor_class(self.observation_space.shape[0], self.num_actions, self.action_parameter_size, **actor_kwargs).to(device)
        hard_update_target_network(self.actor, self.actor_target)
        self.actor_target.eval()

        # 定义隐藏层结构（可以根据需要调整）
        default_hidden_layers = (256, 128, 64)  # 使用一个合理的默认值

        # 检查 kwargs 中是否包含 hidden_layers
        hidden_layers = actor_param_kwargs.pop('hidden_layers', default_hidden_layers)

        # 添加 hidden_layers 作为位置参数
        self.actor_param = actor_param_class(
            self.observation_space.shape[0], 
            self.num_actions, 
            self.action_parameter_size, 
            hidden_layers,  # 添加 hidden_layers 参数
            **actor_param_kwargs
        ).to(device)

        self.actor_param_target = actor_param_class(
            self.observation_space.shape[0], 
            self.num_actions, 
            self.action_parameter_size, 
            hidden_layers,  # 添加 hidden_layers 参数
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
        passthrough_layer.requires_grad = False
        passthrough_layer.weight.requires_grad = False
        passthrough_layer.bias.requires_grad = False

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
        self.np_random = np.random.RandomState(seed=seed)
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
            state = torch.from_numpy(state).to(self.device)
            all_action_parameters = self.actor_param.forward(state)

            # 动态衰减的ε-贪婪策略
            epsilon_end = 0.01
            epsilon_start = 1 
            epsilon_decay = 3000
            
            self.epsilon = epsilon_end + (epsilon_start - epsilon_end) * \
                          math.exp(-1. * self.actions_count / epsilon_decay)
            self.actions_count += 1

            # 以epsilon的概率随机探索
            rnd = self.np_random.uniform()
            if rnd < self.epsilon:
                # 随机探索生成的动作参数，范围在[min, max]之间
                all_action_parameters = torch.from_numpy(np.random.uniform(
                    self.action_parameter_min_numpy[:self.action_parameter_size],
                    self.action_parameter_max_numpy[:self.action_parameter_size]))
            else:
                # 根据策略选择动作
                Q_a = self.actor.forward(state.unsqueeze(0), all_action_parameters.unsqueeze(0))
                Q_a = Q_a.detach().cpu().data.numpy()

            # 转换为numpy数组并返回扁平数组
            all_action_parameters = all_action_parameters.cpu().data.numpy()
            
            # 确保参数在正确范围内
            all_action_parameters = np.clip(all_action_parameters, 
                                          self.action_parameter_min_numpy[:len(all_action_parameters)],
                                          self.action_parameter_max_numpy[:len(all_action_parameters)])
            
            return all_action_parameters  # 返回扁平数组

    def random_action(self,state):
        all_action_parameters = torch.from_numpy(np.random.uniform(
            self.action_parameter_min_numpy[:self.action_parameter_size],
            self.action_parameter_max_numpy[:self.action_parameter_size]))
        all_action_parameters = all_action_parameters.cpu().data.numpy()
        
        return all_action_parameters

    def _zero_index_gradients(self, grad, batch_action_indices, inplace=True):
        assert grad.shape[0] == batch_action_indices.shape[0]
        grad = grad.cpu()

        if not inplace:
            grad = grad.clone()
        with torch.no_grad():
            ind = torch.zeros(self.action_parameter_size, dtype=torch.long)
            for a in range(self.num_actions):
                ind[self.action_parameter_offsets[a]:self.action_parameter_offsets[a+1]] = a
            # ind_tile = np.tile(ind, (self.batch_size, 1))
            ind_tile = ind.repeat(self.batch_size, 1).to(self.device)
            actual_index = ind_tile != batch_action_indices[:, np.newaxis]
            grad[actual_index] = 0.
        return grad

    def _invert_gradients(self, grad, vals, grad_type, inplace=True):
        # 5x faster on CPU (for Soccer, slightly slower for Goal, Platform?)
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
            # index = grad < 0  # actually > but Adam minimises, so reversed (could also double negate the grad)
            index = grad > 0
            grad[index] *= (index.float() * (max_p - vals) / rnge)[index]
            grad[~index] *= ((~index).float() * (vals - min_p) / rnge)[~index]

        return grad

    def store(self, state, action, reward, next_state, terminals):
        all_action_parameters = action
        self._step += 1

        # self._add_sample(state, np.concatenate((all_actions.data, all_action_parameters.data)).ravel(), reward, next_state, terminal)
        self._add_sample(state, all_action_parameters, reward, next_state, terminals)
        if self._step >= self.batch_size and self._step >= self.initial_memory_threshold:
            self._optimize_td_loss()
            self.updates += 1

    def _add_sample(self, state, action, reward, next_state, terminals):
        # 直接存储完整的动作向量
        self.replay_memory.append(state, action, reward, next_state, terminals)

    def _optimize_td_loss(self):
        if self._step < self.batch_size or self._step < self.initial_memory_threshold:
            return
        # Sample a batch from replay memory
        states, actions_combined, rewards, next_states, terminals = self.replay_memory.sample(self.batch_size, random_machine=self.np_random)

        states = torch.from_numpy(states).to(self.device)
        actions_combined = torch.from_numpy(actions_combined).to(self.device)
        rewards = torch.from_numpy(rewards).to(self.device).squeeze()
        next_states = torch.from_numpy(next_states).to(self.device)
        terminals = torch.from_numpy(terminals).to(self.device).squeeze()

        # 将动作参数分为离散部分和连续部分
        # 前 self.num_actions_discrete 个参数是离散动作
        # 后续参数是连续动作参数
        discrete_actions = actions_combined[:, :self.num_actions_discrete]
        continuous_parameters = actions_combined[:, self.num_actions_discrete:]
        
        # 对于Q值评估，我们使用离散动作索引
        # 找到每个样本中最大值对应的索引作为动作
        values, indices = torch.max(discrete_actions, dim=1)
        actions = indices.long()
        
        # ---------------------- optimize Q-network ----------------------
        with torch.no_grad():
            # 获取下一状态的动作参数
            pred_next_action_parameters = self.actor_param_target.forward(next_states)
            # 使用目标网络评估下一状态的Q值
            pred_Q_a = self.actor_target(next_states, pred_next_action_parameters)
            # 选择最大Q值
            Qprime = torch.max(pred_Q_a, 1, keepdim=True)[0].squeeze()

            # 计算TD目标
            target = rewards + (1 - terminals) * self.gamma * Qprime

        # 计算当前策略的Q值
        q_values = self.actor(states, actions_combined)
        y_predicted = q_values.gather(1, actions.view(-1, 1)).squeeze()
        y_expected = target
        loss_Q = self.loss_func(y_predicted, y_expected)

        # 优化Q网络
        self.actor_optimiser.zero_grad()
        loss_Q.backward()
        if self.clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.clip_grad)
        self.actor_optimiser.step()

        # ---------------------- optimize actor ----------------------
        with torch.no_grad():
            action_params = self.actor_param(states)
        action_params.requires_grad = True
        
        # 计算策略网络输出的Q值
        Q = self.actor(states, action_params)
        Q_val = Q
        
        # 根据设置应用加权策略
        if self.weighted:
            # 使用动作计数统计作为权重
            counts = Counter(actions.cpu().numpy())
            weights = torch.from_numpy(
                np.array([counts[a] / actions.shape[0] for a in range(self.num_actions)])).float().to(self.device)
            Q_val = weights * Q
        elif self.average:
            Q_val = Q / self.num_actions
        elif self.random_weighted:
            weights = np.random.uniform(0, 1., self.num_actions)
            weights /= np.linalg.norm(weights)
            weights = torch.from_numpy(weights).float().to(self.device)
            Q_val = weights * Q
            
        # 计算策略网络的损失
        if self.indexed:
            Q_indexed = Q_val.gather(1, actions.unsqueeze(1))
            Q_loss = torch.mean(Q_indexed)
        else:
            Q_loss = torch.mean(torch.sum(Q_val, 1))
            
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
            delta_a[:] = self._zero_index_gradients(batch_action_indices=actions, inplace=True)

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
            state = torch.from_numpy(state).to(self.device)
            all_action_parameters = self.actor_param.forward(state)
            
            # 使用actor选择最优动作
            Q_a = self.actor.forward(state.unsqueeze(0), all_action_parameters.unsqueeze(0))
            Q_a = Q_a.detach().cpu().data.numpy()
            
            # 不使用epsilon探索，直接选择最大Q值的动作
            all_action_parameters = all_action_parameters.cpu().data.numpy()
            
        return all_action_parameters

