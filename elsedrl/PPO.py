import math
import os
import random
import torch as T
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Beta

class ContinuousActorNetwork(nn.Module):
    def __init__(self, n_actions, input_dims, alpha,
                 fc1_dims=400, fc2_dims=300, chkpt_dir='models/'):
        super(ContinuousActorNetwork, self).__init__()
        self.checkpoint_file = os.path.join(chkpt_dir,
                                            'actor_continuous_ppo')
        self.fc1 = nn.Linear(input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.alpha = nn.Linear(fc2_dims, n_actions)
        self.beta = nn.Linear(fc2_dims, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        # print("stateeee",state)
        x = T.tanh(self.fc1(state))
        x = T.tanh(self.fc2(x))
        alpha = F.softplus(self.alpha(x)) 
        beta = F.softplus(self.beta(x)) 

        # print("alooo",alpha.size())
        # if torch.isnan(alpha).any() or torch.isnan(beta).any():
        #     alpha = torch.rand(1, 75) + 1.0  # 可以根据你的需求进行调整
        #     beta = torch.rand(1, 75) + 1.0 
        
        dist = Beta(alpha, beta)
        # print("dist",dist)
        return dist

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))


class ContinuousCriticNetwork(nn.Module):
    def __init__(self, input_dims, n_actions,alpha,
                 fc1_dims=400, fc2_dims=300, chkpt_dir='models/'):
        super(ContinuousCriticNetwork, self).__init__()
        self.checkpoint_file = os.path.join(chkpt_dir,
                                            'critic_continuous_ppo')
        self.fc1 = nn.Linear(input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.v = nn.Linear(fc2_dims, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = T.tanh(self.fc1(state))
        x = T.tanh(self.fc2(x))
        v = self.v(x)

        return v

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))


import numpy as np


class PPOMemory:
    def __init__(self, batch_size):
        self.states = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.new_states = []

        self.batch_size = batch_size

    def recall(self):
        return np.array(self.states),\
            np.array(self.new_states),\
            np.array(self.actions),\
            np.array(self.probs),\
            np.array(self.rewards),\
            np.array(self.dones)

    def generate_batches(self):
        n_states = len(self.states)
        # batch_start = np.arange(0, n_states, self.batch_size)
        n_batches = int(n_states // self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        # batches = [indices[i:i+self.batch_size] for i in batch_start]
        batches = [indices[i*self.batch_size:(i+1)*self.batch_size]
                   for i in range(n_batches)]
        return batches

    def store_memory(self, state, state_, action, probs, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.rewards.append(reward)
        self.dones.append(done)
        self.new_states.append(state_)

    def clear_memory(self):
        self.states = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.new_states = []


import torch as T
# from memory import PPOMemory
# from networks import ContinuousActorNetwork, ContinuousCriticNetwork


class PPO:
    def __init__(self, 
                  observation_space, action_space ,
                 n_actions, input_dims, gamma=0.99, 
                 alpha=1e-4,
                 lr=1e-3,
                 lr_c = 1e-4,
                 update_iteration = 20,
                 gae_lambda=0.95, policy_clip=0.1, seed=10, batch_size=64 , device = "cpu"):
        self.observation_space = observation_space
        self.action_space = action_space
        self.gamma = 0.80 #  gamma
        self.policy_clip = policy_clip
        self.n_epochs = 10 # update_iteration
        self.gae_lambda = gae_lambda
        self.entropy_coefficient = 1e-3
        ''' new add'''
        self.seed = seed
        self._seed(seed)
        self.lr = lr
        self.lr_c = lr_c 
        self.batch_size = batch_size
        self.device = device
        self.action_epsilon = 0



        self.num_actions_discrete = action_space.spaces[0].nvec.shape[0]#  默认动作空间中离散参数的维度
        self.action_parameter_sizes = np.array([ ( action_space.spaces[1].shape[0] * action_space.spaces[1].shape[1] ) ]) # 这里应该是连续的动作空间的数量，二维
        self.num_actions_continuous_1 = self.action_parameter_sizes[0] # 发射功率那个维度的连续动作空间的数量
        self.action_parameter_size = self.num_actions_discrete + int(self.action_parameter_sizes.sum()) #  这里其实应该是所有的动作空间的参数
        self.num_actions = self.num_actions_discrete + self.num_actions_continuous_1  # + self.num_actions_continuous_2  # 离散的动作+连续的动作 总的数量
        # 设置动作空间的最大最小值和范围
        discrete_action_max = 1
        discrete_action_min = 0
        continuous_action_power_max = 1 # 发射功率的最大值
        continuous_action_ratio_max = 1 # 卸载比例的最大值
        continuous_action_min = 0
        self.action_max = torch.from_numpy(np.concatenate([
            np.full(self.num_actions_discrete, discrete_action_max),
            np.full(self.num_actions_continuous_1, continuous_action_power_max),
        ])).float().to(device)
        self.action_min = torch.from_numpy(np.concatenate([
            np.full(self.num_actions_discrete, discrete_action_min),
            np.full(self.num_actions_continuous_1, continuous_action_min),
        ])).float().to(device)
        self.action_range = (self.action_max - self.action_min).detach()

        self.action_parameter_max_numpy = np.concatenate([
            np.full(self.num_actions_discrete, discrete_action_max),
            np.full(self.num_actions_continuous_1, continuous_action_power_max),
        ]).ravel()
        self.action_parameter_min_numpy = np.concatenate([
            np.full(self.num_actions_discrete, discrete_action_min),
            np.full(self.num_actions_continuous_1, continuous_action_min),
        ]).ravel()
        
        

        self.actor = ContinuousActorNetwork(n_actions,
                                            input_dims, alpha)
        self.critic = ContinuousCriticNetwork(input_dims,n_actions, alpha)
        self.memory = PPOMemory(batch_size)

    def remember(self, state, state_, action, probs, reward, done):
        self.memory.store_memory(state, state_, action,
                                 probs, reward, done)

    def save_models(self):
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()

    def select_action(self, observation):
        with T.no_grad():
            state = T.tensor([observation], dtype=T.float).to(
                    self.actor.device)
            dist = self.actor(state)
            action = dist.sample()
            probs = dist.log_prob(action)
            
            epsilon_end = 0.01
            epsilon_start = 1 
            epsilon_decay= 10000 # 越大越探索吧  # 上次能到了6000
            self.epsilon = epsilon_end + (epsilon_start - epsilon_end) * \
                        math.exp(-1. * self.action_epsilon / epsilon_decay)
            self.action_epsilon += 1
            rnd = self.np_random.uniform()

            if rnd < self.epsilon:
                action = torch.from_numpy(self.np_random.uniform(self.action_parameter_min_numpy,
                                                            self.action_parameter_max_numpy))
            else:
                action = self.action_min + action[0] * (self.action_max - self.action_min)

            # print("action",action)
            
        return action.cpu().numpy().flatten(),  probs.cpu().numpy().flatten()

    def calc_adv_and_returns(self, memories):
        states, new_states, r, dones = memories
        with T.no_grad():
            values = self.critic(states)
            values_ = self.critic(new_states)
            deltas = r + self.gamma * values_ - values
            deltas = deltas.cpu().flatten().numpy()
            adv = [0]
            for dlt, mask in zip(deltas[::-1], dones[::-1]):
                advantage = dlt + self.gamma * self.gae_lambda * adv[-1] * \
                            (1 - mask)
                adv.append(advantage)
            adv.reverse()
            adv = adv[:-1]
            adv = T.tensor(adv).float().unsqueeze(1).to(self.critic.device)
            # print('adv', adv)
            returns = adv + values
            # print('returns', returns)
            adv = (adv - adv.mean()) / (adv.std()+1e-4)
        return adv, returns

    def update(self):
        # print("afasfsaf")
        state_arr, new_state_arr, action_arr, old_prob_arr,\
            reward_arr, dones_arr = \
            self.memory.recall()
        state_arr = T.tensor(state_arr, dtype=T.float).to(
                self.critic.device)
        action_arr = T.tensor(action_arr, dtype=T.float).to(
                self.critic.device)
        old_prob_arr = T.tensor(old_prob_arr, dtype=T.float).to(
                self.critic.device)
        new_state_arr = T.tensor(new_state_arr, dtype=T.float).to(
                self.critic.device)
        r = T.tensor(reward_arr, dtype=T.float).unsqueeze(1).to(
                self.critic.device)
        adv, returns = self.calc_adv_and_returns((state_arr, new_state_arr,
                                                 r, dones_arr))
        for epoch in range(self.n_epochs):
            batches = self.memory.generate_batches()
            for batch in batches:
                states = state_arr[batch]
                old_probs = old_prob_arr[batch]
                actions = action_arr[batch]
                ''' 不然后续有很多问题的 '''
                # actions = torch.clamp(actions, 0, 1)

                dist = self.actor(states)
                ''' 截断一下到0-1的范围  '''
                actions = torch.sigmoid(actions)
                new_probs = dist.log_prob(actions)

                prob_ratio = T.exp(new_probs.sum(1, keepdim=True) - old_probs.
                                   sum(1, keepdim=True))
                # print('probs ratio', prob_ratio.shape)
                weighted_probs = adv[batch] * prob_ratio
                weighted_clipped_probs = T.clamp(
                        prob_ratio, 1-self.policy_clip, 1+self.policy_clip) * \
                    adv[batch]
                # print('weighted clipped probs', weighted_clipped_probs.shape)
                entropy = dist.entropy().sum(1, keepdims=True)
                # print('entropy', entropy.shape)
                actor_loss = -T.min(weighted_probs,
                                    weighted_clipped_probs)
                actor_loss -= self.entropy_coefficient * entropy
                # print('actor loss', actor_loss.shape)
                # exit()
                self.actor.optimizer.zero_grad()
                actor_loss.mean().backward()
                T.nn.utils.clip_grad_norm_(self.actor.parameters(), 40)
                self.actor.optimizer.step()

                critic_value = self.critic(states)
                critic_loss = (critic_value - returns[batch]).pow(2).mean()
                self.critic.optimizer.zero_grad()
                critic_loss.backward()
                self.critic.optimizer.step()
        self.memory.clear_memory()

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
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        if seed is not None:
            torch.manual_seed(seed)
            # if self.device == torch.device("cuda"):
            #     torch.cuda.manual_seed(seed)

