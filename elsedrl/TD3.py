import argparse
from collections import namedtuple
from itertools import count

import os, sys, random
import numpy as np

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
# from tensorboardX import SummaryWriter
import math


'''
Implementation of TD3 with pytorch 
Original paper: https://arxiv.org/abs/1802.09477
Not the author's implementation !
'''

class Replay_buffer():
    '''
    Code based on:
    https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py
    Expects tuples of (state, next_state, action, reward, done)
    '''
    def __init__(self, max_size,random_machine):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0
        self.random_machine = random_machine

    def push(self, data):
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)

    def sample(self, batch_size):
        ind = self.random_machine.randint(0, len(self.storage), size=batch_size)
        x, y, u, r, d = [], [], [], [], []

        for i in ind:
            X, Y, U, R, D = self.storage[i]
            x.append(np.array(X, copy=False))
            y.append(np.array(Y, copy=False))
            u.append(np.array(U, copy=False))
            r.append(np.array(R, copy=False))
            d.append(np.array(D, copy=False))

        return np.array(x), np.array(y), np.array(u), np.array(r).reshape(-1, 1), np.array(d).reshape(-1, 1)


class Actor(nn.Module):

    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.fc1 = nn.Linear(state_dim, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, action_dim)

        # self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.fc1(state))
        a = F.relu(self.fc2(a))
        # a = torch.tanh(self.fc3(a)) * self.max_action
        a = torch.sigmoid(self.fc3(a))
        return a

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, action_dim)

    def forward(self, state, action):
        # print("state shape:", state.shape)
        # print("action shape:", action.shape)
        
        state_action = torch.cat([state, action], 1)
        # print("state_action shape:", state_action.shape)

        # print("fc1 weight shape:", self.fc1.weight.shape)

        
        q = F.relu(self.fc1(state_action))
        q = F.relu(self.fc2(q))
        q = torch.sigmoid(self.fc3(q))
        
        return q


class TD3():
    def __init__(self, observation_space, action_space , state_dim, action_dim, max_action, 
                                        batch_size,lr,lr_c,
                                        tau,capacity,exploration_noise,seed, update_iteration, num_actions_discrete, num_actions_continuous_1,policy_noise
                                        ,policy_delay ,gamma=0.85, device = "cpu"):
        self.observation_space = observation_space
        self.action_space = action_space
        self.seed = seed
        self._seed(seed)
        
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.critic_1 = Critic(state_dim, action_dim).to(device)
        self.critic_1_target = Critic(state_dim, action_dim).to(device)
        self.critic_2 = Critic(state_dim, action_dim).to(device)
        self.critic_2_target = Critic(state_dim, action_dim).to(device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_1_optimizer = optim.Adam(self.critic_1.parameters(), lr=lr_c)
        self.critic_2_optimizer = optim.Adam(self.critic_2.parameters(), lr=lr_c)


        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_1_target.load_state_dict(self.critic_1.state_dict())
        self.critic_2_target.load_state_dict(self.critic_2.state_dict())

        self.max_action = max_action
        self.memory = Replay_buffer(capacity,np.random)
        # self.writer = SummaryWriter(directory)
        self.num_critic_update_iteration = 0
        self.num_actor_update_iteration = 0
        self.num_training = 0

        ''' 新增 '''
        self.batch_size = batch_size

        self.gamma = gamma[0] if isinstance(gamma, tuple) else gamma

        self.tau = tau
        self.capacity = capacity
        self.exploration_noise = exploration_noise

        self.update_iteration = update_iteration
        self.num_actions_discrete = num_actions_discrete
        self.num_actions_continuous_1 = num_actions_continuous_1
        self.policy_noise = policy_noise
        self.policy_delay = policy_delay
        self.device = device

        self.action_epsilon = 0

        self.noise_clip = 0.5

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
        

    def select_action(self, state):
        with torch.no_grad():
            state = torch.tensor(state.reshape(1, -1)).float().to(self.device)
            action = self.actor(state)
            


            epsilon_end = 0.01
            epsilon_start = 1 
            epsilon_decay=10000 # 越大越探索吧  # 

            self.epsilon = epsilon_end + (epsilon_start - epsilon_end) * \
                        math.exp(-1. * self.action_epsilon / epsilon_decay)
            self.action_epsilon += 1


            rnd = self.np_random.uniform()
            if rnd < self.epsilon:
                action = torch.from_numpy(self.np_random.uniform(self.action_parameter_min_numpy,
                                                            self.action_parameter_max_numpy))
            else:
                # action = torch.sigmoid(action)  # 将输出映射到 (0, 1) 范围
                action = self.action_min + action[0] * (self.action_max - self.action_min)
                

            action = action.cpu().data.numpy()
            # print("action",action)
        return action

    def update(self):
        for i in range(self.update_iteration):
            x, y, u, r, d = self.memory.sample(self.batch_size)
            state = torch.FloatTensor(x).to(self.device)
            action = torch.FloatTensor(u).to(self.device)
            next_state = torch.FloatTensor(y).to(self.device)
            done = torch.FloatTensor(d).to(self.device)
            reward = torch.FloatTensor(r).to(self.device)

            # Select next action according to target policy:
            # noise = torch.ones_like(action).data.normal_(0, self.policy_noise).to(self.device)
            # # noise = noise.clamp(-self.policy_noise_clip, self.policy_noise_clip)
            # next_action = (self.actor_target(next_state) + noise).clamp(0, 1)
            # next_action = self.action_min + next_action[0] * (self.action_max - self.action_min)

            noise = torch.ones_like(action).data.normal_(0, self.policy_noise).to(self.device)
            noise = noise.clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_state) + noise)
            next_action = self.action_min + next_action * (self.action_max - self.action_min)

            # next_action = next_action.clamp(0, self.max_action)

            # print("action",action.size())
            # print("next_action",next_action.size())
            # print("next_state",next_state.size())

            target_Q1 = self.critic_1_target(next_state, next_action)
            target_Q2 = self.critic_2_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)

            # print("target_Q",target_Q.detach().size())
            # print("reward",reward.size())
            # print("target_Q dtype:", target_Q.dtype)
            # self.gamma = self.gamma.to(torch.float32)
            # self.gamma = float(self.gamma)

            target_Q = reward +  self.gamma * target_Q.detach()

            # Optimize Critic 1:
            current_Q1 = self.critic_1(state, action)
            loss_Q1 = F.mse_loss(current_Q1, target_Q)
            self.critic_1_optimizer.zero_grad()
            loss_Q1.backward()
            self.critic_1_optimizer.step()
            # self.writer.add_scalar('Loss/Q1_loss', loss_Q1, global_step=self.num_critic_update_iteration)

            # Optimize Critic 2:
            current_Q2 = self.critic_2(state, action)
            loss_Q2 = F.mse_loss(current_Q2, target_Q)
            self.critic_2_optimizer.zero_grad()
            loss_Q2.backward()
            self.critic_2_optimizer.step()
            # self.writer.add_scalar('Loss/Q2_loss', loss_Q2, global_step=self.num_critic_update_iteration)

            # Delayed policy updates:
            if i % self.policy_delay == 0:
                # Compute actor loss:
                actor_loss = -self.critic_1(state, self.actor(state)).mean()

                # Optimize the actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()
                # self.writer.add_scalar('Loss/actor_loss', actor_loss, global_step=self.num_actor_update_iteration)

                # Update target networks using a soft update:
                self.soft_update_target_networks()

                self.num_actor_update_iteration += 1

        self.num_critic_update_iteration += 1
        self.num_training += 1

    def soft_update_target_networks(self):
        # print("tauuu",self.tau)
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(((1 - self.tau) * target_param.data) + self.tau * param.data)

        for param, target_param in zip(self.critic_1.parameters(), self.critic_1_target.parameters()):
            target_param.data.copy_(((1 - self.tau) * target_param.data) + self.tau * param.data)

        for param, target_param in zip(self.critic_2.parameters(), self.critic_2_target.parameters()):
            target_param.data.copy_(((1 - self.tau) * target_param.data) + self.tau * param.data)


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

    # def save(self):
    #     torch.save(self.actor.state_dict(), directory+'actor.pth')
    #     torch.save(self.actor_target.state_dict(), directory+'actor_target.pth')
    #     torch.save(self.critic_1.state_dict(), directory+'critic_1.pth')
    #     torch.save(self.critic_1_target.state_dict(), directory+'critic_1_target.pth')
    #     torch.save(self.critic_2.state_dict(), directory+'critic_2.pth')
    #     torch.save(self.critic_2_target.state_dict(), directory+'critic_2_target.pth')
    #     print("====================================")
    #     print("Model has been saved...")
    #     print("====================================")

    # def load(self):
    #     self.actor.load_state_dict(torch.load(directory + 'actor.pth'))
    #     self.actor_target.load_state_dict(torch.load(directory + 'actor_target.pth'))
    #     self.critic_1.load_state_dict(torch.load(directory + 'critic_1.pth'))
    #     self.critic_1_target.load_state_dict(torch.load(directory + 'critic_1_target.pth'))
    #     self.critic_2.load_state_dict(torch.load(directory + 'critic_2.pth'))
    #     self.critic_2_target.load_state_dict(torch.load(directory + 'critic_2_target.pth'))
    #     print("====================================")
    #     print("model has been loaded...")
    #     print("====================================")


