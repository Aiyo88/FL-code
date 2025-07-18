import argparse
from itertools import count

import os, sys, random
import numpy as np
import math
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
# from tensorboardX import SummaryWriter
# self.writer.add_scalar


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
        
        # 确保参数有效
        assert state_dim is not None and state_dim > 0, f"无效的state_dim: {state_dim}"
        assert action_dim is not None and action_dim > 0, f"无效的action_dim: {action_dim}"
        
        # 转换为整数以确保兼容性
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)

        self.l1 = nn.Linear(self.state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, self.action_dim)

        # self.max_action = max_action

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        # print("action",action_dim)
        # print("xxxxx",x)
        # 将tanh[-1,1] 映射到 [-max_action,max_action]区间内
        # x = self.max_action * torch.tanh(self.l3(x))
        x = torch.sigmoid(self.l3(x))
        return x


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        
        # 确保参数有效
        assert state_dim is not None and state_dim > 0, f"无效的state_dim: {state_dim}"
        assert action_dim is not None and action_dim > 0, f"无效的action_dim: {action_dim}"
        
        # 转换为整数以确保兼容性
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)

        self.l1 = nn.Linear(self.state_dim + self.action_dim, 400)
        self.l2 = nn.Linear(400, 300)
        # 输出维度
        self.l3 = nn.Linear(300, self.action_dim)

    def forward(self, x, u):
        x = F.relu(self.l1(torch.cat([x, u], 1)))
        x = F.relu(self.l2(x))
        # x = self.l3(x)
        x = torch.sigmoid(self.l3(x))
        return x


class DDPG(object):
    def __init__(self, observation_space, action_space ,state_dim, action_dim, max_action, 
                                        batch_size,gamma,lr,lr_c , tau,capacity,exploration_noise,seed, update_iteration, num_actions_discrete, num_actions_continuous_1,
                                        device= "cpu",
                                        ):
        self.observation_space = observation_space
        self.action_space = action_space

        self.seed = seed
        self._seed(seed)

        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_c)
        self.replay_buffer = Replay_buffer(capacity,np.random)
        # self.writer = SummaryWriter(directory)

        self.num_critic_update_iteration = 0
        self.num_actor_update_iteration = 0
        self.num_training = 0


        ''' 新增 '''
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.capacity = capacity
        self.exploration_noise = exploration_noise

        self.update_iteration = update_iteration
        self.num_actions_discrete = num_actions_discrete
        self.num_actions_continuous_1 = num_actions_continuous_1
        self.device = device
        
        self.action_epsilon = 0

        '''
        
        '''

        
        self.num_actions_discrete = action_space.spaces[0].nvec.shape[0]  # N+1，训练决策(N)和聚合决策(1)
        self.action_parameter_sizes = np.array([ ( action_space.spaces[1].shape[0] * action_space.spaces[1].shape[1] ) ]) # 这里应该是连续的动作空间的数量，二维
        self.num_actions_continuous = action_space.spaces[1].shape[0]     # M，边缘资源分配
        self.action_parameter_size = self.num_actions_discrete + int(self.action_parameter_sizes.sum()) #  这里其实应该是所有的动作空间的参数
        self.num_actions = self.num_actions_discrete + self.num_actions_continuous  # + self.num_actions_continuous_2  # 离散的动作+连续的动作 总的数量



        # 设置动作空间的最大最小值和范围
        discrete_action_max = 1
        discrete_action_min = 0
        continuous_action_max = 1  # 资源分配最大值
        continuous_action_min = 0  # 资源分配最小值

        # 创建动作边界向量
        self.action_max = torch.from_numpy(np.concatenate([
            np.full(self.num_actions_discrete, discrete_action_max),
            np.full(self.num_actions_continuous, continuous_action_max)
        ])).float().to(device)

        self.action_min = torch.from_numpy(np.concatenate([
            np.full(self.num_actions_discrete, discrete_action_min),
            np.full(self.num_actions_continuous, continuous_action_min)
        ])).float().to(device)

        self.action_range = (self.action_max - self.action_min).detach()

        # 为numpy数组也创建相同的边界
        self.action_parameter_max_numpy = np.concatenate([
            np.full(self.num_actions_discrete, discrete_action_max),
            np.full(self.num_actions_continuous, continuous_action_max)
        ]).ravel()

        self.action_parameter_min_numpy = np.concatenate([
            np.full(self.num_actions_discrete, discrete_action_min),
            np.full(self.num_actions_continuous, continuous_action_min)
        ]).ravel()
        

    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
            action = self.actor(state)

            epsilon_end = 0.01
            epsilon_start = 1 
            epsilon_decay=3000 # 越大越探索吧  # 默认20000

            self.epsilon = epsilon_end + (epsilon_start - epsilon_end) * \
                        math.exp(-1. * self.action_epsilon / epsilon_decay)
            self.action_epsilon += 1


            rnd = self.np_random.uniform()
            if rnd < self.epsilon:
                action = torch.from_numpy(np.random.uniform(self.action_parameter_min_numpy,
                                                            self.action_parameter_max_numpy))
            else:
                action = self.action_min + action[0] * (self.action_max - self.action_min)

            action = action.cpu().data.numpy()
            

        return  action

    def update(self):

        for it in range(self.update_iteration):
            # Sample replay buffer
            x, y, u, r, d = self.replay_buffer.sample(self.batch_size)
            state = torch.FloatTensor(x).to(self.device)
            action = torch.FloatTensor(u).to(self.device)
            next_state = torch.FloatTensor(y).to(self.device)
            done = torch.FloatTensor(1-d).to(self.device)
            reward = torch.FloatTensor(r).to(self.device)

            # Compute the target Q value
            # target_Q = self.critic_target(next_state, self.actor_target(next_state))
            # target_Q = reward + (done * self.gamma * target_Q).detach()
            '''  safaa '''

            target_Q = self.critic_target(next_state, self.actor_target(next_state))
            target_Q = reward + (done * self.gamma * target_Q).detach()


            # Get current Q estimate
            current_Q = self.critic(state, action)

            # Compute critic loss
            critic_loss = F.mse_loss(current_Q, target_Q)
            # self.writer.add_scalar('Loss/critic_loss', critic_loss, global_step=self.num_critic_update_iteration)
            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Compute actor loss
            actor_loss = -self.critic(state, self.actor(state)).mean()
            # self.writer.add_scalar('Loss/actor_loss', actor_loss, global_step=self.num_actor_update_iteration)

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            self.num_actor_update_iteration += 1
            self.num_critic_update_iteration += 1

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
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            # if self.device == torch.device("cuda"):
            #     torch.cuda.manual_seed(seed)


    # def save(self):
    #     torch.save(self.actor.state_dict(), directory + 'actor.pth')
    #     torch.save(self.critic.state_dict(), directory + 'critic.pth')
    #     # print("====================================")
    #     # print("Model has been saved...")
    #     # print("====================================")

    # def load(self):
    #     self.actor.load_state_dict(torch.load(directory + 'actor.pth'))
    #     self.critic.load_state_dict(torch.load(directory + 'critic.pth'))
    #     print("====================================")
    #     print("model has been loaded...")
    #     print("====================================")
