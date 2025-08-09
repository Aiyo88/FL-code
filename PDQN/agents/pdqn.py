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
from PDQN.agents.agent import Agent
from PDQN.agents.memory.memory import Memory


class OrnsteinUhlenbeckActionNoise(object):
    def __init__(self, action_dim, mu=0.0, theta=0.15, sigma=0.2, random_machine=None):
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

def soft_update_target_network(source_network, target_network, tau):
    for target_param, param in zip(target_network.parameters(), source_network.parameters()):
        target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

def hard_update_target_network(source_network, target_network):
    for target_param, param in zip(target_network.parameters(), source_network.parameters()):
        target_param.data.copy_(param.data)


class ParamActor(nn.Module):
    """
    参数化Actor：输出一个高维的上下文向量，辅助QActor进行决策。
    """
    def __init__(self, observation_space, context_vector_size, hidden_dim=128, num_heads=4, **kwargs):
        super(ParamActor, self).__init__()
        node_feature_dim = observation_space['x'].shape[1]
        
        self.gnn = GNNModel(
            input_dim=node_feature_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            num_heads=num_heads
        )
        self.fc = nn.Linear(hidden_dim, context_vector_size)
        
    def forward(self, state):
        graph_embedding = self.gnn(state)
        return self.fc(graph_embedding)


class QActor(nn.Module):
    """
    Q-Actor网络：接收GNN状态嵌入和ParamActor的上下文向量，输出所有离散动作的Q值。
    """
    def __init__(self, observation_space, action_size, context_vector_size, hidden_dim=128, num_heads=4, **kwargs):
        super(QActor, self).__init__()
        node_feature_dim = observation_space['x'].shape[1]
        
        self.gnn = GNNModel(
            input_dim=node_feature_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            num_heads=num_heads
        )
        
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim + context_vector_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_size)
        )

    def forward(self, state, context_vector):
        graph_embedding = self.gnn(state)
        combined = torch.cat([graph_embedding, context_vector], dim=1)
        return self.mlp(combined)

class PDQNAgent(Agent):
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
                 loss_func=F.mse_loss,
                 clip_grad=1.0,
                 device="cuda" if torch.cuda.is_available() else "cpu",
                 seed=None):
        super(PDQNAgent, self).__init__(observation_space, action_space)
        self.device = torch.device(device)
        
        if not isinstance(action_space, spaces.Discrete):
            raise TypeError("Action space must be of type gym.spaces.Discrete")
        self.num_actions = action_space.n
        self.action_space = action_space

        self.epsilon = epsilon_initial
        self.epsilon_initial = epsilon_initial
        self.epsilon_final = epsilon_final
        self.epsilon_steps = epsilon_steps
        
        self.batch_size = batch_size
        self.gamma = gamma
        self.initial_memory_threshold = initial_memory_threshold
        self.learning_rate_actor = learning_rate_actor
        self.learning_rate_actor_param = learning_rate_actor_param
        self.tau_actor = tau_actor
        self.tau_actor_param = tau_actor_param
        self._step = 0
        self._episode = 0
        self.updates = 0
        self.clip_grad = clip_grad

        self.np_random = np.random.RandomState()
        self._seed(seed)

        self.replay_memory = Memory(replay_memory_size)
        
        self.q_actor1 = actor_class(self.observation_space, self.num_actions, **actor_kwargs).to(device)
        self.q_actor_target1 = actor_class(self.observation_space, self.num_actions, **actor_kwargs).to(device)
        hard_update_target_network(self.q_actor1, self.q_actor_target1)
        self.q_actor_target1.eval()

        self.q_actor2 = actor_class(self.observation_space, self.num_actions, **actor_kwargs).to(device)
        self.q_actor_target2 = actor_class(self.observation_space, self.num_actions, **actor_kwargs).to(device)
        hard_update_target_network(self.q_actor2, self.q_actor_target2)
        self.q_actor_target2.eval()

        self.actor_param = actor_param_class(self.observation_space, **actor_param_kwargs).to(device)
        self.actor_param_target = actor_param_class(self.observation_space, **actor_param_kwargs).to(device)
        hard_update_target_network(self.actor_param, self.actor_param_target)
        self.actor_param_target.eval()

        self.loss_func = loss_func

        self.q_actor1_optimiser = optim.Adam(self.q_actor1.parameters(), lr=self.learning_rate_actor)
        self.q_actor2_optimiser = optim.Adam(self.q_actor2.parameters(), lr=self.learning_rate_actor)
        self.actor_param_optimiser = optim.Adam(self.actor_param.parameters(), lr=self.learning_rate_actor_param)

    def _seed(self, seed=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            self.np_random.seed(seed)
            torch.manual_seed(seed)
            if self.device == torch.device("cuda"):
                torch.cuda.manual_seed(seed)
    
    def end_episode(self):
        self._episode += 1
        ep = self._episode
        if ep < self.epsilon_steps:
            self.epsilon = self.epsilon_initial - (self.epsilon_initial - self.epsilon_final) * (ep / self.epsilon_steps)
        else:
            self.epsilon = self.epsilon_final

    def select_action(self, state):
        with torch.no_grad():
            state.to(self.device)
            if self.np_random.uniform() < self.epsilon:
                return self.action_space.sample()
            else:
                context_vector = self.actor_param(state)
                q_values = self.q_actor1(state, context_vector)
                return torch.argmax(q_values).item()

    def store(self, state, action, reward, next_state, terminal):
        # 此方法由 learn_from_episode_data 内部调用，以填充修正后的经验
        self._step += 1
        self.replay_memory.append(state, np.array([action]), reward, next_state, np.array([terminal], dtype=np.uint8))
        
    def learn_from_episode_data(self, episode_data, num_updates=100):
        """
        [Off-Policy 回合制学习] 在Episode结束后，使用修正后的奖励填充经验池，然后批量学习。
        """
        # 1. 计算未来累积折扣回报并标准化
        rewards = episode_data['rewards']
        discounted_rewards = np.zeros_like(rewards, dtype=np.float32)
        running_add = 0
        for t in reversed(range(0, len(rewards))):
            running_add = running_add * self.gamma + rewards[t]
            discounted_rewards[t] = running_add
            
        mean = np.mean(discounted_rewards)
        std = np.std(discounted_rewards) + 1e-8
        normalized_rewards = (discounted_rewards - mean) / std

        # 2. 将带有修正后奖励的经验，全部存入回放池
        for i in range(len(rewards)):
            self.store(
                episode_data['states'][i],
                episode_data['actions'][i],
                normalized_rewards[i],
                episode_data['next_states'][i],
                episode_data['terminals'][i]
            )

        # 3. 只要经验池大小足够，就进行批量学习
        if self.replay_memory.__len__() >= self.initial_memory_threshold:
            print(f"--- Episode结束，经验池大小 ({self.replay_memory.__len__()})，开始进行 {num_updates} 次批量学习 ---")
            total_q_loss = 0.0
            total_param_loss = 0.0
            
            for i in range(num_updates):
                q_loss, param_loss = self._optimize_td_loss()
                if q_loss is not None:
                    total_q_loss += q_loss
                    total_param_loss += param_loss
                if (i + 1) % 10 == 0:
                     print(f"  Update {(i+1)}/{num_updates}: Avg_Critic_Loss={(total_q_loss/(i+1)):.6f}, Avg_Actor_Loss={(total_param_loss/(i+1)):.6f}")
            print(f"--- 批量学习完成 ---")
        else:
            print(f"--- Episode结束，正在填充经验池 ({self.replay_memory.__len__()}/{self.initial_memory_threshold}) ---")

    def _optimize_td_loss(self):
        if self.replay_memory.__len__() < self.batch_size:
            return None, None
            
        states, actions, rewards, next_states, terminals = self.replay_memory.sample(self.batch_size)

        states = [s.to(self.device) for s in states]
        next_states = [ns.to(self.device) for ns in next_states]

        states = Batch.from_data_list(states).to(self.device)
        actions = torch.from_numpy(actions).long().to(self.device)
        rewards = torch.from_numpy(rewards).float().to(self.device).squeeze()
        next_states = Batch.from_data_list(next_states).to(self.device)
        terminals = torch.from_numpy(terminals).to(self.device).squeeze()

        with torch.no_grad():
            next_context_vectors = self.actor_param_target(next_states)
            pred_Q_a1 = self.q_actor_target1(next_states, next_context_vectors)
            pred_Q_a2 = self.q_actor_target2(next_states, next_context_vectors)
            Qprime, _ = torch.max(torch.min(pred_Q_a1, pred_Q_a2), 1)
            target = rewards + (1 - terminals) * self.gamma * Qprime

        context_vectors = self.actor_param(states)
        q_values1 = self.q_actor1(states, context_vectors)
        q_values2 = self.q_actor2(states, context_vectors)
        
        y_predicted1 = q_values1.gather(1, actions).squeeze()
        y_predicted2 = q_values2.gather(1, actions).squeeze()

        loss_Q1 = self.loss_func(y_predicted1, target)
        loss_Q2 = self.loss_func(y_predicted2, target)
        loss_Q = loss_Q1 + loss_Q2

        self.q_actor1_optimiser.zero_grad()
        self.q_actor2_optimiser.zero_grad()
        loss_Q.backward()

        if self.clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(self.q_actor1.parameters(), self.clip_grad)
            torch.nn.utils.clip_grad_norm_(self.q_actor2.parameters(), self.clip_grad)
        self.q_actor1_optimiser.step()
        self.q_actor2_optimiser.step()

        # Optimize Actor-Param
        for p in self.q_actor1.parameters():
            p.requires_grad = False
        
        context_vectors_pred = self.actor_param(states)
        q_values_pred = self.q_actor1(states, context_vectors_pred)
        actor_param_loss = -torch.mean(q_values_pred)

        self.actor_param_optimiser.zero_grad()
        actor_param_loss.backward()
        
        if self.clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(self.actor_param.parameters(), self.clip_grad)
        
        self.actor_param_optimiser.step()

        for p in self.q_actor1.parameters():
            p.requires_grad = True

        soft_update_target_network(self.q_actor1, self.q_actor_target1, self.tau_actor)
        soft_update_target_network(self.q_actor2, self.q_actor_target2, self.tau_actor)
        soft_update_target_network(self.actor_param, self.actor_param_target, self.tau_actor_param)
        
        return loss_Q.item(), actor_param_loss.item()
    
    def save_models(self, prefix):
        torch.save(self.q_actor1.state_dict(), prefix + '_q_actor1.pt')
        torch.save(self.q_actor2.state_dict(), prefix + '_q_actor2.pt')
        torch.save(self.actor_param.state_dict(), prefix + '_actor_param.pt')
        print('Models saved successfully')

    def load_models(self, prefix):
        self.q_actor1.load_state_dict(torch.load(prefix + '_q_actor1.pt', map_location='cpu'))
        self.q_actor2.load_state_dict(torch.load(prefix + '_q_actor2.pt', map_location='cpu'))
        self.actor_param.load_state_dict(torch.load(prefix + '_actor_param.pt', map_location='cpu'))
        print('Models loaded successfully')

