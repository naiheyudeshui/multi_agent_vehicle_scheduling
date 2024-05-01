'''
Actor-Critic, actually Advantage Actor-Critic (A2C).

Policy loss in Vanilla Actor-Critic is: -log_prob(a)*Q(s,a) ,
Policy loss in A2C is: -log_prob(a)*[Q(s,a)-V(s)], while Adv(s,a)=Q(s,a)-V(s)=r+gamma*V(s')-V(s)=TD_error ,
and in this implementation we provide another approach that the V(s') is replaced by R(s'),
which is derived from the rewards in the episode for on-policy update without evaluation.

Discrete and Non-deterministic
'''

import math
import random
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from torch.distributions import Categorical
from collections import namedtuple

from IPython.display import clear_output
import matplotlib.pyplot as plt


# use_cuda = torch.cuda.is_available()
# device   = torch.device("cuda" if use_cuda else "cpu")
# print(device)


class ReplayBuffer:
    """A simple numpy replay buffer."""

    def __init__(self, obs_dim: int, size: int, batch_size: int = 32):
        self.obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.next_obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size], dtype=np.float32)
        self.rews_buf = np.zeros([size], dtype=np.float32)
        self.Q_buf = np.zeros([size], dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.max_size, self.batch_size = size, batch_size
        self.ptr, self.size, = 0, 0

    def store(
        self,
        obs: np.ndarray,
        act: np.ndarray,
        rew: float,
        Q: float,
        next_obs: np.ndarray,
        done: bool,
    ):
        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.Q_buf[self.ptr] = Q
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self) -> Dict[str, np.ndarray]:
        idxs = np.random.choice(self.size, size=self.batch_size, replace=False)
        return dict(obs=self.obs_buf[idxs],
                    next_obs=self.next_obs_buf[idxs],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    Q=self.Q_buf[idxs],
                    done=self.done_buf[idxs])

    def __len__(self) -> int:
        return self.size


class ActorNetwork(nn.Module):
    def __init__(self, in_dim, hidden_dim, action_dim=21, init_w=3e-3):
        super(ActorNetwork, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, action_dim)
        )
        torch.nn.init.uniform_(self.layers[-1].weight, -init_w, init_w)

    def forward(self, state):
        x = self.layers(state)
        return F.softmax(x, dim=-1)


class CriticNetwork(nn.Module):
    def __init__(self, in_dim, hidden_dim, action_dim=21, init_w=3e-3):
        super(CriticNetwork, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, state):
        return self.layers(state)


class Actor_agent:
    def __init__(self,
                 Critic_agent,
                 memory_size: int,
                 batch_size: int,
                 gamma: float = 0,
                 obs_dim: int = 180,
                 hidden_dim: int = 1024,
                 action_dim: int = 21,
                 lr: float = 1e-2
                 ):
        self.Critic_agent = Critic_agent
        self.memory = ReplayBuffer(obs_dim, memory_size, batch_size)
        self.gamma = gamma
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        print(self.device)

        self.ActorNetwork = ActorNetwork(obs_dim, hidden_dim, action_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.ActorNetwork.parameters(), lr=lr)

    def select_action(self, state):
        '''
        Select an action without gradients flowing through it, for interaction with the environment.
        '''
        state = torch.FloatTensor(state).to(self.device)
        probs = self.ActorNetwork.forward(state)
        m = Categorical(probs)
        action = m.sample()
        return action.detach().cpu().numpy()

    def evaluate_action(self, state, recorded_action):
        '''
        Evaluate action within the GPU computation graph, allowing gradients to flow through it.
        '''
        # state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        # recorded_action = torch.LongTensor(recorded_action).to(self.device)
        probs = self.ActorNetwork.forward(state)
        m = Categorical(probs)

        # Evaluate log_prob for the recorded action
        log_prob = m.log_prob(recorded_action)
        entropy = m.entropy().mean()

        return log_prob, entropy.detach().cpu().numpy()

    def _compute_loss(self, samples: Dict[str, np.ndarray]) -> torch.Tensor:
        """Return loss."""
        device = self.device  # for shortening the following lines
        state = torch.FloatTensor(samples["obs"]).to(device)
        next_state = torch.FloatTensor(samples["next_obs"]).to(device)
        action = torch.LongTensor(samples["acts"].reshape(-1, 1)).to(device)
        reward = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(device)
        Q = torch.FloatTensor(samples["Q"]).to(device)
        done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(device)

        state_value = self.Critic_agent.CriticNetwork(state).detach()
        log_prob, entropy = self.evaluate_action(state, action)
        next_state_value = self.Critic_agent.CriticNetwork(next_state).detach()
        policy_losses = -log_prob * Q
        # policy_loss = torch.mean(policy_losses)

        return policy_losses

    def update_model(self) -> torch.Tensor:
        """先计算出损失，再由损失计算梯度，最后由梯度进行更新，任何含计算图的，理论上都可以对参数求导也就是梯度，而梯度一定可以优化
        所以梯度和优化器往往在一块"""

        losses = []  # 用于存储每个样本的损失值

        # 遍历经验池中的所有样本
        for i in range(self.memory.ptr):
            # 从经验池中获取单个样本
            sample = {
                'obs': self.memory.obs_buf[i:i + 1],
                'next_obs': self.memory.next_obs_buf[i:i + 1],
                'acts': self.memory.acts_buf[i:i + 1],
                'rews': self.memory.rews_buf[i:i + 1],
                'Q': self.memory.Q_buf[i:i + 1],
                'done': self.memory.done_buf[i:i + 1]
            }

            # 计算单个样本的损失
            loss = self._compute_loss(sample)

            # 梯度下降和参数更新
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # 存储损失值
            losses.append(loss.item())

            # 清空经验池
        self.memory.ptr = 0
        self.memory.size = 0

        return sum(losses) / len(losses)


class Critic_agent:
    def __init__(self,
                 memory_size: int,
                 batch_size: int,
                 gamma: float = 1,
                 obs_dim = 180,
                 hidden_dim = 1024,
                 action_dim = 21,
                 lr = 5e-3
    ):
        self.memory = ReplayBuffer(obs_dim, memory_size, batch_size)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.CriticNetwork = CriticNetwork(obs_dim, hidden_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.CriticNetwork.parameters(), lr=lr)
        self.gamma = gamma
        self.transition = list()

    def _compute_loss(self, samples: Dict[str, np.ndarray]) -> torch.Tensor:
        """Return loss."""
        device = self.device  # for shortening the following lines
        state = torch.FloatTensor(samples["obs"]).to(device)
        next_state = torch.FloatTensor(samples["next_obs"]).to(device)
        action = torch.LongTensor(samples["acts"].reshape(-1, 1)).to(device)
        reward = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(device)
        Q = torch.FloatTensor(samples["Q"].reshape(-1, 1)).to(device)
        done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(device)
        curr_q_value = self.CriticNetwork(state).gather(1, action)
        next_q_value = self.CriticNetwork(
            next_state
        ).detach()
        mask = 1 - done
        target = (Q + self.gamma * next_q_value * mask).to(self.device)
        # calculate dqn loss
        loss = F.smooth_l1_loss(curr_q_value, target)
        return loss

    def update_model(self) -> torch.Tensor:
        """先计算出损失，再由损失计算梯度，最后由梯度进行更新，任何含计算图的，理论上都可以对参数求导也就是梯度，而梯度一定可以优化
        所以梯度和优化器往往在一块"""
        samples = self.memory.sample_batch()

        loss = self._compute_loss(samples)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def select_action(self, state):
        '''
        Select an action
        '''
        state = torch.FloatTensor(state).to(self.device)
        action = self.CriticNetwork.forward(state)
        action = action.argmax()
        return action.detach().cpu().numpy()




