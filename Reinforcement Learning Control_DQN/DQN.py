# -*-coding:utf-8-*-
# @Time  : 2022/3/2 11:04
# @Author: hsy
# @File  : ddqn.py
"""
Deep Q Learning (DQN) with Fix-Q-target, Reinforcement Learning.
只适合离散动作的DQN算法，已实现Fix-Q-target，support gqu.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque, namedtuple
import random
import torch
import numpy as np


class ReplayBuffer:
    """
    Experience Replay Buffer，fixed-size buffer to store experience tuples.
    """
    def __init__(self, buffer_size, batch_size, device):
        """
        :param buffer_size: maximum size of buffer
        :param batch_size: size of each training batch
        :param device: cpu or gpu
        """
        self.device = device
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state"])

    def add(self, state, action, reward, next_state):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        if self.batch_size > len(self.memory):
            return self.return_all_samples()
        else:

            experiences = random.sample(self.memory, k=self.batch_size)

            states = torch.from_numpy(np.stack([e.state for e in experiences if e is not None])).float().to(self.device)
            actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).to(self.device)
            rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
            next_states = torch.from_numpy(np.stack([e.next_state for e in experiences if e is not None])).float().to(self.device)

            return states, actions, rewards, next_states

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

    def size(self):
        return len(self.memory)

    def return_all_samples(self):
        all_transitions = list(self.memory)
        state, action, reward, next_state = zip(*all_transitions)
        states = torch.from_numpy(np.array(state)).float().to(self.device)
        actions = torch.from_numpy(np.array(action)).unsqueeze(dim=1).to(self.device)
        rewards = torch.from_numpy(np.array(reward)).unsqueeze(dim=1).float().to(self.device)
        next_states = torch.from_numpy(np.array(next_state)).float().to(
            self.device)
        return states, actions, rewards, next_states

    def clear(self):
        self.memory.clear()


class DQNetwork(nn.Module):
	def __init__(self, state_dim, action_dim, hidden_dim_1, hidden_dim_2):
		super(DQNetwork, self).__init__()
		self.state_dim = state_dim
		self.action_dim = action_dim

		# full-connected layer
		self.fc1 = nn.Linear(self.state_dim, hidden_dim_1)
		self.fc2 = nn.Linear(hidden_dim_1, hidden_dim_2)
		self.fc3 = nn.Linear(hidden_dim_2, hidden_dim_2)
		self.output = nn.Linear(hidden_dim_2, action_dim)

	def forward(self, x):
		# use ReLU as activate function
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = F.relu(self.fc3(x))

		q_value = self.output(x)

		return q_value


class DQN(object):

	def __init__(self, state_dim, action_dim, device):
		super(DQN, self).__init__()
		self.state_dim = state_dim
		self.action_dim = action_dim
		self.lr = 1e-4
		self.epsilon = 1.0
		self.epsilon_decay = 0.995
		self.min_epsilon = 0.01
		self.gamma = 0.99
		self.buffer_size = int(1e6)
		self.tau = 0.01
		self.batch_size = 256
		self.hidden_dim_1 = 1024
		self.hidden_dim_2 = 1024
		self.device = device  # set device to cpu or gpu

		# Network
		self.eval_net = DQNetwork(state_dim, action_dim, self.hidden_dim_1, self.hidden_dim_2).to(device)
		self.target_net = DQNetwork(state_dim, action_dim, self.hidden_dim_1, self.hidden_dim_2).to(device)
		self.target_net.load_state_dict(self.eval_net.state_dict())

		self.optimizer = torch.optim.Adam(self.eval_net.parameters(), self.lr)  # Adam Optimizer
		self.loss_func = nn.MSELoss()

		# Experience Memory Replay
		self.buffer = ReplayBuffer(self.buffer_size, self.batch_size, device)

	def select_action(self, s):
		# epsilon greedy
		if np.random.random() < self.epsilon:
			action = np.random.randint(self.action_dim)
		else:
			s = torch.tensor(np.array([s]), dtype=torch.float).to(self.device)
			action_prob = self.eval_net(s)
			action = torch.argmax(action_prob).item()
		return action

	def decrement_epsilon(self):
		"""
		衰减贪心程度，前期多些探索，后期减小探索
		Decrements the epsilon after each step till it reaches minimum epsilon
		epsilon = epsilon - decrement (default is 0.99e-6)
		"""
		self.epsilon = self.epsilon - self.epsilon_decay if self.epsilon > self.min_epsilon \
			else self.min_epsilon

	def update(self):

		# Sample data
		states, actions, rewards, states_ = self.buffer.sample()
		
		actions = actions.squeeze(dim=-1)
		rewards = rewards.squeeze(dim=-1)

		batches = np.arange(self.batch_size)

		q_eval = self.eval_net(states)[batches, actions]

		q_next = self.target_net.forward(states_)

		max_q_next = q_next.max(dim=1)[0]

		q_target = rewards + self.gamma * max_q_next

		loss = self.loss_func(q_target, q_eval).to(self.device)

		loss_np = loss.clone().data.numpy() if self.device == torch.device('cpu') else loss.clone().data.cpu().numpy()

		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()

		self.decrement_epsilon()  # decay epsilon

		# soft update target net 学习一段时间更新目标网络
		# soft的意思是每次learn的时候更新部分参数
		for target_param, eval_param in zip(self.target_net.parameters(), self.eval_net.parameters()):
			target_param.data.copy_(self.tau * eval_param.data + (1.0 - self.tau) * target_param.data)

		return loss_np