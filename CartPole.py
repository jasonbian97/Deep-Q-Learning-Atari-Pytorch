import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from itertools import count
import torch.nn.functional as F
from PIL import Image
import torch
import torch.optim as optim
import torchvision.transforms as T
# customized import
from DQNs import DQN
from utils import *
from EnvManagers import CartPoleEnvManager


class Agent():
    def __init__(self, strategy, num_actions, device):
        self.current_step = 0
        self.strategy = strategy
        self.num_actions = num_actions
        self.device = device

    def select_action(self, state, policy_net):
        rate = self.strategy.get_exploration_rate(self.current_step)
        self.current_step += 1

        if rate > random.random():
            action = random.randrange(self.num_actions)
            return torch.tensor([action]).to(self.device)  # explore
        else:
            with torch.no_grad():
                return policy_net(state).argmax(dim=1).to(self.device)  # exploit

# Hyperparameters
batch_size = 256
gamma = 0.999
eps_start = 1
eps_end = 0.05
eps_decay = 0.001
target_update = 10
memory_size = 100000
lr = 0.001
num_episodes = 10000
eps_kneepoint = 500000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
em = CartPoleEnvManager(device)
strategy = EpsilonGreedyStrategyLinear(eps_start, eps_end, eps_kneepoint)
agent = Agent(strategy, em.num_actions_available(), device)
memory = ReplayMemory(memory_size)

policy_net = DQN(em.get_screen_height(), em.get_screen_width()).to(device)
target_net = DQN(em.get_screen_height(), em.get_screen_width()).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval() # this network will only be used for inference.
optimizer = optim.Adam(params=policy_net.parameters(), lr=lr)

episode_durations = []
plt.figure()
for episode in range(num_episodes):
    em.reset()
    state = em.get_state()

    for timestep in count():
        action = agent.select_action(state, policy_net)
        reward = em.take_action(action)
        next_state = em.get_state()
        memory.push(Experience(state, action, next_state, reward))
        state = next_state

        if memory.can_provide_sample(batch_size):
            experiences = memory.sample(batch_size)
            states, actions, rewards, next_states = extract_tensors(experiences)

            current_q_values = QValues.get_current(policy_net, states, actions)
            next_q_values = QValues.get_next(target_net, next_states)
            target_q_values = (next_q_values * gamma) + rewards

            loss = F.mse_loss(current_q_values, target_q_values.unsqueeze(1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if em.done:
            episode_durations.append(timestep)
            plot(episode_durations, 100)
            break
    if episode % target_update == 0:
        target_net.load_state_dict(policy_net.state_dict())

em.close()

