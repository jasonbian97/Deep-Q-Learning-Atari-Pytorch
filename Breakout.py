import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import time
from itertools import count
import os
import sys
import torch.nn.functional as F
import datetime
from PIL import Image
import torch
import torch.optim as optim
import torchvision.transforms as T
# customized import
from DQNs import DQN_CNN1
from utils import *
from EnvManagers import BreakoutEnvManager

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
# Configuration:
CHECK_POINT_PATH = "./checkpoints/"
GAME_NAME = "Breakout/"
DATE_FORMAT = "%m-%d-%Y-%H-%M-%S"
EPISODES_PER_CHECKPOINT = 5000


# Hyperparameters
batch_size = 32
gamma = 0.99
eps_start = 1
eps_end = 0.1
eps_decay = 0.001
target_update = 4
memory_size = 60000 #BZX: apporximately would take 6.7GB on your GPU
lr = 0.001
num_episodes = 100000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
em = BreakoutEnvManager(device)
strategy = EpsilonGreedyStrategy(eps_start, eps_end, eps_decay)
agent = Agent(strategy, em.num_actions_available(), device)
memory = ReplayMemory(memory_size)

""" BZX:
cfgs: the configuration of CNN architecture used in feature extraction.
Option1 "standard": original setting in paper(2013,Mnih et al).
Option2 TODO.. [TRY]
"""
cfgs = {
        'standard': [16, 'M', 16, 'M', 16, 'M', 32, 'M', 32, 'M', 64]
    }
policy_net = DQN_CNN1(cfgs['standard'],num_classes=em.num_actions_available(),init_weights=True).to(device)
target_net = DQN_CNN1(cfgs['standard'],num_classes=em.num_actions_available(),init_weights=True).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval() # BZX: this network will only be used for inference.
optimizer = optim.Adam(params=policy_net.parameters(), lr=lr)
criterion = torch.nn.SmoothL1Loss()
# BZX: Sanity Check: print(policy_net)
# print("num_actions_available: ",em.num_actions_available())
# print("action_meanings:" ,em.env.get_action_meanings())

rewards_hist = []
running_reward = 0
plt.figure()

for episode in range(num_episodes):
    em.reset()
    state = em.get_state()
    tol_reward = 0

    while(1):
        # time.sleep(0.3)
        # em.env.render() # BZX: will this slow down the speed?
        action = agent.select_action(state, policy_net)
        reward = em.take_action(action)
        tol_reward += reward
        next_state = em.get_state()
        memory.push(Experience(state, action, next_state, reward))
        state = next_state

        if memory.can_provide_sample(batch_size):
            experiences = memory.sample(batch_size)
            states, actions, rewards, next_states = extract_tensors(experiences)

            current_q_values = QValues.get_current(policy_net, states, actions)
            next_q_values = QValues.get_next(target_net, next_states)
            target_q_values = (next_q_values * gamma) + rewards
            # loss = criterion(current_q_values, target_q_values.unsqueeze(1))
            loss = F.mse_loss(current_q_values, target_q_values.unsqueeze(1)) #BZX: huber loss is better? [TRY]
            # print("loss=", loss.cpu().item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if em.done:
            rewards_hist.append(tol_reward)
            running_reward = plot(rewards_hist, 100)
            break
    if episode % target_update == 0:
        target_net.load_state_dict(policy_net.state_dict())
        print("len of reply memory:", len(memory.memory))

    # BZX: checkpoint model
    if episode % EPISODES_PER_CHECKPOINT == 0:
        path = CHECK_POINT_PATH+GAME_NAME
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(policy_net.state_dict(), path + "Episodes:{}-Reward:{}-Time:".format(episode,running_reward) + \
                   datetime.datetime.now().strftime(DATE_FORMAT) +".pth")

em.close()


