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
from DQNs import *
from utils import *
from EnvManagers import BreakoutEnvManager
import pickle

class Agent():
    def __init__(self, strategy, num_actions, device):
        self.current_step = 0
        self.strategy = strategy
        self.num_actions = num_actions
        self.device = device

    def select_action(self, state, policy_net):
        rate = self.strategy.get_exploration_rate(self.current_step)
        self.current_step += 1
        # print("eps = ",rate)
        if rate > random.random():
            action = random.randrange(self.num_actions)
            return torch.tensor([action]).to(self.device)  # explore
        else:
            with torch.no_grad():
                return policy_net(state).argmax(dim=1).to(self.device)  # exploit
# Configuration:
CHECK_POINT_PATH = "./checkpoints/"
FIGURES_PATH = "./figures/"
GAME_NAME = "Breakout/"
DATE_FORMAT = "%m-%d-%Y-%H-%M-%S"
UPDATE_PER_CHECKPOINT = 100000
# generate heldout set
HELD_OUT_SET = []
heldoutset_counter = 0
minibatch_updates_counter = 1

# Hyperparameters
batch_size = 32
gamma = 0.99
eps_start = 1
eps_end = 0.1
eps_startpoint = 50000
eps_kneepoint = 1000000 #BZX: the number of action taken by agent

target_update = 10000 # per minibatch_updates_counter
memory_size = 1000000
lr = 0.00001
num_episodes = 100000
replay_start_size = 50000
update_freq = 4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
em = BreakoutEnvManager(device)
strategy = EpsilonGreedyStrategyLinear(eps_start, eps_end, eps_startpoint, eps_kneepoint)
agent = Agent(strategy, em.num_actions_available(), device)
# memory = ReplayMemory(memory_size)
memory = ReplayMemory_economy(memory_size)

""" BZX:
cfgs: the configuration of CNN architecture used in feature extraction.
Option1 "standard": original setting in paper(2013,Mnih et al).
Option2 TODO.. [TRY]
"""
# availible models: DQN_CNN_2013,DQN_CNN_2015, Dueling_DQN_2016_Modified
policy_net = Dueling_DQN_2016_Modified(num_classes=em.num_actions_available(),init_weights=True).to(device)
target_net = Dueling_DQN_2016_Modified(num_classes=em.num_actions_available(),init_weights=True).to(device)


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

# try:
for episode in range(num_episodes):
    em.reset()
    state = em.get_state() # initialize sate
    tol_reward = 0
    # visualize_state(state)
    while(1):
        # em.env.render() # BZX: will this slow down the speed?
        action = agent.select_action(state, policy_net)
        #print("action = ", action.cpu().item())
        reward = em.take_action(action)
        # print("reward = ", reward.cpu().item())
        tol_reward += reward
        next_state = em.get_state()
        if random.random()<0.001:
            HELD_OUT_SET.append((next_state * 255).type(torch.uint8)) #store on GPU
            if len(HELD_OUT_SET) == 5000:
                heldoutset_file = open('heldoutset-{}'.format(heldoutset_counter), 'wb')
                pickle.dump(HELD_OUT_SET, heldoutset_file)
                heldoutset_file.close()
                HELD_OUT_SET=[]
                heldoutset_counter += 1
        # visualize_state(state)
        memory.push(Experience(state[0,-1,:,:].clone(), action, "", reward))

        state = next_state

        if (agent.current_step % update_freq == 0) and memory.can_provide_sample(batch_size,replay_start_size):
            experiences = memory.sample(batch_size)
            states, actions, rewards, next_states = extract_tensors(experiences)
            current_q_values = QValues.get_current(policy_net, states, actions) # checked
            # next_q_values = QValues.DQN_get_next(target_net, next_states)
            next_q_values = QValues.DDQN_get_next(policy_net,target_net, next_states)
            target_q_values = (next_q_values * gamma) + rewards

            loss = criterion(current_q_values, target_q_values.unsqueeze(1))
            # loss = F.mse_loss(current_q_values, target_q_values.unsqueeze(1)) #BZX: huber loss is better? [TRY]
            # print("loss=", loss.cpu().item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            minibatch_updates_counter += 1

            if minibatch_updates_counter % target_update == 0:
                target_net.load_state_dict(policy_net.state_dict())
                print("----")
                print("len of reply memory:", len(memory.memory))
                print("minibatch_updates_counter = ", minibatch_updates_counter)
                print("current_step of agent = ", agent.current_step)
                print("exploration rate = ", strategy.get_exploration_rate(agent.current_step))

            # BZX: checkpoint model
            if minibatch_updates_counter % UPDATE_PER_CHECKPOINT == 0:
                path = CHECK_POINT_PATH + GAME_NAME
                if not os.path.exists(path):
                    os.makedirs(path)
                torch.save(policy_net.state_dict(),
                           path + "Iterations:{}-Reward:{:.2f}-Time:".format(minibatch_updates_counter, running_reward) + \
                           datetime.datetime.now().strftime(DATE_FORMAT) + ".pth")
                plt.savefig(FIGURES_PATH + "Iterations:{}-Time:".format(minibatch_updates_counter) + datetime.datetime.now().strftime(
                    DATE_FORMAT) + ".jpg")

        if em.done:
            rewards_hist.append(tol_reward)
            running_reward = plot(rewards_hist, 100)
            break

# except:
#     heldoutset_file = open('heldoutset-exception', 'wb')
#     pickle.dump(HELD_OUT_SET, heldoutset_file)
#     heldoutset_file.close()

em.close()

# write heldoutset

