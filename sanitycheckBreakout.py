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

"""
Test how env works in a random sense.
"""
env = gym.make('BreakoutDeterministic-v4')
rewards_hist = []
num_episodes = 10000
for episode in range(num_episodes):
    env.reset()
    tol_reward = 0
    while (1):
        # time.sleep(0.3)
        action = env.action_space.sample()
        _, reward, done, _ = env.step(action)
        tol_reward += reward
        env.render()
        print(reward)

        if done:
            rewards_hist.append(tol_reward)
            plot(rewards_hist, 100)
            break
# avg reword should be 1
