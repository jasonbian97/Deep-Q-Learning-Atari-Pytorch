# RL-Atari-gym
Reinforcement Learning on Atari Games and Control

Entrance of program: 
1. Breakout.py 
2. CartPole.py

File Instruction:

'EnvManagers.py' includes the different environment classes for different games. They wrapped the gym.env and its interface.

'DQNs.py' includes different Deep Learning architectures for feature extraction and regression.

'utils.py' includes tool functions and classes. To be specific, it includes:
- Experience (namedtuple)
- ReplayMemory (class)
- EpsilonGreedyStrategy (class)
- plot (func)
- extract_tensors (func)
- QValues (class)