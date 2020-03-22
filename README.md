# RL-Atari-gym
Reinforcement Learning on Atari Games and Control

Entrance of program: 
1. Breakout.py 
2. CartPole.py


TODO list:
-[ ] Implement Priority Queue and compare the performance.

-[ ] Evaluation script. Load model and take greedy strategy to interact with environment.
Test a few epochs and give average performance. Write the best one to .gif file for presentation.

-[ ] Write validation script on heldout sets. Load models and heldout sets, track average max Q value on heldout sets.
(NOTE: load and test models in time sequence indicated by the name of model file.)

-[ ] Test two more Atari games. Give average performance(reward) and write .gif file. Store other figures & model for
writing final report.

-[ ] Rewrite image preprocessing class to tackle with more general game input.(e.g. remove crop step)

-[ ] Implement policy gradient for Atari games. [TBD]


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