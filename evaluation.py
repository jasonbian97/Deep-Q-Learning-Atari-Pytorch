import datetime
import torch.optim as optim
import time
# customized import
from DQNs import *
from utils import *
from EnvManagers import BreakoutEnvManager
from Agent import *


"""
test a few episodes and average the reward.
OOP?
set policy_net.eval()
"""
model_fpath = "./checkpoints/Breakout/Iterations:1400000-Reward:155.95-Time:03-27-2020-08-10-54.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# load model from file
em = BreakoutEnvManager(device)
policy_net = DQN_CNN_2015(num_classes=em.num_actions_available(),init_weights=False).to(device)
policy_net.load_state_dict(torch.load(model_fpath))
policy_net.eval() # this network will only be used for inference.
# setup greedy strategy and Agent class
strategy = FullGreedyStrategy(0.05)
agent = Agent(strategy, em.num_actions_available(), device)

# Auxilary variables
config_dict = {}
config_dict["IS_RENDER_GAME_PROCESS"] = True
config_dict["IS_VISUALIZE_STATE"] = True
config_dict["EVAL_EPISODE"] = 5
config_dict["FIR_SAVE_PATH"] = "./GIF_Reuslts/"
tracker_dict = {}
tracker_dict["eval_reward_list"] = []
best_framegits_for_gif = None
best_reward =  0
for episode in range(config_dict["EVAL_EPISODE"]):
    em.reset()
    state = em.get_state() # initialize sate
    reward_per_episode = 0
    frames_for_gif = []
    while(1):
        if config_dict["IS_RENDER_GAME_PROCESS"]: em.env.render() #render in 'human  mode. BZX: will this slow down the speed?
        if config_dict["IS_VISUALIZE_STATE"]: visualize_state(state)x
        frame = em.render('rgb_array')
        frames_for_gif.append(frame)
        # Given s, select a by strategy
        action = 1 if (em.done or em.is_lives_loss) else agent.select_action(state, policy_net)
        # take action
        reward = em.take_action(action)
        # collect unclipped reward from env along the action
        reward_per_episode += reward
        # after took a, get s'
        next_state = em.get_state()
        # update current state
        state = next_state

        if em.done:
            tracker_dict["eval_reward_list"].append(reward_per_episode)
            break
    #write git
    if reward_per_episode > best_reward:
        best_reward = reward_per_episode
        #update best frames list
        best_frames_for_gif = frames_for_gif.copy()

print( tracker_dict["eval_reward_list"])

generate_gif(config_dict["FIR_SAVE_PATH"],best_frames_for_gif,best_reward)






