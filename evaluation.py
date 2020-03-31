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

param_json_fname = "DDQN_params.json" #TODO
model_list_fname = "./eval_model_list_txt/ModelName:2015_CNN_DQN-GameName:Breakout-Time:03-30-2020-02-57-36.txt" #TODO

config_dict, hyperparams_dict, eval_dict = read_json(param_json_fname)

with open(model_list_fname) as f:
    model_list = f.readlines()
model_list = [x.strip() for x in model_list] # remove whitespace characters like `\n` at the end of each line
subfolder = model_list_fname.split("/")[-1][:-4]
# setup params

# Auxilary variables
tracker_dict = {}
tracker_dict["UPDATE_PER_CHECKPOINT"] = config_dict["UPDATE_PER_CHECKPOINT"]
tracker_dict["eval_reward_list"] = []
global_best_reward = -1
for model_fpath in model_list:
    print("testing:  ",model_fpath)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # load model from file
    em = BreakoutEnvManager(device) #TODO
    policy_net = DQN_CNN_2015(num_classes=em.num_actions_available(),init_weights=False).to(device) #TODO
    policy_net.load_state_dict(torch.load(model_fpath))
    policy_net.eval() # this network will only be used for inference.
    # setup greedy strategy and Agent class
    strategy = FullGreedyStrategy(0.01)
    agent = Agent(strategy, em.num_actions_available(), device)


    best_frames_for_gif = None
    best_reward =  -1

    reward_list_episodes = []
    for episode in range(eval_dict["EVAL_EPISODE"]):
        em.reset()
        state = em.get_state() # initialize sate
        reward_per_episode = 0
        frames_for_gif = []

        while(1):
            if eval_dict["IS_RENDER_GAME_PROCESS"]: em.env.render() #render in 'human  mode. BZX: will this slow down the speed?
            if eval_dict["IS_VISUALIZE_STATE"]: visualize_state(state)
            frame = em.render('rgb_array')
            frames_for_gif.append(frame)
            # Given s, select a by strategy
            if em.done or em.is_lives_loss or em.is_initial_action():
                action = torch.tensor([1])
            else:
                action = agent.select_action(state, policy_net)
            # take action
            reward = em.take_action(action)
            # collect unclipped reward from env along the action
            reward_per_episode += reward
            # after took a, get s'
            next_state = em.get_state()
            # update current state
            state = next_state

            if em.done:
                reward_list_episodes.append(reward_per_episode.cpu().item())
                break
        #write gif
        if reward_per_episode > best_reward:
            best_reward = reward_per_episode
            #update best frames list
            best_frames_for_gif = frames_for_gif.copy()

    tracker_dict["eval_reward_list"].append(np.median(reward_list_episodes))
    # print( tracker_dict["eval_reward_list"])
    # save results
    if  best_reward > 0.8 * np.median(tracker_dict["eval_reward_list"]):
        model_name = model_fpath.split("/")[-1][:-4]
        generate_gif(eval_dict["GIF_SAVE_PATH"] + subfolder + "/", model_name, best_frames_for_gif, global_best_reward.cpu().item())

    if not os.path.exists(config_dict["RESULT_PATH"]):
        os.makedirs(config_dict["RESULT_PATH"])

tracker_fname = subfolder + "-Eval.pkl"
with open(config_dict["RESULT_PATH"] + tracker_fname, 'wb') as f:
    pickle.dump(tracker_dict, f)






