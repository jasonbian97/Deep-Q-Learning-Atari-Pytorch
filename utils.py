import math
import random
import torch
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
import numpy as np
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display
else:
    matplotlib.use('TkAgg')

Experience = namedtuple(
    'Experience',
    ('state', 'action', 'next_state', 'reward')
)

Eco_Experience = namedtuple(
    'Eco_Experience',
    ('state', 'action', 'reward')
)

class ReplayMemory():
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.push_count = 0
        # self.dtype = torch.uint8

    def push(self, experience):
        if len(self.memory) < self.capacity:
            self.memory.append(experience)
        else:
            self.memory[self.push_count % self.capacity] = experience
        self.push_count += 1

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def can_provide_sample(self, batch_size):
        return len(self.memory) >= batch_size

class ReplayMemory_economy():
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.push_count = 0
        self.dtype = torch.uint8
    def push(self, experience):
        state = (experience.state * 255).type(self.dtype).cpu()
        # next_state = (experience.next_state * 255).type(self.dtype)
        new_experience = Eco_Experience(state,experience.action,experience.reward)

        if len(self.memory) < self.capacity:
            self.memory.append(new_experience)
        else:
            self.memory[self.push_count % self.capacity] = new_experience
        # print(id(experience))
        # print(id(self.memory[0]))
        self.push_count += 1

    def sample(self, batch_size):
        experience_index = np.random.randint(3, len(self.memory)-1, size = batch_size)
        # memory_arr = np.array(self.memory)
        experiences = []
        for index in experience_index:
            state = torch.stack(([self.memory[index+j].state for j in range(-3,1)])).unsqueeze(0)
            next_state = torch.stack(([self.memory[index+1+j].state for j in range(-3,1)])).unsqueeze(0)
            experiences.append(Experience(state.float().cuda()/255, self.memory[index].action, next_state.float().cuda()/255, self.memory[index].reward))
        # return random.sample(self.memory, batch_size)
        return experiences

    def can_provide_sample(self, batch_size, replay_start_size):
        return (len(self.memory) >= replay_start_size) and (len(self.memory) >= batch_size + 3)


class EpsilonGreedyStrategyExp():
    def __init__(self, start, end, decay):
        self.start = start
        self.end = end
        self.decay = decay

    def get_exploration_rate(self, current_step):
        return self.end + (self.start - self.end) * \
               math.exp(-1. * current_step * self.decay)

class EpsilonGreedyStrategyLinear():
    def __init__(self, start, end, startpoint = 50000, kneepoint=1000000):
        self.start = start
        self.end = end
        self.kneepoint = kneepoint
        self.startpoint = startpoint

    def get_exploration_rate(self, current_step):
        if current_step < self.startpoint:
            return 1.
        return self.end + \
               np.maximum(0, (1-self.end)-(1-self.end)/self.kneepoint * (current_step-self.startpoint))


def get_moving_average(period, values):
    values = torch.tensor(values, dtype=torch.float)
    if len(values) >= period:
        moving_avg = values.unfold(dimension=0, size=period, step=1) \
            .mean(dim=1).flatten(start_dim=0)
        moving_avg = torch.cat((torch.zeros(period-1), moving_avg))
        return moving_avg.numpy()
    else:
        moving_avg = torch.zeros(len(values))
        return moving_avg.numpy()

def plot(values, moving_avg_period):
    """
    test: plot(np.random.rand(300), 100)
    :param values: numpy 1D vector
    :param moving_avg_period:
    :return: None
    """
    # plt.figure()
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.plot(values)
    moving_avg = get_moving_average(moving_avg_period, values)
    plt.plot(moving_avg)
    print("Episode", len(values), "\n",moving_avg_period, "episode moving avg:", moving_avg[-1])
    plt.pause(0.0001)
    if is_ipython: display.clear_output(wait=True)
    return moving_avg[-1]

def extract_tensors(experiences):
    # Convert batch of Experiences to Experience of batches
    batch = Experience(*zip(*experiences))

    t1 = torch.cat(batch.state)
    t2 = torch.cat(batch.action)
    t3 = torch.cat(batch.reward)
    t4 = torch.cat(batch.next_state)

    return (t1,t2,t3,t4)

class QValues():
    """
    This is the class that we used to calculate the q-values for the current states using the policy_net,
     and the next states using the target_net
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def get_current(policy_net, states, actions):
        return policy_net(states).gather(dim=1, index=actions.unsqueeze(-1))

    @staticmethod
    def DQN_get_next(target_net, next_states, mode = "stacked"):
        if mode == "stacked":
            last_screens_of_state = next_states[:,-1,:,:] #(B,H,W)
            final_state_locations = last_screens_of_state.flatten(start_dim=1).max(dim=1)[0].eq(0).type(torch.bool)
            non_final_state_locations = (final_state_locations == False)
            non_final_states = next_states[non_final_state_locations] #(B',4,H,W)
            batch_size = next_states.shape[0]
            print("# of none terminal states = ", batch_size)
            values = torch.zeros(batch_size).to(QValues.device)
            if non_final_states.shape[0]==0: # BZX: check if there is survival
                print("EXCEPTION: this batch is all the last states of the episodes!")
                return values
            values[non_final_state_locations] = target_net(non_final_states).max(dim=1)[0]
            return values

    @staticmethod
    def DDQN_get_next(policy_net, target_net, next_states, mode = "stacked"):
        """
        To get Q_target, we need twice inference stage (one for policy net, another for target net)
        """
        if mode == "stacked":
            last_screens_of_state = next_states[:,-1,:,:] #(B,H,W)
            final_state_locations = last_screens_of_state.flatten(start_dim=1).max(dim=1)[0].eq(0).type(torch.bool)
            non_final_state_locations = (final_state_locations == False)
            non_final_states = next_states[non_final_state_locations] #(B',4,H,W)
            batch_size = next_states.shape[0]
            # print("# of none terminal states = ", batch_size)
            values = torch.zeros(batch_size).to(QValues.device)
            if non_final_states.shape[0]==0: # BZX: check if there is survival
                print("EXCEPTION: this batch is all the last states of the episodes!")
                return values
            # BZX: different from DQN
            argmax_a = policy_net(non_final_states).max(dim=1)[1]

            values[non_final_state_locations] = target_net(non_final_states).gather(dim=1, index=argmax_a.unsqueeze(-1)).squeeze(-1)
            return values


def visualize_state(state):
    # settings
    nrows, ncols = 1, 4  # array of sub-plots
    figsize = [8, 4]  # figure size, inches

    # prep (x,y) for extra plotting on selected sub-plots
    # xs = np.linspace(0, 2 * np.pi, 60)  # from 0 to 2pi
    # ys = np.abs(np.sin(xs))  # absolute of sine

    # create figure (fig), and array of axes (ax)
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

    # plot simple raster image on each sub-plot
    for i, axi in enumerate(ax.flat):
        # i runs from 0 to (nrows*ncols-1)
        # axi is equivalent with ax[rowid][colid]
        img = state.squeeze(0)[i,None]
        cpu_img = img.squeeze(0).cpu()
        axi.imshow(cpu_img*255,cmap='gray', vmin=0, vmax=255)

        # get indices of row/column
        rowid = i // ncols
        colid = i % ncols
        # write row/col indices as axes' title for identification
        # axi.set_title("Row:" + str(rowid) + ", Col:" + str(colid))

    # one can access the axes by ax[row_id][col_id]
    # do additional plotting on ax[row_id][col_id] of your choice
    # ax[0][2].plot(xs, 3 * ys, color='red', linewidth=3)
    # ax[4][3].plot(ys ** 2, xs, color='green', linewidth=3)

    plt.tight_layout(True)
    plt.show()