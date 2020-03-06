import math
import random
import torch
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display
else:
    matplotlib.use('TkAgg')

Experience = namedtuple(
    'Experience',
    ('state', 'action', 'next_state', 'reward')
)

class ReplayMemory():
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.push_count = 0

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

class EpsilonGreedyStrategy():
    def __init__(self, start, end, decay):
        self.start = start
        self.end = end
        self.decay = decay

    def get_exploration_rate(self, current_step):
        return self.end + (self.start - self.end) * \
               math.exp(-1. * current_step * self.decay)

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
    plt.pause(0.001)
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
    def get_next(target_net, next_states, mode = "stacked"):
        if mode == "stacked":
            last_screens_of_state = next_states[:,-1,:,:] #(B,H,W)
            final_state_locations = last_screens_of_state.flatten(start_dim=1).max(dim=1)[0].eq(0).type(torch.bool)
            non_final_state_locations = (final_state_locations == False)
            non_final_states = next_states[non_final_state_locations] #(B',4,H,W)
            batch_size = next_states.shape[0]
            values = torch.zeros(batch_size).to(QValues.device)
            if non_final_states.shape[0]==0: # BZX: check if there is survival
                print("EXCEPTION: this batch is all the last states of the episodes!")
                return values
            values[non_final_state_locations] = target_net(non_final_states).max(dim=1)[0]
            return values

