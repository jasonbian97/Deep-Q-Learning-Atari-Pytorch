"""
    1. wrap the env object of Gym
    2. like the dataloader, we do image preprocessing in this class.
"""
import gym
import numpy as np
import torch
import torchvision.transforms as T

class BreakoutEnvManager():
    def __init__(self, device):
        self.device = device
        self.env = gym.make('BreakoutDeterministic-v4').unwrapped #BZX: v4 automatically return skipped screens
        self.env.reset()
        self.current_screen = None
        self.done = False
        # BZX: running_K: stacked K images together to present a state
        # BZx: running_queue: maintain the latest running_K images
        self.running_K = 4
        self.running_queue = []
        self.is_lives_loss = False
        self.current_lives = None

    def reset(self):
        self.env.reset()
        self.current_screen = None
        self.running_queue = [] #BZX: clear the state
        # self.current_lives = 5

    def close(self):
        self.env.close()

    def render(self, mode='human'):
        return self.env.render(mode)

    def num_actions_available(self):
        return self.env.action_space.n

    def print_action_meanings(self):
        print(self.env.get_action_meanings())

    def take_action(self, action):
        _, reward, self.done, lives = self.env.step(action.item())
        if lives['ale.lives'] < self.current_lives:
            self.is_lives_loss = True
        else:
            self.is_lives_loss = False
        self.current_lives = lives['ale.lives']
        # print(lives['ale.lives'])
        return torch.tensor([reward], device=self.device)
        # torch.tensor(self.done, device=self.device), torch.tensro(lives, device=self.device)

    def just_starting(self):
        return self.current_screen is None

    def can_provide_state(self):
        """
        BZX: if the number of black screens is less or equal to 1, then return True.
        else return False
        :return: bool
        """
        #TODO

    def init_running_queue(self):
        """
        initialize running queue with K black images
        :return:
        """
        self.current_screen = self.get_processed_screen()
        black_screen = torch.zeros_like(self.current_screen)
        for _ in range(self.running_K):
            self.running_queue.append(black_screen)

    def get_state(self):
        if self.just_starting():
            self.init_running_queue()
        elif self.done or self.is_lives_loss:
            self.current_screen = self.get_processed_screen()
            black_screen = torch.zeros_like(self.current_screen)
            # BZX: update running_queue
            self.running_queue.pop(0)
            self.running_queue.append(black_screen)
        else: #BZX: normal case
            s2 = self.get_processed_screen()
            self.current_screen = s2
            # BZx: update running_queue with s2
            self.running_queue.pop(0)
            self.running_queue.append(s2)

        return torch.stack(self.running_queue,dim=1).squeeze(2) #BZX: check if shape is (1KHW)


    def get_screen_height(self):
        screen = self.get_processed_screen()
        return screen.shape[2]

    def get_screen_width(self):
        screen = self.get_processed_screen()
        return screen.shape[3]

    def get_processed_screen(self):
        screen = self.render('rgb_array').transpose((2, 0, 1))  # PyTorch expects CHW
        # screen = self.crop_screen(screen)
        return self.transform_screen_data(screen) #shape is [1,1,110,84]

    def crop_screen(self, screen):
        bbox = [34,0,160,160]
        screen = screen[:, bbox[0]:bbox[1], bbox[2]:bbox[3]] #BZX:(CHW)
        return screen

    def transform_screen_data(self, screen):
        # Convert to float, rescale, convert to tensor
        screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
        screen = torch.from_numpy(screen)
        screen = self.crop_screen(screen)
        # Use torchvision package to compose image transforms
        resize = T.Compose([
            T.ToPILImage()
            , T.Grayscale()
            , T.Resize((84, 84)) #BZX: the original paper's settings:(110,84), however, for simplicty we...
            , T.ToTensor()
        ])
        # add a batch dimension (BCHW)
        screen = resize(screen)

        return screen.unsqueeze(0).to(self.device)   # BZX: Pay attention to the shape here. should be [1,1,84,84]


class CartPoleEnvManager():

    def __init__(self, device):
        self.device = device
        self.env = gym.make('CartPole-v0').unwrapped
        self.env.reset()
        self.current_screen = None
        self.done = False

    def reset(self):
        self.env.reset()
        self.current_screen = None

    def close(self):
        self.env.close()

    def render(self, mode='human'):
        return self.env.render(mode)

    def num_actions_available(self):
        return self.env.action_space.n

    def take_action(self, action):
        _, reward, self.done, _ = self.env.step(action.item())
        return torch.tensor([reward], device=self.device)

    def just_starting(self):
        return self.current_screen is None

    def get_state(self):
        if self.just_starting() or self.done:
            self.current_screen = self.get_processed_screen()
            black_screen = torch.zeros_like(self.current_screen)
            return black_screen
        else:
            s1 = self.current_screen
            s2 = self.get_processed_screen()
            self.current_screen = s2
            return s2 - s1

    def get_screen_height(self):
        screen = self.get_processed_screen()
        return screen.shape[2]

    def get_screen_width(self):
        screen = self.get_processed_screen()
        return screen.shape[3]

    def get_processed_screen(self):
        screen = self.render('rgb_array').transpose((2, 0, 1))  # PyTorch expects CHW
        screen = self.crop_screen(screen)
        return self.transform_screen_data(screen)

    def crop_screen(self, screen):
        screen_height = screen.shape[1]

        # Strip off top and bottom
        top = int(screen_height * 0.4)
        bottom = int(screen_height * 0.8)
        screen = screen[:, top:bottom, :]
        return screen

    def transform_screen_data(self, screen):
        # Convert to float, rescale, convert to tensor
        screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
        screen = torch.from_numpy(screen)

        # Use torchvision package to compose image transforms
        resize = T.Compose([
            T.ToPILImage()
            , T.Resize((40, 90))
            , T.ToTensor()
        ])

        return resize(screen).unsqueeze(0).to(self.device)  # add a batch dimension (BCHW)