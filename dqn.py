import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T


env = gym.make('Breakout-v0').unwrapped

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

    plt.ion()

    # if gpu is to be used
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DQN():

    def __init__(self):
        self.vals = {}



def get_screen(obs):
    im = Image.fromarray(obs)
    im = im.convert('LA')
    im = im.resize((84,110))
    w, h = im.size
    im = im.crop((0, 26, w, h))
    return im


while True:
    obs = env.reset()
    frame1 = obs
    frame2 = obs
    frame3 = obs
    total_reward = 0
    while True:
        env.render()
        action = env.action_space.sample()
        frame3 = frame2
        frame2 = frame1
        frame1 = obs
        obs, reward, done, info = env.step(action)
        total_reward += reward
        if done:
            plt.subplot(1, 4, 1)
            plt.imshow(obs)
            plt.subplot(1, 4, 2)
            plt.imshow(frame1)
            plt.subplot(1, 4, 3)
            plt.imshow(frame2)
            plt.subplot(1, 4, 4)
            plt.imshow(frame3)
            print("Got %d points" % total_reward)
            plt.show()
            break
