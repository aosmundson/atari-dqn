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
steps_done = 0

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

Transition = namedtuple('Transition',
        ('state', 'action', 'reward', 'newstate'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity #replaces oldest transitions once at capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):

    def __init__(self):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(4, 16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        self.fc1 = nn.Linear(2592, 256)
        self.fc2 = nn.Linear(256, 6)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 2592)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def act(self, state):
        global steps_done
        roll = random.random()
        if steps_done < 1000000:
            eps = 1-0.9*steps_done/1000000
        else:
            eps = 0.05
        steps_done += 1
        if roll > eps:
            with torch.no_grad():
                return self(state).max(1)[1].view(1,1)
        else:
            return torch.tensor([[random.randrange(6)]], dtype=torch.long)


resize = T.Compose([T.ToPILImage(),
    T.ToTensor()])

def preprocess(obs):
    im = Image.fromarray(obs)
    im = im.convert('L')
    im = im.resize((84,110))
    w, h = im.size
    im = im.crop((0, 26, w, h))
    return np.array(im)


dqn = DQN()
target_dqn = DQN()
optimizer = optim.RMSprop(dqn.parameters())
memory = ReplayMemory(10000)
BATCH_SIZE = 32
GAMMA = 0.99
TARGET_UPDATE = 10

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return

    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    non_terminal_mask = torch.tensor(tuple(map(lambda s: s is not None,
        batch.newstate)), dtype=torch.uint8)
    non_terminal_newstates = torch.cat([s for s in batch.newstate if s is not None])

    states = torch.cat(batch.state)
    actions = torch.cat(batch.action)
    rewards = torch.cat(batch.reward)

    policy_Qs = dqn(states).gather(1, actions)

    newstate_Vs = torch.zeros(BATCH_SIZE)
    newstate_Vs[non_terminal_mask] = target_dqn(non_terminal_newstates).max(1)[0].detach()
    target_Qs = rewards + GAMMA*newstate_Vs

    loss = F.mse_loss(policy_Qs, target_Qs.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    for param in dqn.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


for i in range(1000):
    frame0 = env.reset()
    frame1 = frame0
    frame2 = frame0
    frame3 = frame0
    preprocessed = (preprocess(frame0), preprocess(frame1),
            preprocess(frame2), preprocess(frame3))
    state = np.stack(preprocessed, axis=2)
    state = resize(state).unsqueeze(0)
    total_reward = 0
    while True:
        env.render()
        #action = env.action_space.sample()
        action = dqn.act(state)
        frame3 = frame2
        frame2 = frame1
        frame1 = frame0
        frame0, reward, done, info = env.step(action.item())
        preprocessed = (preprocess(frame0), preprocess(frame1),
                preprocess(frame2), preprocess(frame3))
        if not done:
            newstate = np.stack(preprocessed, axis=2)
            newstate = resize(newstate).unsqueeze(0)
        else:
            newstate = None
        memory.push(state, action, torch.tensor([reward]), newstate)
        print(dqn(state))
        state = newstate
        total_reward += reward
        optimize_model()
        if done:
            #for i in range(len(preprocessed)):
                #plt.subplot(1, 4, i+1)
                #plt.imshow(preprocessed[i])
            print("Got %d points" % total_reward)
            plt.show()
            break

    if i % TARGET_UPDATE == 0:
        target_dqn.load_state_dict(dqn.state_dict())
