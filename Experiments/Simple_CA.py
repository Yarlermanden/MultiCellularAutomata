#!/usr/bin/env python
# coding: utf-8

# Playing around with: https://www.kaggle.com/code/teddykoker/training-cellular-automata-part-i-game-of-life/notebook

# In[11]:


import time
import imageio

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from IPython.display import clear_output

from lib.CAModel import CAModel
from lib.utils_vis import SamplePool, to_alpha, to_rgb, get_living_mask, make_seed, make_circle_masks


# Initial Conway's Game of Life

# In[12]:


train_size, width, height = (10, 32, 32)
device = torch.device("mps")

def gol(x):
    # performs one step of Conway's Game of Life
    # http://jakevdp.github.io/blog/2013/08/07/conways-game-of-life/
    nbrs_count = sum(np.roll(np.roll(x, i, 0), j, 1)
                     for i in (-1, 0, 1) for j in (-1, 0, 1)
                     if (i != 0 or j != 0))
    return (nbrs_count == 3) | (x & (nbrs_count == 2))


x_train = np.random.randint(0, 2, size=(train_size, width, height))
y_train = np.stack([gol(d) for d in x_train])
x_train = np.stack([1 - x_train, x_train], axis=1) # one channel per color

x_train = torch.from_numpy(x_train).float().to(device)
y_train = torch.from_numpy(y_train).to(device)


# #### Model

# In[13]:


class CircularPad(nn.Module):
    def forward(self, x):
        return F.pad(x, (1,1,1,1), mode='circular')

gol_model = nn.Sequential(
    CircularPad(),
    nn.Conv2d(2, 8, kernel_size=3),
    nn.ReLU(),
    nn.Conv2d(8, 8, kernel_size=1),
    nn.ReLU(),
    nn.Conv2d(8, 2, kernel_size=1),
).to(device)


# #### Training

# In[15]:


num_epochs = 1000

optimizer = torch.optim.Adam(gol_model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

losses = np.zeros(num_epochs)

for e in range(num_epochs):
    optimizer.zero_grad()
    y_pred = gol_model(x_train)
    loss = criterion(y_pred, y_train)
    losses[e] = loss.item()
    loss.backward()
    optimizer.step()

plt.plot(losses)


# #### Test

# In[17]:


glider = np.array([[0,1,0],
                   [0,0,1],
                   [1,1,1]])
state = np.zeros((32, 32))
state[:3,:3] = glider
state = np.stack([1 - state, state])
state = torch.from_numpy(state).float().unsqueeze(0).to(device)

@torch.no_grad()
def animate(i):
    global state
    state = torch.softmax(gol_model(state), dim=1).to(device)
    mat.set_data(state.detach().cpu().numpy()[0,0])

fig, ax = plt.subplots()
mat = ax.matshow(state.detach().cpu().numpy()[0,0], cmap="gray")

anim = animation.FuncAnimation(fig, animate, frames=100, interval=60)
#plt.show()


# In[18]:


import os
import pygame
from lib.displayer import displayer
from lib.utils import mat_distance

os.environ['KMP_DUPLICATE_LIB_OK']='True'

pix_size = 12
_map_shape = (64,64)

glider = np.array([[0,1,0],
                   [0,0,1],
                   [1,1,1]])
state = np.zeros(_map_shape)
state[:3,:3] = glider
state = np.stack([1 - state, state])
state = torch.from_numpy(state).float().unsqueeze(0).to(device)
disp = displayer(_map_shape, pix_size)

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    state = torch.softmax(gol_model(state), dim=1).to(device)
    rgb = state.detach().cpu().numpy()[0,0]
    #state.detach().cpu().reshape([1,_map_shape[0],_map_shape[1],2)

    #rgb = np.dstack((rgb, rgb, rgb)).astype(np.float32)
    #_map = np.zeros_like(rgb, dtype=np.float32)
    #_map += rgb

    _map = np.dstack((rgb, rgb, rgb)).astype(np.float32)

    disp.update(_map)

