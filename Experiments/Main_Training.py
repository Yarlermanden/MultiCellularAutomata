#!/usr/bin/env python
# coding: utf-8

# ## Script for training and saving models

# #### Layout of files

# Model.py
# - Defines the entire model
#     - Layers
#     - How it trainsforms data from input to output
# 
# Trainer.py
# - Defines how the model is trained 
# - Defines curriculum learning
#     - How it's trained initially
#         - X steps just to get similar output as input
#         - make it stable
#     - How it's trained when we assume it's getting smarter
#         - Learn it to move in the direction of the reward
# 
# Utils.py
# - Contains all helper functions
# 
# States.py
# - How to get new random states
#     - Output x, y and food
#         - x being the random initial state
#             - Should be of size in range of (x1, x2)
#             - Should be one entity complying with certain rules
#         - y being the target output after e epochs
#         - food being the desired target location determine the direction of the CA

# #### Imports

# In[1]:


import torch
import numpy as np
from Trainer import Trainer
from Model import Complex_CA


# #### Setup

# In[2]:


#device = torch.device('cuda:0' if torch.cuda.is_available else 'cpu')
device = torch.device('mps:0' if torch.backends.mps.is_available else 'cpu')
#device = torch.device('cpu')
batch_size = 16
model = Complex_CA(device, batch_size)
model = model.to(device)
trainer = Trainer(model, device)
print(device)


# #### Training

# In[3]:


seed = 2
#torch.manual_seed(seed)
#np.random.seed(seed)
model, losses = trainer.train()


# #### Save model and losses graph

# In[ ]:


#save model
torch.save(model.state_dict(), 'models/complex_ca7.pth')

#save graph
print(losses.shape)
np.save('losses_sigmoid', losses)

