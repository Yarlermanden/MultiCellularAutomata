import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Complex_CA(nn.Module):
    def __init__(self, device):
        super(Complex_CA, self).__init__()
        #self.conv1 = nn.Conv2d(1, 1, 3, padding=1) # food spread smell
        #self.conv2 = nn.Conv2d(4, 8, 3, padding=1) # CA perceive world
        #self.conv3 = nn.Conv2d(8, 8, 1, padding=0) # use hidden parameters
        #self.conv4 = nn.Conv2d(8, 4, 1, padding=0) #TODO could try replacing this to 3 and add on the last layer later
        self.device = device

        self.conv3 = nn.Conv2d(12, 16, 1, padding=0) # use hidden parameters
        self.conv4 = nn.Conv2d(16, 4, 1, padding=0) #TODO could try replacing this to 3 and add on the last layer later

        def init_weights(m):
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.zeros_(m.bias)
        with torch.no_grad():
            self.apply(init_weights)
        self.alive = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

    def perceive_scent(self, food):
        conv_weights = torch.tensor([[[
            [0.0, 0.125, 0.25, 0.125, 0.0],
            [0.125, 0.25, 0.5, 0.25, 0.125],
            [0.25, 0.5, 1, 0.5, 0.25],
            [0.125, 0.25, 0.5, 0.25, 0.125],
            [0.0, 0.125, 0.25, 0.125, 0.0]
        ]]])
        #food = torch.stack([torch.stack([food])])
        food = food.view(1,1,17,17)
        x = F.conv2d(food, conv_weights, padding=2)[0][0]
        return x

    def alive_filter(self, x):
        return self.alive(x[0:1]) > 0.1

    def alive_count(self, x):
        #TODO maybe best to only look at nearest x neighbors and not total count in case we make a big world
        #x = torch.stack([x])
        #conv_weights = torch.ones_like(x)
        #live_count = F.conv2d(x, conv_weights)[0][0]
        live_count = torch.sum(x)
        return live_count

    def perceive_cell_surrounding(self, x):
        def _perceive_with(x, conv_weights):
            #TODO understand this and see if it's possible to do without group - what's the effect
            #TODO also see what the effect of doing 2 instead of 4 in kernel is...

            conv_weights = conv_weights.view(1,1,3,3).repeat(4, 1, 1, 1)
            return F.conv2d(x.view(1,4, 17, 17), conv_weights, padding=1, groups=4)[0]

        dx = torch.outer(torch.tensor([1,2,1]), torch.tensor([-1, 0, 1])) / 8.0 # Sobel filter
        dy = torch.transpose(dx, 0, 1)

        y1 = _perceive_with(x, dx)
        y2 = _perceive_with(x, dy)
        y = torch.cat((x,y1,y2),0) #what does the last 0 do?
        #y = torch.relu(self.conv2(x)) #perceive neighbor cell state
        return y

    def update(self, cell, food):
        #TODO: handle somewhere in some way if food is reached and consumed. Remove food and increase CA size
        x = cell

        x[3] = self.perceive_scent(food) #update scent 

        pre_life_mask = self.alive_filter(x)
        x = self.perceive_cell_surrounding(x) #perceive neighbor cell states
        x = torch.relu(self.conv3(x)) #single cell 
        x = self.conv4(x) #single cell

        x = cell + x

        post_life_mask = self.alive_filter(x) #only do this on cell state
        life_mask = torch.bitwise_and(pre_life_mask, post_life_mask).to(torch.float)
        x = x*life_mask
        x[3] = cell[3] #ensure smell stay consistent #TODO does this break the chain?
        x = torch.clamp(x, -10.0, 10.0)
        #x[0] = torch.clamp(x[0], 0.0, 1.0) #TODO ensure values are between 0 and 1

        #TODO hidden states and especially smell will grow incontrollable if we don't optimize on it
        #TODO some way of forcing higher loss from counting wrong? - some way of counting and backpropagating?

        #TODO could add clamp forcing cells to either be dead or alive - all dead cells need to have reset their hidden states reset exect for smell
        #Mask cells to either be dead or alive
        #alive = (x[0] > 0.5).to(torch.float)
        #x[0] = x[0] * alive


        #TODO we need some way of ensuring cells close to each other have a much much much lower cost difference than cells far from each other...
        #TODO definely need some better way of measuring loss...
        #amount of living cells...
        #placement of living cells...

        return x, food
        

    def forward(self, cell: torch.Tensor, food: torch.Tensor, steps: int):
        for _ in range(steps):
            cell, food = self.update(cell, food)

        #TODO could maybe even add count of cells above some threshold e.g. 0.9 - force completely living cells
        living_count = self.alive_count(cell[0:1])
        return cell, food, living_count