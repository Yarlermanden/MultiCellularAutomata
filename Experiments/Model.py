import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Complex_CA(nn.Module):
    def __init__(self, device, batch_size):
        super(Complex_CA, self).__init__()
        #self.conv1 = nn.Conv2d(1, 1, 3, padding=1) # food spread smell
        #self.conv2 = nn.Conv2d(4, 8, 3, padding=1) # CA perceive world
        #self.conv3 = nn.Conv2d(8, 8, 1, padding=0) # use hidden parameters
        #self.conv4 = nn.Conv2d(8, 4, 1, padding=0) #TODO could try replacing this to 3 and add on the last layer later
        self.device = device

        self.conv3 = nn.Conv2d(12, 16, 1, padding=0) # use hidden parameters
        self.conv4 = nn.Conv2d(16, 4, 1, padding=0) #TODO could try replacing this to 3 and add on the last layer later
        self.batch_size = batch_size

        self.scent_conv_weights = torch.tensor([[[
            [0.0, 0.125, 0.25, 0.125, 0.0],
            [0.125, 0.25, 0.5, 0.25, 0.125],
            [0.25, 0.5, 1, 0.5, 0.25],
            [0.125, 0.25, 0.5, 0.25, 0.125],
            [0.0, 0.125, 0.25, 0.125, 0.0]
        ]]], device=self.device)

        self.dx = torch.outer(torch.tensor([1,2,1], device=self.device), torch.tensor([-1, 0, 1], device=self.device)) / 8.0 # Sobel filter
        self.dy = torch.transpose(self.dx, 0, 1)

        def init_weights(m):
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.zeros_(m.bias)
        with torch.no_grad():
            self.apply(init_weights)
        self.alive = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

    def perceive_scent(self, food):
        food = food.view(self.batch_size,1,17,17)
        x = F.conv2d(food, self.scent_conv_weights, padding=2)[:, 0]
        return x

    def alive_filter(self, x):
        return self.alive(x[:, 0:1, :, :]) > 0.1

    def live_count_above(self, x, threshold):
        return (x > threshold).to(torch.float).sum(dim=(1,2,3))

    def alive_count(self, x):
        #TODO maybe best to only look at nearest x neighbors and not total count in case we make a big world
        #x = torch.stack([x])
        #conv_weights = torch.ones_like(x)
        #live_count = F.conv2d(x, conv_weights)[0][0]

        live_count = x.sum(dim=(1,2,3))
        return live_count

    def perceive_cell_surrounding(self, x):
        def _perceive_with(x, conv_weights):
            conv_weights = conv_weights.view(1,1,3,3).repeat(4, 1, 1, 1)
            return F.conv2d(x.view(self.batch_size,4, 17, 17), conv_weights, padding=1, groups=4)

        y1 = _perceive_with(x, self.dx)
        y2 = _perceive_with(x, self.dy)
        y = torch.cat((x,y1,y2), dim=1) 
        #y = torch.relu(self.conv2(x)) #perceive neighbor cell state
        return y

    def update(self, cell, food):
        #TODO: handle somewhere in some way if food is reached and consumed. Remove food and increase CA size
        x = cell

        x[:, 3] = self.perceive_scent(food) #update scent 

        pre_life_mask = self.alive_filter(x)
        x = self.perceive_cell_surrounding(x) #perceive neighbor cell states
        x = torch.relu(self.conv3(x)) #single cell 
        x = self.conv4(x) #single cell

        x = cell + x

        #force harder boundaries - is this necessary when the other two masks almost do the same?
        #enabling this will cause the model to never move
        #threshold_mask = (x[:, 0] > 0.1).to(torch.float) #ensures cells have to be more than a certain amount alive to count - ensures harder boundaries
        #x[:, 0] = x[:, 0]*threshold_mask

        #Force into range between 0 and 1
        # Could replace with function that only allows 0 or 1 - argmax
        #x[:, 0] = torch.sigmoid(x[:, 0])
        #x[:, 0] = F.softmax(x[:, 0], dim=1) #softmax definitely won't work as it needs 2 outputs representing 0 or 1 for each cell to be softmaxed between...

        post_life_mask = self.alive_filter(x) #only do this on cell state
        life_mask = torch.bitwise_and(pre_life_mask, post_life_mask).to(torch.float)
        x = x*life_mask

        #x[0] = torch.relu(x[0]) #force all cells to be between 0 and 1
        x = torch.clamp(x, -10.0, 10.0)
        

        #TODO we need some way of ensuring cells close to each other have a much much much lower cost difference than cells far from each other...
        #TODO definely need some better way of measuring loss...
        #TODO could we measure loss from the center of the cell aswell - to reward for being close to the actual representation

        return x, food
        

    def forward(self, cell: torch.Tensor, food: torch.Tensor, steps: int):
        for _ in range(steps):
            cell, food = self.update(cell, food)

        #TODO could maybe even add count of cells above some threshold e.g. 0.9 - force completely living cells
        living_count = self.alive_count(cell[:, 0:1])
        #TODO could also try with a much lower threshold 
        # living_count should force close to equal amount of total pixel values while a low threshold ensures no cells with smaller values as these would quickly add up
        living_above = self.live_count_above(cell[:, 0:1], 0.1)
        return cell, food, living_count, living_above