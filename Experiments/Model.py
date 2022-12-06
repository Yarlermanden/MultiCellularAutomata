import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Complex_CA(nn.Module):
    def __init__(self, device, batch_size):
        super(Complex_CA, self).__init__()
        self.device = device
        self.fc1 = nn.Linear(12, 64)
        self.fc2 = nn.Linear(64, 4)
        self.batch_size = batch_size
        self.grid_dim = 17*2

        #self.scent_conv_weights = torch.tensor([[[
        #    [0.02, 0.04, 0.07, 0.10, 0.12, 0.13, 0.12, 0.10, 0.07, 0.04, 0.02],
        #    [0.04, 0.08, 0.13, 0.20, 0.25, 0.28, 0.25, 0.20, 0.13, 0.08, 0.04],
        #    [0.07, 0.13, 0.23, 0.35, 0.44, 0.48, 0.44, 0.35, 0.23, 0.13, 0.07],
        #    [0.10, 0.20, 0.35, 0.52, 0.66, 0.72, 0.66, 0.52, 0.35, 0.20, 0.10],
        #    [0.12, 0.25, 0.44, 0.66, 0.84, 0.91, 0.84, 0.66, 0.44, 0.25, 0.12],
        #    [0.13, 0.28, 0.48, 0.72, 0.91, 0.99, 0.91, 0.72, 0.48, 0.28, 0.13],
        #    [0.12, 0.25, 0.44, 0.66, 0.84, 0.91, 0.84, 0.66, 0.44, 0.25, 0.12],
        #    [0.10, 0.20, 0.35, 0.52, 0.66, 0.72, 0.66, 0.52, 0.35, 0.20, 0.10],
        #    [0.07, 0.13, 0.23, 0.35, 0.44, 0.48, 0.44, 0.35, 0.23, 0.13, 0.07],
        #    [0.04, 0.08, 0.13, 0.20, 0.25, 0.28, 0.25, 0.20, 0.13, 0.08, 0.04],
        #    [0.02, 0.04, 0.07, 0.10, 0.12, 0.13, 0.12, 0.10, 0.07, 0.04, 0.02]
        #]]], device=self.device)
        self.scent_conv_weights = self.generate_scent_conv_weights(19).to(self.device)

        self.dx = torch.outer(torch.tensor([1,2,1], device=self.device), torch.tensor([-1, 0, 1], device=self.device)) / 8.0 # Sobel filter
        self.dy = torch.transpose(self.dx, 0, 1)

        def init_weights(m):
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.zeros_(m.bias)
        with torch.no_grad():
            self.apply(init_weights)

    def generate_scent_conv_weights(self, size):
        h, w = size, size
        x0, y0 = torch.rand(2, 1)
        x0, y0 = torch.tensor(([0.5], [0.5]))

        origins = torch.stack((x0*h, y0*w)).T

        def gaussian_2d(x=0, y=0, mx=0, my=0, sx=1, sy=1):
            return 1 / (2*math.pi*sx*sy) * \
                torch.exp(-((x - mx)**2 / (2*sx**2) + (y - my)**2 / (2*sy**2)))

        x = torch.linspace(0, h, h)
        y = torch.linspace(0, w, w)
        x, y = torch.meshgrid(x, y)

        z = torch.zeros(h, w)
        for x0, y0 in origins:
            z += gaussian_2d(x, y, mx=x0, my=y0, sx=h/4, sy=w/4)*80
        return z.view(1, 1, size, size)

    def perceive_scent(self, food):
        food = food.view(self.batch_size,1,self.grid_dim,self.grid_dim)
        x = F.conv2d(food, self.scent_conv_weights, padding=9)[:, 0]
        return x

    def alive_filter(self, x):
        alive = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        return alive(x[:, 0:1, :, :]) > 0.1

    def living_cells_above(self, x, threshold):
        return (x > threshold).to(torch.float).sum(dim=(1,2,3))

    def total_pixel_val(self, x):
        live_count = x.sum(dim=(1,2,3))
        return live_count

    def perceive_cell_surrounding(self, x):
        def _perceive_with(x, conv_weights):
            conv_weights = conv_weights.view(1,1,3,3).repeat(4, 1, 1, 1)
            return F.conv2d(x.view(self.batch_size, 4, self.grid_dim, self.grid_dim), conv_weights, padding=1, groups=4)

        y1 = _perceive_with(x, self.dx)
        y2 = _perceive_with(x, self.dy)
        y = torch.cat((x,y1,y2), dim=1) 
        return y

    #For each in batch, only let the kth largest cells in cell layer survive
    def keep_k_largest(self, cell, kths):
        input = cell[:, 0:1].view(self.batch_size, -1)
        sorted, _ = torch.sort(input, dim=1, descending=False)
        kths = (torch.full(size=(kths.shape), fill_value=(input.shape[1]-1), device=self.device) - kths).to(torch.long)
        kths = (input.shape[1]-kths-1).to(torch.long)
        #sorted, _ = torch.sort(input, dim=1, descending=True)
        #TODO: should be fixed...
        #kths = kths.to(torch.long)-1
        kth_values = sorted[torch.arange(len(kths)), kths] #(batch_size)
        masked = (input.transpose(0, 1) > kth_values[torch.arange(0, len(kth_values))]).transpose(0,1) #(batch_size, grid_dim^2)
        cell[:, 0:1] = torch.where(masked, input, torch.zeros(1,1, dtype=torch.float, device=self.device)).view(self.batch_size, 1, self.grid_dim, self.grid_dim)
        return cell

    def update(self, cell, food):
        current_living = self.living_cells_above(cell[:, 0:1], 0.8)
        x = cell
        #x[:, 3] = self.perceive_scent(food) #update scent 

        pre_life_mask = self.alive_filter(x)
        x = self.perceive_cell_surrounding(x) #perceive neighbor cell states

        x = x.transpose(1, 3)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        x = x.transpose(1, 3)

        x = cell + x

        post_life_mask = self.alive_filter(x) #only do this on cell state
        life_mask = torch.bitwise_and(pre_life_mask, post_life_mask).to(torch.float)
        x = x*life_mask

        x = torch.clamp(x, -10.0, 10.0)

        #only for evolution
        x[:, 0] = torch.clamp(x[:, 0], 0.0, 1.0)
        
        #Check if food can be consumed - consume and generate new food - not directly on top of CA
        #TODO: Simulate consumption of food - something like at this point - if some 3x3 kernel on the food result in some value above a threshold, then the cell has consumed the food
        #In that case remove the food and add 1 to the value of the cell at the exact same location - does it learn to grow from this?
        #What to do with the missing food now?

        x = self.keep_k_largest(x, current_living)
        return x, food
        
    def forward(self, cell: torch.Tensor, food: torch.Tensor, steps: int):
        #min_living = self.living_cells_above(cell[:, 0:1], 0.1)
        #max_living = min_living
        #current_living = self.living_cells_above(cell[:, 0:1], 0.8)
        scent = self.perceive_scent(food)
        for _ in range(steps):
            cell[:, 3] = scent
            cell, food = self.update(cell, food)
            #mask out cells except kth largest - ensure cells can't grow

            #cell = self.keep_k_largest(cell, current_living)
            #current_living = self.living_cells_above(cell[:, 0:1], 0.8)

            #current_living = self.living_cells_above(cell[:, 0:1], 0.1)
            #min_living = torch.minimum(min_living, current_living)
            #max_living = torch.maximum(max_living, current_living)

        total_pixel_val = self.total_pixel_val(cell[:, 0:1])
        living_count = self.living_cells_above(cell[:, 0:1], 0.1)
        return cell, food, total_pixel_val, living_count #, min_living, max_living