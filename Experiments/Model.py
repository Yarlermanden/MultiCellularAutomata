import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Complex_CA(nn.Module):
    def __init__(self, device, batch_size):
        super(Complex_CA, self).__init__()
        self.device = device
        self.fc1 = nn.Linear(12, 64)
        self.fc2 = nn.Linear(64, 4)
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

    def perceive_scent(self, food):
        food = food.view(self.batch_size,1,17,17)
        x = F.conv2d(food, self.scent_conv_weights, padding=2)[:, 0]
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
            return F.conv2d(x.view(self.batch_size,4, 17, 17), conv_weights, padding=1, groups=4)

        y1 = _perceive_with(x, self.dx)
        y2 = _perceive_with(x, self.dy)
        y = torch.cat((x,y1,y2), dim=1) 
        return y

    #For each in batch, only let the kth largest cells in cell layer survive
    def keep_k_largest(self, cell, kths):
        input = cell[:, 0:1].view(self.batch_size, -1)
        sorted, _ = torch.sort(input, dim=1, descending=True)
        kths = kths.to(torch.long)-1
        kth_values = sorted[torch.arange(len(kths)), kths] #(16)
        masked = (input.transpose(0, 1) >= kth_values[torch.arange(0, len(kth_values))]).transpose(0,1) #(16, 289)
        cell[:, 0:1] = torch.where(masked, input, torch.zeros(1,1, dtype=torch.float, device=self.device)).view(16, 1, 17, 17)
        return cell

    def kth_smallest(tensor, indices):
        tensor_sorted, _ = torch.sort(tensor)
        return tensor_sorted[torch.arange(len(indices)), indices]

    def update(self, cell, food):
        #TODO: handle somewhere in some way if food is reached and consumed. Remove food and increase CA size

        x = cell
        x[:, 3] = self.perceive_scent(food) #update scent 

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
        #TODO clamp x[:, 0] to be between 0 and 1
        #TODO: Simulate consumption of food - something like at this point - if some 3x3 kernel on the food result in some value above a threshold, then the cell has consumed the food
        #In that case remove the food and add 1 to the value of the cell at the exact same location - does it learn to grow from this?
        #What to do with the missing food now?
        


        #TODO we need some way of ensuring cells close to each other have a much much much lower cost difference than cells far from each other...
        #TODO definely need some better way of measuring loss...

        #Check if food can be consumed - consume and generate new food - not directly on top of CA

        return x, food
        
    def forward(self, cell: torch.Tensor, food: torch.Tensor, steps: int):
        #min_living = self.living_cells_above(cell[:, 0:1], 0.1)
        #max_living = min_living
        current_living = self.living_cells_above(cell[:, 0:1], 0.1)
        for _ in range(steps):
            cell, food = self.update(cell, food)
            #mask out cells except kth largest - ensure cells can't grow
            cell = self.keep_k_largest(cell, current_living)
            current_living = self.living_cells_above(cell[:, 0:1], 0.1)
            #current_living = self.living_cells_above(cell[:, 0:1], 0.1)
            #min_living = torch.minimum(min_living, current_living)
            #max_living = torch.maximum(max_living, current_living)

        total_pixel_val = self.total_pixel_val(cell[:, 0:1])
        living_count = self.living_cells_above(cell[:, 0:1], 0.1)
        return cell, food, total_pixel_val, living_count #, min_living, max_living