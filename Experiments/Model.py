import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Complex_CA(nn.Module):
    def __init__(self, device, batch_size, toxic=False, evolution=True):
        super(Complex_CA, self).__init__()
        self.device = device
        self.conv1 = nn.Conv2d(12, 12, 1, padding=0)
        self.conv2 = nn.Conv2d(12, 12, 1, padding=0)
        self.conv3 = nn.Conv2d(12, 4, 1, padding=0)
        self.batch_size = batch_size
        self.grid_dim = 34
        self.scent_spread = 19
        self.toxic_spread = 27
        self.toxic = toxic
        self.evolution  = evolution

        self.scent_conv_weights = self.generate_scent_conv_weights(self.scent_spread).to(self.device)
        self.toxic_conv_weights = self.generate_scent_conv_weights(self.toxic_spread, 250).to(self.device)

        self.dx = torch.outer(torch.tensor([1,2,1], device=self.device), torch.tensor([-1, 0, 1], device=self.device)) / 8.0 # Sobel filter
        self.dy = torch.transpose(self.dx, 0, 1)

        def init_weights(m):
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.zeros_(m.bias)
        with torch.no_grad():
            self.apply(init_weights)

    def generate_scent_conv_weights(self, size, value=80):
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
            z += gaussian_2d(x, y, mx=x0, my=y0, sx=h/4, sy=w/4)*value
        return z.view(1, 1, size, size)

    def perceive_scent(self, food):
        food = food.view(self.batch_size,1,self.grid_dim,self.grid_dim)
        x = F.conv2d(food, self.scent_conv_weights, padding=9)[:, 0]
        return x

    def perceive_toxic(self, food):
        food = food.view(self.batch_size,1,self.grid_dim,self.grid_dim)
        x = F.conv2d(food, self.toxic_conv_weights, padding=13)[:, 0]
        return x

    def update_cell(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.conv3(x)
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
        kth_values = sorted[torch.arange(len(kths)), kths] #(batch_size)
        masked = (input.transpose(0, 1) > kth_values[torch.arange(0, len(kth_values))]).transpose(0,1) #(batch_size, grid_dim^2)
        cell[:, 0:1] = torch.where(masked, input, torch.zeros(1,1, dtype=torch.float, device=self.device)).view(self.batch_size, 1, self.grid_dim, self.grid_dim)
        return cell

    def detect_rulebreaks(self, cell, x):
        largePool = nn.AvgPool2d(kernel_size=5, stride=1, padding=2)
        smallPool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        new_cell = (cell[:, 0:1] > 0.1).to(torch.float)
        new_x = (x[:, 0:1] > 0.1).to(torch.float)

        cell_sum_2x2 = smallPool(new_cell[:, 0:1, :, :])*9
        cell_sum_4x4 = largePool(new_cell[:, 0:1, :, :])*25
        x_sum_2x2 = smallPool(new_x[:, 0:1, :, :])*9
        x_sum_4x4 = largePool(new_x[:, 0:1, :, :])*25

        dead_mask = x_sum_4x4 >= cell_sum_2x2 #if not we have rule break
        growth_mask = x_sum_2x2 <= cell_sum_4x4

        mask = torch.bitwise_and(dead_mask, growth_mask).to(torch.float)
        x[:, 0:1] = x[:, 0:1] * mask
        return x


    def update(self, cell, food):
        x = cell.clone()

        pre_life_mask = self.alive_filter(x)
        x = self.perceive_cell_surrounding(x) #perceive neighbor cell states

        x = self.update_cell(x) #update cell depending on all hidden states
        x = cell + x #Added as the model should both be able to add and subtract from the previous

        post_life_mask = self.alive_filter(x) #only do this on cell state
        life_mask = torch.bitwise_and(pre_life_mask, post_life_mask).to(torch.float)
        x = x*life_mask

        x = torch.clamp(x, -10.0, 10.0)
        
        #Check if food can be consumed - consume and generate new food - not directly on top of CA
        #TODO: Simulate consumption of food - something like at this point - if some 3x3 kernel on the food result in some value above a threshold, then the cell has consumed the food
        #In that case remove the food and add 1 to the value of the cell at the exact same location - does it learn to grow from this?

        if self.evolution: #only for evolution
            x[:, 0] = torch.clamp(x[:, 0], 0.0, 1.0)
            current_living = self.living_cells_above(cell[:, 0:1], 0.1)
            x = self.keep_k_largest(x, current_living)
            x = self.detect_rulebreaks(cell, x)
        return x, food
        
    def forward(self, cell: torch.Tensor, food: torch.Tensor, steps: int):
        if self.toxic:
            scent = self.perceive_toxic(food)
        else:
            scent = self.perceive_scent(food)

        toxic_cost = 0
        for i in range(steps):
            cell[:, 3] = scent
            cell, food = self.update(cell, food)
            if self.toxic:
                toxic_cost += (cell[:, 0] * food).sum()*i #makes it worse over time

        total_pixel_val = self.total_pixel_val(cell[:, 0:1])
        living_count = self.living_cells_above(cell[:, 0:1], 0.1)
        return cell, food, total_pixel_val, living_count, toxic_cost