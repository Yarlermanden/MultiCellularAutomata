
import torch
import torch.nn as nn
import torch.nn.functional as F

class Complex_CA(nn.Module):
    def __init__(self):
        super(Complex_CA, self).__init__()
        self.conv1 = nn.Conv2d(1, 1, 3, padding=1) # food spread smell
        self.conv2 = nn.Conv2d(4, 8, 3, padding=1) # CA perceive world
        self.conv3 = nn.Conv2d(8, 8, 1, padding=0) # use hidden parameters
        self.conv4 = nn.Conv2d(8, 4, 1, padding=0) #TODO could try replacing this to 3 and add on the last layer later

        def init_weights(m):
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.zeros_(m.bias)
        with torch.no_grad():
            self.apply(init_weights)

        self.alive = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

    #def perceive_smell(self, food, smell): #smell spread from food
    #    smell += food #smell source is food
    #    smell = smell*0.8 #decay smell
    #    smell = torch.relu(self.conv1(torch.stack([smell]))) #smell spread
    #    return smell[0]

    def perceive_scent(self, food):
        conv_weights = torch.tensor([[[
            [0.25, 0.5, 0.25],
            [0.5, 1, 0.5],
            [0.25, 0.5, 0.25]
        ]]])
        food = torch.stack([food])
        food = torch.stack([food])
        x = F.conv2d(food, conv_weights, padding=1)[0][0]
        return x

    def alive_filter(self, x):
        #only look at cell state - could be changed to also consider smell
        #return self.alive(x[0, :, :]) > 0.1 #only 
        return self.alive(x[0:1]) > 0.1

    def perceive_cell_surrounding(self, x):
        x = torch.relu(self.conv2(x)) #perceive neighbor cell state
        return x

    def update(self, cell, food):
        x = cell

        #x[3] = self.perceive_smell(x[3], food) #update smell
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
        return cell, food