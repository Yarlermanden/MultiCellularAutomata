
import torch
import torch.nn as nn

class Complex_CA(nn.Module):
    def __init__(self):
        super(Complex_CA, self).__init__()
        self.conv1 = nn.Conv2d(1, 1, 3, padding=1) # food spread smell

        #CA, hidden parameters/speed, smell
        self.conv2 = nn.Conv2d(4, 8, 3, padding=1) # CA perceive world
        self.conv3 = nn.Conv2d(8, 8, 1, padding=0) 
        #CA, hidden parameters/speed
        self.conv4 = nn.Conv2d(8, 4, 1, padding=0)
        with torch.no_grad():
            self.conv4.weight.zero_()
        self.alive = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

    def perceive_smell(self, food, smell): #smell spread from food
        smell += food #smell source is food
        smell = smell*0.8 #decay smell
        smell = torch.relu(self.conv1(torch.stack([smell]))) #smell spread
        return smell[0]

    def alive_filter(self, x):
        #only look at cell state - could be changed to also consider smell
        #return self.alive(x[0, :, :]) > 0.1 #only 
        return self.alive(x) > 0.1

    def perceive_cell_surrounding(self, x):
        return ...

    def update(self, cell, food):
        cell[3] = self.perceive_smell(cell[3], food) #update smell

        pre_life_mask = self.alive_filter(cell)
        x = torch.relu(self.conv2(cell)) #perceive neighbor cell states
        x = torch.relu(self.conv3(x)) #single cell 
        x = self.conv4(x) #single cell
        post_life_mask = self.alive_filter(x) #only do this on cell state
        life_mask = torch.bitwise_and(pre_life_mask, post_life_mask).to(torch.float)
        x = x*life_mask

        #maybe do this before post_life_mask
        x[2] = cell[2] #ensure smell stay consistent

        #TODO hidden states and especially smell will grow incontrollable if we don't optimize on it

        #x - cell, hidden states, smell
        #cell state changes, hidden states changes, smell stays relative consistent
        #food - stay consistent
        return x, food
        

    def forward(self, cell, food, steps):
        for _ in range(steps):
            #cell, hidden, smell
            #food
            cell, food = self.update(cell, food)
        return cell, food