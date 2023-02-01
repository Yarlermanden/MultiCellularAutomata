import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch_geometric
from torch_geometric.nn import GCNConv
from graphUtils import add_edges, add_random_food

#all of the usual pytorch model stuff + pytorch_geometric stuff


#forward method with number of time steps

#update method for updating a single timestep
#add edges dynamically using graphUtils

#method for updating velocity - machine learning - gcn

#method for updating location from velocity
#check for if food can be consumed


class GNCA(nn.Module):
    def __init__(self, device, channels=5):
        #batching?
        super(GNCA, self).__init__()
        self.device = device

        self.radius = 0.02
        self.acceleration_scale = 0.02
        self.max_velocity = 0.1
        self.max_pos = 1

        #layer setup
        #self.conv_layers1 = GCNConv(in_channels=channels, out_channels=channels)
        #self.conv_layers2 = GCNConv(in_channels=channels, out_channels=1)
        self.conv_layers = GCNConv(in_channels=channels, out_channels=2)

    def convolve(self, graph):
        '''Convolves the graph for message passing'''
        h = self.conv_layers(graph.x, graph.edge_index)
        return h

    def update_velocity(self, graph, acceleration):
        velocity = graph.x[:, 2:4] + acceleration #update velocity
        velocity = torch.clamp(velocity, -self.max_velocity, self.max_velocity)
        return velocity

    def update_positions(self, graph, velocity):
        positions = graph.x[:, :2] + velocity #update position
        #positions = torch.clamp(positions, -self.max_pos, self.max_pos)
        return positions

    def mask_food(self, graph):
        '''Used to get mask to only update all cell nodes - aka not food sources'''
        mask = graph.x[:, 4] == 1
        return mask.to(torch.float)

    def update(self, graph):
        '''Update the graph a single time step'''
        graph = add_edges(graph, self.radius, self.device) #dynamically add edges
        food_mask = self.mask_food(graph)

        acceleration = self.convolve(graph) * self.acceleration_scale #get acceleration
        acceleration = acceleration * torch.stack((food_mask, food_mask), dim=1)

        #TODO mask away acceleration from food sources - such that they can't move ....

        #compute the velocity and position + check if legal
        velocity = self.update_velocity(graph, acceleration)
        positions = self.update_positions(graph, velocity)

        graph.x[:, 2:4] = velocity
        graph.x[:, :2] = positions

        maskX = graph.x[:, 0].abs() > 1
        maskY = graph.x[:, 1].abs() > 1
        border_costX = graph.x[:, 0].abs().log() * maskX.to(torch.float) 
        border_costY = graph.x[:, 1].abs().log() * maskY.to(torch.float)
        border_cost = (border_costX.sum() + border_costY.sum())
        #position_cost = graph.x[:, :2].abs().log()

        #TODO check for whether food can be consumed

        graph = graph.to(device=self.device)
        return graph, velocity.abs().mean(dim=0), positions.abs().mean(dim=0), border_cost

    def forward(self, graph, time_steps = 1):
        '''update the graph n times for n time steps'''
        #optionally compute losses
        velocity_bonus = torch.tensor([0.0,0.0], device=self.device)
        position_penalty = torch.tensor([0.0,0.0], device=self.device)
        border_costs = 0

        #add_random_food(graph, 2)

        for i in range(time_steps):
            if i % 30 == 0:
                add_random_food(graph)
            graph, velocity, position, border_cost = self.update(graph)
            velocity_bonus += velocity
            position_penalty += position
            border_costs += border_cost

        return graph, velocity_bonus, position_penalty, border_costs