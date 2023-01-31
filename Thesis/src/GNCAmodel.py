import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch_geometric
from torch_geometric.nn import GCNConv
from graphUtils import add_edges

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

        self.radius = 0.05
        self.acceleration_scale = 0.01
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
        positions = torch.clamp(positions, -self.max_pos, self.max_pos)
        return positions

    def update(self, graph):
        '''Update the graph a single time step'''
        #dynamically add all the edges...
        graph = add_edges(graph, self.radius, self.device)

        #convolve to update the acceleration... + check if legal
        acceleration = self.convolve(graph) * self.acceleration_scale

        #compute the velocity and position + check if legal
        velocity = self.update_velocity(graph, acceleration)
        positions = self.update_positions(graph, velocity)

        graph.x[:, 2:4] = velocity
        graph.x[:, :2] = positions

        #TODO check for whether food can be consumed
        graph = graph.to(device=self.device)
        return graph

    def forward(self, graph, time_steps = 1):
        '''update the graph n times for n time steps'''
        #optionally compute losses

        for i in range(time_steps):
            graph = self.update(graph)
        return graph