import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch_geometric
from torch_geometric.nn import GCNConv
from graphUtils import add_edges, add_random_food, consume_food

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

    def consume_food_if_possible(self, graph):
        '''Consumes food if criteria is met and returns reward'''
        #consume food if possible
        food_reward = 0
        food_indices = torch.nonzero(graph.x[:, 4] == 0).flatten()
        edges_pr_node = torch.bincount(graph.edge_index[0], minlength=graph.x.shape[0])

        #for (index, count) in [foodIndices, edges_pr_node[foodIndices]]:
        for index in food_indices:
            count = edges_pr_node[index]
            if count > 4:
                consume_food(graph.x[index])
                food_reward += 1
        return food_reward

    def remove_island_cells(self, graph):
        '''Remove cells without any edges'''
        cell_indices = torch.nonzero(graph.x[:, 4] == 1).flatten()
        edges_pr_node = torch.bincount(graph.edge_index[0], minlength=graph.x.shape[0])
        print('cell_indices: ', cell_indices)
        print('edges_pr_node: ', edges_pr_node)

        #cell_indices_without_edge = edges_pr_node[
        nodes_indices_without_edges = torch.nonzero(edges_pr_node < 5)
        print('nodes_indices_without_edges: ', nodes_indices_without_edges)

        mask = torch.zeros(size=(2, graph.x.shape[0]), dtype=torch.bool)
        mask[0, cell_indices] = 1
        mask[1, nodes_indices_without_edges] = 1
        nodes_to_remove = torch.bitwise_and(mask[0], mask[1])
        #nodes_to_remove = cell_indices.__and__(nodes_indices_without_edges)

        #nodes_to_be_removed = cell_indices[edges_pr_node[cell_indices] == 0]
        print('nodes_to_be_removed: ', nodes_to_remove)

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

        #self.remove_island_cells(graph)

        food_reward = self.consume_food_if_possible(graph)

        graph = graph.to(device=self.device)
        return graph, velocity.abs().mean(dim=0), positions.abs().mean(dim=0), border_cost, food_reward

    def forward(self, graph, time_steps = 1):
        '''update the graph n times for n time steps'''
        #optionally compute losses
        velocity_bonus = torch.tensor([0.0,0.0], device=self.device)
        position_penalty = torch.tensor([0.0,0.0], device=self.device)
        border_costs = 0
        food_rewards = 0

        #add_random_food(graph, 2)

        for i in range(time_steps):
            if i % 30 == 0:
                add_random_food(graph)
            graph, velocity, position, border_cost, food_reward = self.update(graph)
            velocity_bonus += velocity
            position_penalty += position
            border_costs += border_cost
            food_rewards += food_reward

        return graph, velocity_bonus, position_penalty, border_costs, food_rewards