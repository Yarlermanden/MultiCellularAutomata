import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch_geometric
from torch_geometric.nn import GCNConv, EdgeConv, NNConv
from graphUtils import add_edges, add_random_food, consume_food

class Mlp(nn.Module):
    def __init__(self, input: int, output: int):
        super(Mlp, self).__init__()
        self.mlp1 = nn.Linear(input, input)
        #self.mlp2 = nn.Linear(input, input)
        self.mlp3 = nn.Linear(input, output)
    
    def forward(self, x):
        x = torch.relu(self.mlp1(x))
        #x = torch.relu(self.mlp2(x))
        x = self.mlp3(x)
        return x

class GNCA(nn.Module):
    def __init__(self, device, channels=5):
        #batching?
        super(GNCA, self).__init__()
        self.device = device

        self.radius = 0.05
        self.acceleration_scale = 0.4
        self.max_velocity = 0.1
        self.max_pos = 1
        self.consumption_edge_required = 3
        self.edges_to_stay_alive = 1
        self.energy_required = 5

        #self.conv_layers = GCNConv(in_channels=channels, out_channels=2)
        self.input_channels = channels
        self.output_channels = channels
        #self.mlp = Mlp(self.input_channels*2, self.output_channels)
        #self.conv_layers = EdgeConv(self.mlp)

        self.mlp2 = Mlp(self.input_channels, 2)
        #self.mlp = Mlp(self.input_channels, self.output_channels)
        #self.conv_layers = NNConv(self.input_channels, self.output_channels, self.mlp)
        self.mlp = nn.Sequential(nn.Linear(2, 2), nn.ReLU(), nn.Linear(2, channels * self.output_channels))
        self.conv_layers = NNConv(channels, self.output_channels, self.mlp, aggr='add')

    def convolve(self, graph):
        '''Convolves the graph for message passing'''
        #h = self.conv_layers(graph.x, graph.edge_index)
        h = self.conv_layers(x=graph.x, edge_index=graph.edge_index, edge_attr=graph.edge_attr)
        h = self.mlp2(h)
        return h

    def update_velocity(self, graph, acceleration):
        velocity = graph.x[:, 2:4] + acceleration #update velocity
        velocity = torch.clamp(velocity, -self.max_velocity, self.max_velocity)
        return velocity

    def update_positions(self, graph, velocity):
        positions = graph.x[:, :2] + velocity #update position
        return positions

    def mask_food(self, graph):
        '''Used to get mask to only update all cell nodes - aka not food sources'''
        mask = graph.x[:, 4] == 1
        return mask.to(torch.float)

    def get_consume_food_mask(self, graph):
        '''Consumes food if criteria is met and returns reward'''
        food_mask = graph.x[:, 4] == 0
        edge_below_distance = torch.nonzero(graph.edge_attr[:, 0] < self.radius).flatten()
        edges_pr_node = torch.bincount(graph.edge_index[0, edge_below_distance], minlength=graph.x.shape[0])
        edge_mask = edges_pr_node >= self.consumption_edge_required
        consumption_mask = torch.bitwise_and(food_mask, edge_mask)
        return consumption_mask

    def get_island_cells_mask(self, graph):
        '''Remove cells without any edges'''
        cell_mask = graph.x[:, 4] == 1
        cell_edge_indices = torch.nonzero(graph.edge_attr[:, 1] == 1).flatten()
        zero_edge_mask = torch.bincount(graph.edge_index[0, cell_edge_indices], minlength=graph.x.shape[0]) < self.edges_to_stay_alive
        mask = torch.bitwise_and(cell_mask, zero_edge_mask)
        return mask

    def update(self, graph):
        '''Update the graph a single time step'''
        any_edges = add_edges(graph, self.radius, self.device) #dynamically add edges
        if not any_edges:
            return graph, 0, 0, 0, 0
            
        food_mask = self.mask_food(graph)

        acceleration = self.convolve(graph) * self.acceleration_scale #get acceleration
        acceleration = acceleration * torch.stack((food_mask, food_mask), dim=1)
        velocity = self.update_velocity(graph, acceleration)
        #velocity = self.convolve(graph) * 0.1 * torch.stack((food_mask, food_mask), dim=1)
        positions = self.update_positions(graph, velocity)

        graph.x[:, 2:4] = velocity
        graph.x[:, :2] = positions

        epsilon = 0.000001 #Used to prevent taking log of 0.0 resulting in nan values
        maskX = graph.x[:, 0].abs() > 1
        maskY = graph.x[:, 1].abs() > 1
        border_costX = (graph.x[:, 0].abs()+epsilon).log() * maskX.to(torch.float)
        border_costY = (graph.x[:, 1].abs()+epsilon).log() * maskY.to(torch.float)
        border_cost = (border_costX.sum() + border_costY.sum())

        dead_cells_mask = self.get_island_cells_mask(graph)
        dead_cost = dead_cells_mask.sum()
        consumed_mask = self.get_consume_food_mask(graph)
        food_reward = consumed_mask.sum()

        remove_mask = torch.bitwise_or(dead_cells_mask, consumed_mask)
        node_indices_to_keep = torch.nonzero(remove_mask.bitwise_not()).flatten()
        graph.x = graph.x[node_indices_to_keep].view(node_indices_to_keep.shape[0], graph.x.shape[1])

        graph.attr[0] += food_reward
        #TODO add a new cell node pr x graph energy

        graph = graph.to(device=self.device)
        return graph, velocity.abs().mean(dim=0), border_cost, food_reward, dead_cost

    def forward(self, graph, time_steps = 1):
        '''update the graph n times for n time steps'''
        velocity_bonus = torch.tensor([0.0,0.0], device=self.device)
        border_costs, food_rewards, dead_costs = 0, 0, 0

        add_random_food(graph, self.device, 20)

        for i in range(time_steps):
            if len(graph.x) < 3:
                break
            #if i % 10 == 0:
            #    add_random_food(graph)
            graph, velocity, border_cost, food_reward, dead_cost = self.update(graph)
            velocity_bonus += velocity
            border_costs += border_cost
            food_rewards += food_reward
            dead_costs += dead_cost

        return graph, velocity_bonus, border_costs, food_rewards, dead_costs