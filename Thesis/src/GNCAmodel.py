import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch_geometric
from torch_geometric.nn import GCNConv, EdgeConv
from graphUtils import add_edges, add_random_food, consume_food

class Mlp(nn.Module):
    def __init__(self, input: int, output: int):
        super(Mlp, self).__init__()
        self.mlp1 = nn.Linear(input, input)
        self.mlp2 = nn.Linear(input, input)
        self.mlp3 = nn.Linear(input, output)
    
    def forward(self, x):
        x = torch.relu(self.mlp1(x))
        x = torch.relu(self.mlp2(x))
        x = self.mlp3(x)
        return x

class GNCA(nn.Module):
    def __init__(self, device, channels=5):
        #batching?
        super(GNCA, self).__init__()
        self.device = device

        self.radius = 0.05
        self.acceleration_scale = 0.02
        self.max_velocity = 0.1
        self.max_pos = 1
        self.consumption_edge_required = 5
        self.edges_to_stay_alive = 1
        self.energy_required = 4

        #self.conv_layers = GCNConv(in_channels=channels, out_channels=2)
        self.input_channels = channels
        self.output_channels = 2
        self.mlp = Mlp(self.input_channels*2, self.output_channels)
        self.conv_layers = EdgeConv(self.mlp)

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
        
        food_mask = graph.x[:, 4] == 0
        edges_pr_node = torch.bincount(graph.edge_index[0], minlength=graph.x.shape[0])
        edge_mask = edges_pr_node >= self.consumption_edge_required
        if len(edges_pr_node) != len(food_mask):
            print('graph.x shape: ', graph.x.shape)
            print('edge_index shape: ', graph.edge_index.shape)
            print('edge_mask', edge_mask)
            print('food_mask: ', food_mask)
        consumption_mask = torch.bitwise_and(food_mask, edge_mask)

        consumption_indices = torch.nonzero(consumption_mask).flatten()

        index_reduction = 0
        for index in consumption_indices:
            consume_food(graph, index-index_reduction)
            if graph.attr[0] < self.energy_required:
                index_reduction += 1 #we know node will be removed
            food_reward += 1
        return food_reward

    def remove_island_cells(self, graph):
        '''Remove cells without any edges'''
        cell_mask = graph.x[:, 4] == 1
        
        cell_edge_indices = torch.nonzero(graph.edge_attr[:, 1] == 1).flatten()

        zero_edge_mask = torch.bincount(graph.edge_index[0, cell_edge_indices], minlength=graph.x.shape[0]) < self.edges_to_stay_alive
        mask = torch.bitwise_and(cell_mask, zero_edge_mask)
        node_indices_to_keep = torch.nonzero(mask.bitwise_not())

        graph.x = graph.x[node_indices_to_keep].view(node_indices_to_keep.shape[0], graph.x.shape[1])
        nodes_removed = len(cell_mask) - len(graph.x)
        if nodes_removed == 0:
            return nodes_removed

        return nodes_removed

        #TODO need to remove all edges connected to this node - node-cell and cell-cell edges...
        #find indices of all edges, which is connected to these nodes and remove them from both edge_index and edge_attr
        edges_mask1 = torch.isin(graph.edge_index[0], node_indices_to_keep)
        edges_mask2 = torch.isin(graph.edge_index[1], node_indices_to_keep)
        edges_mask = torch.bitwise_or(edges_mask1, edges_mask2)
        #edges1 = graph.edge_index[graph.edge_index[0] == node_indices_to_keep]
        #edges2 = graph.edge_index[graph.edge_index[1] == node_indices_to_keep]

        graph.edge_index = graph.edge_index[:, edges_mask]

        #edge_attr1 = graph.edge_attr[graph.edge_index[0] == node_indices_to_keep]
        #edge_attr2 = graph.edge_attr[graph.edge_index[1] == node_indices_to_keep]
        #graph.edge_attr = torch.concat((edge_attr1, edge_attr2))
        graph.edge_attr = graph.edge_attr[edges_mask]
        return nodes_removed

    def update(self, graph):
        '''Update the graph a single time step'''
        any_edges = add_edges(graph, self.radius, self.device) #dynamically add edges
        if not any_edges:
            return graph, 0, 0, 0, 0, 0
            
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

        dead_cost = self.remove_island_cells(graph)
        #really bad practice
        any_edges = add_edges(graph, self.radius, self.device)
        food_reward = 0
        if any_edges:
            food_reward = self.consume_food_if_possible(graph)
            graph = graph.to(device=self.device)
        return graph, velocity.abs().mean(dim=0), positions.abs().mean(dim=0), border_cost, food_reward, dead_cost

    def forward(self, graph, time_steps = 1):
        '''update the graph n times for n time steps'''
        #optionally compute losses
        velocity_bonus = torch.tensor([0.0,0.0], device=self.device)
        position_penalty = torch.tensor([0.0,0.0], device=self.device)
        border_costs = 0
        food_rewards = 0
        dead_costs = 0

        #add_random_food(graph, 2)

        for i in range(time_steps):
            if len(graph.x) < 3:
                break
            if i % 10 == 0:
                add_random_food(graph)
            graph, velocity, position, border_cost, food_reward, dead_cost = self.update(graph)
            velocity_bonus += velocity
            position_penalty += position
            border_costs += border_cost
            food_rewards += food_reward
            dead_costs += dead_cost

        return graph, velocity_bonus, position_penalty, border_costs, food_rewards, dead_costs