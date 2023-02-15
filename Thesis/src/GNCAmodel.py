import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch_geometric
from torch_geometric.nn import GCNConv, EdgeConv, NNConv, GATConv, GATv2Conv

from graphUtils import add_edges, add_random_food, update_velocity, update_positions, food_mask, cell_mask, get_consume_food_mask, get_island_cells_mask

class GNCA(nn.Module):
    def __init__(self, device, channels=7, edge_dim=4):
        #batching?
        super(GNCA, self).__init__()
        self.device = device
        self.input_channels = channels
        self.edge_dim=edge_dim
        self.output_channels = 4

        self.radius = 0.05
        self.consume_radius = self.radius*2
        self.acceleration_scale = 0.005
        self.max_velocity = 0.05
        self.max_pos = 1
        self.consumption_edge_required = 5
        self.edges_to_stay_alive = 3 #1 more than its self loop
        self.energy_required = 5
        self.node_indices_to_keep = None

    def message_pass(self, graph):
        '''Convolves the graph for message passing'''
        ...

    def update_graph(self, graph):
        '''Updates the graph using convolution to compute acceleration and update velocity and positions'''
        c_mask = cell_mask(graph)
        h = self.message_pass(graph) * self.acceleration_scale * torch.stack((c_mask, c_mask, c_mask, c_mask), dim=1)
        acceleration = h[:, :2]
        graph.x[:, 5:] = h[:, 2:]
        velocity = update_velocity(graph, acceleration, self.max_velocity)
        positions = update_positions(graph, velocity)
        graph.x[:, 2:4] = velocity
        graph.x[:, :2] = positions
        return velocity

    def remove_nodes(self, graph):
        '''Removes dead cells and consumed food nodes from the graph. Most be called after update_graph and as late as possible'''
        dead_cells_mask = get_island_cells_mask(graph, self.edges_to_stay_alive)
        consumed_mask = get_consume_food_mask(graph, self.consume_radius, self.consumption_edge_required)
        remove_mask = torch.bitwise_or(dead_cells_mask, consumed_mask)
        node_indices_to_keep = torch.nonzero(remove_mask.bitwise_not()).flatten()
        self.node_indices_to_keep = node_indices_to_keep
        graph.x = graph.x[node_indices_to_keep].view(node_indices_to_keep.shape[0], graph.x.shape[1])
        return dead_cells_mask.sum(), consumed_mask.sum()

    def update(self, graph):
        '''Update the graph a single time step'''
        any_edges = add_edges(graph, self.radius, self.device) #dynamically add edges
        if not any_edges:
            return graph, 0, 0, 0, 0, 0, False

        velocity = self.update_graph(graph)

        epsilon = 0.000001 #Used to prevent taking log of 0.0 resulting in nan values
        mask = graph.x[:, :2].abs() > 1
        border_costX = (graph.x[:, 0].abs()+epsilon).log() * mask[:,0].to(torch.float)
        border_costY = (graph.x[:, 1].abs()+epsilon).log() * mask[:,1].to(torch.float)
        border_cost = (border_costX.sum() + border_costY.sum())

        #Compute the number of food sources connected to the graph
        visible_food = (graph.edge_attr[:, 3] == 0).sum()
        
        dead_cost, food_reward = self.remove_nodes(graph)
        graph.attr[0] += food_reward
        #TODO add a new cell node pr x graph energy

        graph = graph.to(device=self.device)
        return graph, velocity.abs().mean(dim=0), border_cost, food_reward, dead_cost, visible_food, True

    def forward(self, graph, time_steps = 1):
        '''update the graph n times for n time steps'''
        velocity_bonus = torch.tensor([0.0,0.0], device=self.device, dtype=torch.float)
        border_costs, food_rewards, dead_costs, visible_foods = 0.0, 0.0, 0.0, 0.0

        add_random_food(graph, self.device, 30)

        for i in range(time_steps):
            graph, velocity, border_cost, food_reward, dead_cost, visible_food, viable = self.update(graph)
            if not viable: 
                velocity_bonus, border_costs, food_reward, dead_costs = torch.tensor(0.0, dtype=torch.float), 100, 0, 100
                break
            velocity_bonus += velocity
            border_costs += border_cost
            food_rewards += food_reward
            dead_costs += dead_cost
            visible_foods += visible_food

        return graph, velocity_bonus, border_costs, food_rewards, dead_costs, visible_foods