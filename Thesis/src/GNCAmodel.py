import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch_geometric
from torch_geometric.nn import GCNConv, EdgeConv, NNConv, GATConv, GATv2Conv

from graphUtils import add_edges, add_random_food, update_velocity, update_positions, food_mask, cell_mask, get_consume_food_mask, get_island_cells_mask

class GNCA(nn.Module):
    def __init__(self, device, batch_size, wrap_around, channels=10, edge_dim=4):
        #batching?
        super(GNCA, self).__init__()
        self.device = device
        self.input_channels = channels-2
        self.edge_dim=edge_dim
        self.output_channels = 7
        self.batch_size = batch_size

        self.radius = 0.05
        self.consume_radius = self.radius
        self.acceleration_scale = 0.005
        self.max_velocity = 0.02
        self.max_pos = 1
        self.consumption_edge_required = 1
        self.edges_to_stay_alive = 2 #1 more than its self loop
        self.energy_required = 5
        self.node_indices_to_keep = None
        self.wrap_around = wrap_around

    def message_pass(self, graph):
        '''Convolves the graph for message passing'''
        ...

    def add_noise(self, graph, c_mask):
        noise = 0.005
        x_noise = (torch.rand(graph.x[:, 2].shape)*2-1.0) * noise
        y_noise = (torch.rand(graph.x[:, 3].shape)*2-1.0) * noise
        update_mask = torch.rand_like(x_noise) > 0.5
        graph.x[:, 2] += x_noise * c_mask * update_mask
        graph.x[:, 3] += y_noise * c_mask * update_mask

    def update_graph(self, graph):
        '''Updates the graph using convolution to compute acceleration and update velocity and positions'''
        c_mask = cell_mask(graph)
        h = self.message_pass(graph) * c_mask.view(c_mask.shape[0], 1)
        acceleration = h[:, :2] * self.acceleration_scale
        #graph.x[:, 5:7] = h[:, 2:]
        velocity = update_velocity(graph, acceleration, self.max_velocity, c_mask)
        positions = update_positions(graph, velocity, self.wrap_around, c_mask)
        graph.x[:, 2:4] = velocity
        graph.x[c_mask, :2] = positions
        self.add_noise(graph, c_mask)
        graph.velocity += velocity.abs().mean()

    def remove_nodes(self, graph):
        '''Removes dead cells and consumed food nodes from the graph. Most be called after update_graph and as late as possible'''
        dead_cells_mask = get_island_cells_mask(graph, self.edges_to_stay_alive)
        consumed_mask = get_consume_food_mask(graph, self.consume_radius, self.consumption_edge_required)
        food_val = graph.x[torch.nonzero(consumed_mask).flatten(), 2].sum()
        remove_mask = torch.bitwise_or(dead_cells_mask, consumed_mask)
        node_indices_to_keep = torch.nonzero(remove_mask.bitwise_not()).flatten()
        self.node_indices_to_keep = node_indices_to_keep

        start_index = 0
        for i in range(self.batch_size):
            end_index = start_index + graph.subsize[i]
            graph.subsize[i] -= remove_mask[start_index:end_index].sum()
            start_index = end_index+1

        #TODO split the amounts into each batch and add/adjust accordingly
        graph.x = graph.x[node_indices_to_keep].view(node_indices_to_keep.shape[0], graph.x.shape[1])
        graph.dead_cost += dead_cells_mask.sum()
        graph.food_reward += food_val
        graph.energy += food_val

    def update(self, graph):
        '''Update the graph a single time step'''
        any_edges = add_edges(graph, self.radius, self.device, self.wrap_around, self.batch_size) #dynamically add edges

        self.update_graph(graph)

        epsilon = 0.000001 #Used to prevent taking log of 0.0 resulting in nan values
        mask = graph.x[:, :2].abs() > 1
        border_costX = (graph.x[:, 0].abs()+epsilon).log() * mask[:,0].to(torch.float)
        border_costY = (graph.x[:, 1].abs()+epsilon).log() * mask[:,1].to(torch.float)
        graph.border_cost += (border_costX.sum() + border_costY.sum())

        #Compute the number of food sources connected to the graph
        visible_food = (graph.edge_attr[:, 3] == 0).sum()
        count_visible_food = len(torch.nonzero(graph.edge_attr[:, 3] == 0).flatten())
        graph.food_avg_degree += visible_food/count_visible_food

        graph.food_avg_dist += graph.edge_attr[torch.nonzero(graph.edge_attr[:, 3] == 0).flatten(), 0].sum()
        graph.visible_food += visible_food
        
        self.remove_nodes(graph)
        #TODO add a new cell node pr x graph energy

        graph = graph.to(device=self.device)
        return graph

    def forward(self, graph, time_steps = 1):
        '''update the graph n times for n time steps'''
        for _ in range(time_steps):
            if not cell_mask(graph).any():
                break
            graph = self.update(graph)
        return graph