import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch_geometric
from torch_geometric.nn import GCNConv, EdgeConv, NNConv, GATConv, GATv2Conv

from graphUtils import add_edges, add_random_food, update_velocity, update_positions, food_mask, cell_mask, get_consume_food_mask, get_island_cells_mask, compute_border_cost

class GNCA(nn.Module):
    def __init__(self, device, batch_size, wrap_around, channels=10, edge_dim=4):
        super(GNCA, self).__init__()
        self.device = device
        self.input_channels = channels-2
        self.edge_dim=edge_dim
        self.output_channels = 7
        self.batch_size = batch_size

        self.radius = 0.04
        self.consume_radius = self.radius
        #self.acceleration_scale = 0.005
        self.acceleration_scale = 0.02
        self.max_velocity = 0.02
        self.max_pos = 1
        self.consumption_edge_required = 1
        self.edges_to_stay_alive = 3 #1 more than its self loop
        self.energy_required = 5
        self.node_indices_to_keep = None
        self.wrap_around = wrap_around

    def message_pass(self, graph):
        '''Convolves the graph for message passing'''
        ...

    def add_noise(self, graph, c_mask):
        noise = 0.005
        x_noise = (torch.rand(graph.x[:, 2].shape, device=self.device)*2-1.0) * noise
        y_noise = (torch.rand(graph.x[:, 3].shape, device=self.device)*2-1.0) * noise
        update_mask = torch.rand_like(x_noise, device=self.device) > 0.5
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
        graph.x[c_mask, 2:4] = velocity[c_mask]
        graph.x[c_mask, :2] = positions
        #self.add_noise(graph, c_mask)
        graph.velocity += velocity.abs().mean()

    def remove_nodes(self, graph):
        '''Removes dead cells and consumed food nodes from the graph. Most be called after update_graph and as late as possible'''
        dead_cells_mask = get_island_cells_mask(graph, self.edges_to_stay_alive)
        consumed_mask = get_consume_food_mask(graph, self.consume_radius, self.consumption_edge_required)
        #food_val = graph.x[torch.nonzero(consumed_mask).flatten(), 2].sum()
        remove_mask = torch.bitwise_or(dead_cells_mask, consumed_mask)
        node_indices_to_keep = torch.nonzero(remove_mask.bitwise_not()).flatten()
        self.node_indices_to_keep = node_indices_to_keep

        start_index = 0
        for i in range(self.batch_size):
            end_index = start_index + graph.subsize[i]
            graph.subsize[i] -= remove_mask[start_index:end_index].sum()
            graph.dead_cost[i] += dead_cells_mask[start_index:end_index].sum()
            #food_val = graph.x[torch.nonzero(consumed_mask[start_index:end_index]), 2].sum()
            food_val = graph.x[torch.nonzero(consumed_mask[start_index:end_index])+start_index, 2].sum()
            graph.food_reward[i] += food_val
            graph.energy[i] += food_val
            start_index = end_index

        graph.x = graph.x[node_indices_to_keep].view(node_indices_to_keep.shape[0], graph.x.shape[1])

    def compute_fitness_metrics(self, graph):
        #compute_border_cost(graph)

        #Compute the number of food sources connected to the graph
        visible_food = (graph.edge_attr[:, 3] == 0).sum()
        count_visible_food = len(torch.nonzero(graph.edge_attr[:, 3] == 0).flatten())
        graph.food_avg_degree += visible_food/count_visible_food

        graph.food_avg_dist += graph.edge_attr[torch.nonzero(graph.edge_attr[:, 3] == 0).flatten(), 0].sum()
        graph.visible_food += visible_food

        s_idx = 0
        for i in range(self.batch_size):
            e_idx = s_idx + graph.subsize[i]
            food_nodes_in_batch = torch.nonzero(graph.x[s_idx:e_idx, 4] == 0) + s_idx
            food_edges_in_batch = torch.nonzero(torch.isin(graph.edge_index[1], food_nodes_in_batch)).view(-1) #only edges going from cell to food
            edge_attr = graph.edge_attr[food_edges_in_batch]
            nodes = graph.x[graph.edge_index[0, food_edges_in_batch]]
            x1 = torch.abs(edge_attr[:, 1:3])
            x2 = torch.abs(edge_attr[:, 1:3] + nodes[:, 2:4])
            x3 = x1-x2
            x4 = x3 * nodes[:,4].view(nodes.shape[0], 1) #view(-1, 1)
            x5 = x4.mean()
            if x5.isnan():
                #x5 = -0.02
                x5 = -0.00001
            graph.food_search_movement += x5
            #graph.food_search_movement += ((torch.abs(edge_attr[:, 1:3]) - torch.abs(edge_attr[:, 1:3] - nodes[:, 2:4])) * nodes[:, 4]).mean() #positive is good and negative is bad - food nodes result in 0
            s_idx = e_idx


    def update(self, graph):
        '''Update the graph a single time step'''
        add_edges(graph, self.radius, self.device, self.wrap_around, self.batch_size) #dynamically add edges
        self.update_graph(graph)
        self.compute_fitness_metrics(graph)
        self.remove_nodes(graph)
        #TODO add a new cell node pr x graph energy

        graph = graph.to(device=self.device)
        return graph

    def forward(self, graph, time_steps = 1):
        '''update the graph n times for n time steps'''
        graph = graph.to(device=self.device)
        for _ in range(time_steps):
            if not cell_mask(graph).any():
                break
            graph = self.update(graph)
        return graph