import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch_geometric
from torch_geometric.nn import GCNConv, EdgeConv, NNConv, GATConv
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
        self.consume_radius = self.radius*2
        self.acceleration_scale = 0.01
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

        #self.mlp2 = Mlp(self.input_channels, 2)
        #self.mlp2 = nn.Conv2d(self.input_channels, 2, 1, padding=0)

        self.mlp = nn.Sequential(
            nn.Linear(self.input_channels, self.input_channels),
            nn.ReLU(), 
            nn.Linear(self.input_channels, 2),
            nn.ReLU())
        #self.mlp1 = nn.Conv1d(self.input_channels, self.input_channels, 1, padding=0)
        #self.mlp2 = nn.Conv1d(self.input_channels, 2, 1, padding=0)

        #self.mlp = Mlp(self.input_channels, self.output_channels)
        #self.conv_layers = NNConv(self.input_channels, self.output_channels, self.mlp)
        #self.mlp = nn.Sequential(nn.Linear(2, 4), nn.ReLU(), nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, channels * self.output_channels))

        #self.mlp = nn.Sequential(nn.Linear(2, 4), nn.Tanh(), nn.Linear(4, channels * self.output_channels))
        #self.conv_layers = NNConv(channels, self.output_channels, self.mlp, aggr='add')
        self.conv_layers = GATConv(channels, self.output_channels, heads=1, edge_dim=2)
        #self.conv_layers2 = GATConv(channels, self.output_channels, heads=1, dropout=0.02)
        #self.mlp_before = nn.Linear(channels, channels*4)

        #self.mlp4 = nn.Sequential(nn.Linear(2, 4), nn.Tanh(), nn.Linear(4, channels * self.output_channels))
        #self.conv_layers2 = NNConv(channels, self.output_channels, self.mlp4, aggr='add')

    def convolve(self, graph):
        '''Convolves the graph for message passing'''
        #h = self.conv_layers(graph.x, graph.edge_index)

        #h = self.mlp_before(graph.x)
        h = self.conv_layers(x=graph.x, edge_index=graph.edge_index, edge_attr=graph.edge_attr)
        #h = self.conv_layers2(x=h, edge_index=graph.edge_index, edge_attr=graph.edge_attr)

        h = self.mlp(h)
        h = h*2 - 1 #forces acceleration to be between -1 and 1 while using ReLU instead of Tanh
        #h = self.mlp1(h.permute(1,0))
        #h = self.tanh(h)
        #h = self.mlp2(h)
        #h = h.permute(1,0)
        #h = self.tanh(h) #force the acceleration between -1 and 1
        return h

    def update_velocity(self, graph, acceleration):
        '''Updates the velocity of the nodes given the acceleration and previous velocity'''
        velocity = graph.x[:, 2:4] + acceleration
        velocity = torch.clamp(velocity, -self.max_velocity, self.max_velocity)
        return velocity

    def update_positions(self, graph, velocity):
        '''Updates the position of the nodes given the velocity and previous positions'''
        positions = graph.x[:, :2] + velocity
        return positions

    def mask_food(self, graph):
        '''Used to get mask to only update all cell nodes - aka not food sources'''
        mask = graph.x[:, 4] == 1
        return mask.to(torch.float)

    def get_consume_food_mask(self, graph):
        '''Consumes food if criteria is met and returns reward'''
        food_mask = graph.x[:, 4] == 0
        edge_below_distance = torch.nonzero(graph.edge_attr[:, 0] < self.consume_radius).flatten()
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

    def update_graph(self, graph):
        '''Updates the graph using convolution to compute acceleration and update velocity and positions'''
        food_mask = self.mask_food(graph)
        acceleration = self.convolve(graph) * self.acceleration_scale * torch.stack((food_mask, food_mask), dim=1)
        velocity = self.update_velocity(graph, acceleration)
        #velocity = self.convolve(graph) * 0.1 * torch.stack((food_mask, food_mask), dim=1)
        #velocity = torch.clamp(velocity, -self.max_velocity, self.max_velocity)
        positions = self.update_positions(graph, velocity)
        graph.x[:, 2:4] = velocity
        graph.x[:, :2] = positions
        return velocity

    def remove_nodes(self, graph):
        '''Removes dead cells and consumed food nodes from the graph. Most be called after update_graph and as late as possible'''
        dead_cells_mask = self.get_island_cells_mask(graph)
        consumed_mask = self.get_consume_food_mask(graph)
        remove_mask = torch.bitwise_or(dead_cells_mask, consumed_mask)
        node_indices_to_keep = torch.nonzero(remove_mask.bitwise_not()).flatten()
        graph.x = graph.x[node_indices_to_keep].view(node_indices_to_keep.shape[0], graph.x.shape[1])
        return dead_cells_mask.sum(), consumed_mask.sum()

    def update(self, graph):
        '''Update the graph a single time step'''
        any_edges = add_edges(graph, self.radius, self.device) #dynamically add edges
        if not any_edges:
            return graph, 0, 0, 0, 0, False

        velocity = self.update_graph(graph)

        epsilon = 0.000001 #Used to prevent taking log of 0.0 resulting in nan values
        mask = graph.x[:, :2].abs() > 1
        border_costX = (graph.x[:, 0].abs()+epsilon).log() * mask[:,0].to(torch.float)
        border_costY = (graph.x[:, 1].abs()+epsilon).log() * mask[:,1].to(torch.float)
        border_cost = (border_costX.sum() + border_costY.sum())
        
        #velocity *= mask[:, :2]

        dead_cost, food_reward = self.remove_nodes(graph)
        graph.attr[0] += food_reward
        #TODO add a new cell node pr x graph energy

        graph = graph.to(device=self.device)
        return graph, velocity.abs().mean(dim=0), border_cost, food_reward, dead_cost, True

    def forward(self, graph, time_steps = 1):
        '''update the graph n times for n time steps'''
        velocity_bonus = torch.tensor([0.0,0.0], device=self.device)
        border_costs, food_rewards, dead_costs = 0, 0, 0

        add_random_food(graph, self.device, 100)

        for i in range(time_steps):
            graph, velocity, border_cost, food_reward, dead_cost, viable = self.update(graph)
            if not viable: 
                velocity_bonus, border_costs, food_reward, dead_costs = torch.tensor(0), 100, 0, 100
                break
            velocity_bonus += velocity
            border_costs += border_cost
            food_rewards += food_reward
            dead_costs += dead_cost

        return graph, velocity_bonus, border_costs, food_rewards, dead_costs