import torch
import torch.nn as nn
import time
from torch_geometric.utils import to_networkx
import networkx as nx
from torch_geometric.data import Data

from graphUtils import *
from datastructure import DataStructure
from enums import *

class GNCA(nn.Module):
    def __init__(self, settings):
        super(GNCA, self).__init__()
        self.settings = settings
        self.edge_dim=4
        self.input_channels = 8
        self.output_channels = 7
        self.hidden_size = self.input_channels*1

        self.device = settings.device
        self.model_type = settings.model_type

        self.acceleration_scale = 0.02
        self.max_velocity = 0.02
        self.max_pos = 1

        self.node_indices_to_keep = None
        self.datastructure = DataStructure(settings)

    def message_pass(self, graph):
        '''Convolves the graph for message passing'''
        ...

    def add_noise(self, graph, c_mask):
        x_noise = (torch.rand(graph.x[:, 2].shape, device=self.device)*2-1.0) * self.settings.noise
        y_noise = (torch.rand(graph.x[:, 3].shape, device=self.device)*2-1.0) * self.settings.noise
        update_mask = torch.rand_like(x_noise, device=self.device) > 0.5
        graph.x[:, 2] += x_noise * c_mask * update_mask
        graph.x[:, 3] += y_noise * c_mask * update_mask

    def update_graph(self, graph):
        '''Updates the graph using convolution to compute acceleration and update velocity and positions'''
        c_mask = cell_mask(graph.x)
        moveable_mask = torch.bitwise_or(c_mask, graph.x[:,4] == NodeType.GlobalCell)
        
        h = self.message_pass(graph) * moveable_mask.view(-1,1)
        acceleration = h[:, :2] * self.acceleration_scale
        graph.x[:, 6:] = h[:, 2:]
        velocity = update_velocity(graph, acceleration, self.max_velocity, moveable_mask)
        positions = update_positions(graph, velocity, self.settings.wrap_around, moveable_mask, self.settings.scale)
        graph.x[moveable_mask, 2:4] = velocity[moveable_mask]
        graph.x[moveable_mask, :2] = positions
        graph.x[c_mask, 5] -= 1 #Energy cost
        #TODO decrease energy of cells for each time step
            #could consider decreasing energy more slowly when not moving and depending on size of subgraph...    
            #cost x amount of energy for being an individual organism - decrease depending on subgraph
        self.add_noise(graph, c_mask)
        graph.velocity += velocity.abs().mean()

    def remove_nodes(self, graph):
        '''Removes dead cells and consumed food nodes from the graph. Most be called after update_graph and as late as possible'''
        consumed_mask = get_consume_food_mask(graph, self.settings.consume_radius, self.settings.consumption_edge_required)

        #dead_cells_mask = get_island_cells_mask(graph, self.edges_to_stay_alive)
        dead_cells_mask = get_dead_cells_mask(graph, 0)

        remove_mask = torch.bitwise_or(dead_cells_mask, consumed_mask)
        node_indices_to_keep = torch.nonzero(remove_mask.bitwise_not()).flatten()
        self.node_indices_to_keep = node_indices_to_keep

        if torch.any(consumed_mask):
            edges = graph.edge_index[:, graph.edge_attr[:, 3] != 2]
            graph1 = Data(x=graph.x, edge_index=edges)
            G = to_networkx(graph1, to_undirected=False)
            food_in_nx = torch.nonzero(consumed_mask).flatten()
            for x in food_in_nx:
                des = nx.descendants(G, x.item())
                if len(des) > 0:
                    graph.x[list(des), 5] += 2 #all of their energy should be increased -  #TODO could even adjust this to be higher depending on the food energy size...
            #could it possibly be faster to make the entire subgraphs as they are supported to 
            # and then from there create sets of each subgraph
            # then we check which subgraph a node belongs to and easily index on entire subgraph
        graph.x[:, 5] = torch.clamp(graph.x[:, 5], max=10)

        start_index = 0
        for i in range(self.settings.batch_size):
            end_index = start_index + graph.subsize[i]
            graph.subsize[i] -= remove_mask[start_index:end_index].sum()
            graph.dead_cost[i] += dead_cells_mask[start_index:end_index].sum()
            food_val = graph.x[torch.nonzero(consumed_mask[start_index:end_index])+start_index, 2].sum()
            graph.food_reward[i] += food_val
            start_index = end_index

        graph.x = graph.x[node_indices_to_keep].view(node_indices_to_keep.shape[0], graph.x.shape[1])

    def compute_fitness_metrics(self, graph):
        '''Computes the fitness metrics of each batch used for evaluating the network'''
        #compute_border_cost(graph)

        #visible_food = (graph.edge_attr[:, 3] == 0).sum()
        #count_visible_food = len(torch.nonzero(graph.edge_attr[:, 3] == 0).flatten())
        #graph.food_avg_degree += visible_food/count_visible_food
        #graph.food_avg_dist += graph.edge_attr[torch.nonzero(graph.edge_attr[:, 3] == 0).flatten(), 0].sum()
        #graph.visible_food += visible_food

        s_idx = 0
        for i in range(self.settings.batch_size):
            e_idx = s_idx + graph.subsize[i]
            food_nodes_in_batch = torch.nonzero(food_mask(graph.x[s_idx:e_idx])) + s_idx
            food_edges_in_batch = torch.nonzero(torch.isin(graph.edge_index[0], food_nodes_in_batch)).view(-1)
            edge_attr = graph.edge_attr[food_edges_in_batch]
            nodes = graph.x[graph.edge_index[1, food_edges_in_batch]]
            dist = torch.abs(edge_attr[:, 1:3])
            dist_and_movement = torch.abs(edge_attr[:, 1:3] + nodes[:, 2:4])
            x5 = ((dist-dist_and_movement) * nodes[:,4].view(-1,1)).mean() #positive is good and negative is bad
            if x5.isnan(): x5 = -0.00001
            graph.food_search_movement += x5
            if len(food_nodes_in_batch) == 0 and (cell_mask(graph.x[s_idx:e_idx])).sum() != 0: #no more food but still cells in batch
                graph.food_reward[i] += 1
            s_idx = e_idx

    def update(self, graph):
        '''Update the graph a single time step'''
        time1 = time.perf_counter()
        any_edges = False
        if self.model_type == ModelType.WithGlobalNode: any_edges = self.datastructure.add_edges_with_global_node(graph)
        else: any_edges = self.datastructure.add_edges(graph)
        if not any_edges:
            return graph
        time2 = time.perf_counter()
        self.update_graph(graph)
        time3 = time.perf_counter()
        self.compute_fitness_metrics(graph)
        time4 = time.perf_counter()
        self.remove_nodes(graph)
        time5 = time.perf_counter()
        #TODO add a new cell node pr x graph energy

        #print('dynamic edges: ', time2-time1)
        #print('update graph: ', time3-time2)
        #print('metrics: ', time4-time3)
        #print('remove nodes: ', time5-time4)
        graph = graph.to(device=self.device)
        return graph

    def forward(self, graph, time_steps = 1):
        '''update the graph n times for n time steps'''
        graph = graph.to(device=self.device)
        for _ in range(time_steps):
            if not cell_mask(graph.x).any():
                break
            graph = self.update(graph)
        return graph