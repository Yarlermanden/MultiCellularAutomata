from cell import Cell
from typing import List
import torch
from torch_geometric.data import Data
from enums import *

def set_default_metrics(graph):
    graph.velocity = 0.0
    graph.border_cost = 0.0
    graph.food_reward = 0.0
    graph.dead_cost = 0.0
    graph.visible_food = 0.0
    graph.food_avg_dist = 0.0
    graph.food_avg_degree = 0.0
    graph.food_search_movement = 0.0
    graph.subsize=len(graph.x)

def toGraph(nodes, device):
    '''transforms an environment with given nodes into a graph'''
    edges = torch.tensor([[]], device=device)
    graph = Data(x=nodes, edge_index=edges, device=device, subsize=len(nodes))
    set_default_metrics(graph)
    return graph

class Organism():
    def __init__(self, cells: List[Cell], settings):
        self.cells = cells
        self.device = settings.device
        self.model_type = settings.model_type
        self.food_amount = settings.food_amount
        self.env_type = settings.env_type
        self.scale = settings.scale
        self.clusters = settings.clusters
        self.cluster_size = settings.cluster_size

    def toGraph(self):
        from graphUtils import add_random_food, add_global_node, add_clusters_of_food
        '''transforms all cells in organism to nodes in a graph'''
        hidden = [0, 0, 0, 0, 0]
        x = torch.tensor([[cell.pos[0], cell.pos[1], cell.vel[0], cell.vel[1], 1, 20, *hidden] for cell in self.cells], device=self.device)
        edges = torch.tensor([[]], device=self.device)
        graph = Data(x=x, edge_index=edges, device=self.device, subsize=len(x))
        if self.env_type == EnvironmentType.Clusters: 
            add_clusters_of_food(graph, self.device, n=self.clusters, cluster_size=self.cluster_size, std_dev=0.04, scale=self.scale)
        else: add_random_food(graph, self.device, self.food_amount, self.scale)

        #TODO could consider implementing the entire global node as a virtual node
        if self.model_type == ModelType.WithGlobalNode: add_global_node(graph, self.device)

        set_default_metrics(graph)
        return graph