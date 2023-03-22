from cell import Cell
from typing import List
import torch
from torch_geometric.data import Data
from enums import EnvironmentType

def set_default_metrics(graph):
    graph.energy = 0.0
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
    def __init__(self, cells: List[Cell], device, with_global_node: bool, food_amount: int, env_type: EnvironmentType):
        self.cells = cells
        self.device = device
        self.with_global_node = with_global_node
        self.food_amount = food_amount
        self.env_type = env_type

    def toGraph(self):
        from graphUtils import add_random_food, add_global_node, add_clusters_of_food
        '''transforms all cells in organism to nodes in a graph'''
        x = torch.tensor([[cell.pos[0], cell.pos[1], cell.vel[0], cell.vel[1], 1] for cell in self.cells], device=self.device)
        edges = torch.tensor([[]], device=self.device)
        graph = Data(x=x, edge_index=edges, device=self.device, subsize=len(x))
        if self.env_type == EnvironmentType.Clusters: 
            add_clusters_of_food(graph, self.device, n=20, cluster_size=10, std_dev=0.05)
        else: add_random_food(graph, self.device, self.food_amount)

        #TODO could consider implementing the entire global node as a virtual node
        if self.with_global_node: add_global_node(graph, self.device)

        set_default_metrics(graph)
        return graph