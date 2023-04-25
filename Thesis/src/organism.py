from cell import Cell
from typing import List
import torch
from torch_geometric.data import Data
from enums import *
import random

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
        self.settings = settings
        self.device = settings.device

    def toGraph(self):
        from graphUtils import add_random_food, add_global_node, add_clusters_of_food, add_circular_food, add_spiral_food
        '''transforms all cells in organism to nodes in a graph'''
        hidden = [0, 0, 0, 0, 0]
        x = torch.tensor([[cell.pos[0], cell.pos[1], cell.vel[0], cell.vel[1], 1, 50, *hidden] for cell in self.cells], device=self.device)
        edges = torch.tensor([[]], device=self.device)
        graph = Data(x=x, edge_index=edges, device=self.device, subsize=len(x))

        if self.settings.model_type == ModelType.SmallWorld:
            graph.x[:self.settings.n2, 4] = NodeType.LongRadiusCell
        elif self.settings.model_type == ModelType.WithGlobalNode: add_global_node(graph, self.device)


        random_number = random.randint(0, 3)
        food_env = self.settings.food_envs[random_number]
        
        match food_env.env_type:
            case EnvironmentType.Clusters:
                add_clusters_of_food(graph, self.settings, food_env)
            case EnvironmentType.Circular:
                add_circular_food(graph, self.settings, food_env)
            case EnvironmentType.Spiral:
                add_spiral_food(graph, self.settings, food_env)
            case _:
                add_random_food(graph, self.settings, food_env)

        set_default_metrics(graph)
        return graph