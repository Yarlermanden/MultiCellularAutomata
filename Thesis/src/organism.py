from cell import Cell
from typing import List
import torch
from torch_geometric.data import Data
from enums import *
import random
import numpy as np
import math

def set_default_metrics(graph):
    graph.velocity = 0.0
    graph.pos_reward = 0.0
    graph.border_cost = 0.0
    graph.food_reward = 0.0
    graph.dead_cost = 0.0
    graph.visible_food = 0.0
    graph.food_avg_dist = 0.0
    graph.food_avg_degree = 0.0
    graph.food_search_movement = 0.0
    graph.cells_alive = 0.0
    graph.subsize=len(graph.x)
    graph.timesteps = 0.0

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

    def toGraph(self, food_env = None):
        from graphUtils import add_random_food, add_global_node, add_clusters_of_food, add_circular_food, add_spiral_food, add_labyrinth_food, add_bottleneck_food, add_box_food, add_grid_food, cell_mask, add_food_grid_food
        '''transforms all cells in organism to nodes in a graph'''
        #hidden = [0, 0, 0, 0, 0]
        #hidden = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        #hidden = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        hidden = np.random.rand((10)).tolist()
        x = torch.tensor([[cell.pos[0], cell.pos[1], cell.vel[0], cell.vel[1], 1, 50, *hidden] for cell in self.cells], device=self.device)
        edges = torch.tensor([[]], device=self.device)
        graph = Data(x=x, edge_index=edges, device=self.device, subsize=len(x))

        if self.settings.model_type == ModelType.SmallWorld:
            graph.x[:self.settings.n2, 4] = NodeType.LongRadiusCell
        elif self.settings.model_type == ModelType.WithGlobalNode: add_global_node(graph, self.device)

        if food_env == None:
            random_number = random.randint(0, len(self.settings.food_envs)-1)
            food_env = self.settings.food_envs[random_number]
        
        match food_env.env_type:
            case EnvironmentType.Clusters:
                add_clusters_of_food(graph, self.settings, food_env)
            case EnvironmentType.Circular:
                add_circular_food(graph, self.settings, food_env)
            case EnvironmentType.Spiral:
                add_spiral_food(graph, self.settings, food_env)
            case EnvironmentType.Labyrinth:
                add_labyrinth_food(graph, self.settings, food_env)
            case EnvironmentType.Bottleneck:
                add_bottleneck_food(graph, self.settings, food_env)
            case EnvironmentType.Box:
                add_box_food(graph, self.settings, food_env)
            case EnvironmentType.Grid:
                add_grid_food(graph, self.settings, food_env)
            case EnvironmentType.Food_Grid:
                add_food_grid_food(graph, self.settings, food_env)
            case _:
                add_random_food(graph, self.settings, food_env)

        #rotate environment
        degrees = random.uniform(0, 360)
        theta = math.radians(degrees)
        rotation_matrix = torch.tensor([[math.cos(theta), -math.sin(theta)],
                                        [math.sin(theta), math.cos(theta)]])
        not_cells = torch.bitwise_not(cell_mask(graph.x))
        graph.x[not_cells, :2] = torch.matmul(graph.x[not_cells, :2], rotation_matrix)

        set_default_metrics(graph)
        return graph