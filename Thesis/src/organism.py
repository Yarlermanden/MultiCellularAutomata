from cell import Cell
from typing import List
import torch
from torch_geometric.data import Data

class Organism():
    def __init__(self, cells: List[Cell], device):
        self.cells = cells
        self.device = device

    def toGraph(self):
        from graphUtils import add_random_food
        '''transforms all cells in organism to nodes in a graph'''
        #each node consists of: posX, posY, velX, velY, isCell(1)
        #x = torch.tensor([[cell.pos[0], cell.pos[1], cell.vel[0], cell.vel[1], 1, 0, 0, 0, 0, 0] for cell in self.cells], device=self.device)
        x = torch.tensor([[cell.pos[0], cell.pos[1], cell.vel[0], cell.vel[1], 1] for cell in self.cells], device=self.device)
        edges = torch.tensor([[]], device=self.device)
        graph = Data(x=x, edge_index=edges, device=self.device, subsize=len(x))
        add_random_food(graph, self.device, 50)
        graph.energy = 0.0
        graph.velocity = 0.0
        graph.border_cost = 0.0
        graph.food_reward = 0.0
        graph.dead_cost = 0.0
        graph.visible_food = 0.0
        graph.food_avg_dist = 0.0
        graph.food_avg_degree = 0.0
        graph.subsize=len(graph.x)
        return graph