from cell import Cell
from typing import List
from torch_geometric.data import Data
import torch

class Organism():
    def __init__(self, cells: List[Cell], device):
        self.cells = cells
        self.device = device

    def toGraph(self):
        '''transforms all cells in organism to nodes in a graph'''

        #each node consists of: posX, posY, velX, velY, isCell(1)
        x = torch.tensor([[cell.pos[0], cell.pos[1], cell.vel[0], cell.vel[1], 1, 0, 0, 0, 0, 0] for cell in self.cells], device=self.device)
        edges = torch.tensor([[]], device=self.device)
        graph = Data(x=x, edge_index=edges, device=self.device)
        graph.attr = [0] #energy

        return graph

    def fromGraph(self, graph):
        #transforms all nodes in graph to cells
        ...


