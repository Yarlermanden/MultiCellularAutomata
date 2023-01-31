from cell import Cell
from typing import List
from torch_geometric.data import Data
import torch

class Organism():
    def __init__(self, cells: List[Cell]):
        self.cells = cells

    def toGraph(self):
        '''transforms all cells in organism to nodes in a graph'''

        #each node consists of: posX, posY, velX, velY, isCell(1)
        x = torch.tensor([[cell.pos[0], cell.pos[1], cell.vel[0], cell.vel[1], 1] for cell in self.cells])
        edges = torch.tensor([[]])
        graph = Data(x=x, edge_index=edges)

        return graph

    def fromGraph(self, graph):
        #transforms all nodes in graph to cells
        ...

