from evotorch.neuroevolution import NEProblem
import torch
import ray
import networkx as nx
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from evotorch.decorators import vectorized, on_aux_device

from graphUtils import *

class Custom_NEProblem(NEProblem):
    def __init__(self, settings, global_var, **kwargs):
        super(Custom_NEProblem, self).__init__(**kwargs)
        self.global_var = global_var
        self.settings = settings

    @vectorized
    @on_aux_device
    def _evaluate_network(self, network: torch.nn.Module):
        steps, graphs = ray.get(self.global_var.get_global_var.remote())
        batch = next(iter(graphs))
        alive_start = cell_mask(batch.x).sum()

        with torch.no_grad():
            graph = network(batch, steps)

        #food_reward = graph.food_reward.mean()
        #fitness1 = food_reward #food consumed - average of batch size

        #cells = graph.x[cell_mask(graph.x)]
        #fitness2 = cells[:, 5].sum() / alive_start * 10 #energy left - for now always 0 as it ends when all cells die
        fitness3 = graph.cells_alive.mean() / self.settings.n #average ratio of cells alive across batch - between 0 and 1 pr timestep
        #fitness = fitness1 + fitness2 + fitness3 
        fitness = fitness3

        #movement = torch.max(graph.velocity.sum() / self.settings.batch_size * 4, 20)
        pos = torch.clamp(graph.pos_reward.mean(), max=10)
        if not torch.isnan(pos):
            fitness += pos

        if torch.any(torch.isnan(fitness)):
            print('fitness is nan')
            fitness = 0
            fitness3 = 0

        ray.get(self.global_var.update_pool.remote(graph))
        return torch.tensor([fitness])