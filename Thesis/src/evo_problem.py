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

        fitness3 = graph.food_search_movement.mean() * 50

        food_reward = graph.food_reward.mean()
        fitness1 = food_reward

        cells = graph.x[cell_mask(graph.x)]
        fitness2 = cells[:, 5].sum() / alive_start * 10

        fitness = fitness3 + fitness1 + fitness2

        if torch.any(torch.isnan(food_reward)):
            print('fitness is nan')
            fitness = 0

        ray.get(self.global_var.update_pool.remote(graph))
        return torch.tensor([fitness, fitness1, fitness2, fitness3])