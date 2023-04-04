from evotorch.neuroevolution import NEProblem
import torch
import ray
from torch_geometric.utils import to_networkx
import networkx as nx
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from evotorch.decorators import vectorized, on_aux_device

class Custom_NEProblem(NEProblem):
    #def __init__(self, n, global_var, batch_size, **kwargs):
    def __init__(self, settings, global_var, **kwargs):
        super(Custom_NEProblem, self).__init__(**kwargs)
        self.global_var = global_var
        self.settings = settings

    @vectorized
    @on_aux_device
    def _evaluate_network(self, network: torch.nn.Module):
        steps, graphs = ray.get(self.global_var.get_global_var.remote())
        loader = DataLoader(graphs, batch_size=self.settings.batch_size) #TODO move this part into the global_state to not do multiple times
        batch = next(iter(loader))
        alive_start = (batch.x[:, 4] == 1).sum()

        with torch.no_grad():
            graph = network(batch, steps)

        food_reward = graph.food_reward.mean()
        fitness = food_reward

        cells = graph.x[graph.x[:, 4] == 1]        
        fitness = cells[:, 5].sum() / alive_start

        if torch.any(torch.isnan(food_reward)):
            print('fitness is nan')
            fitness = 0

        ray.get(self.global_var.update_pool.remote(graph))
        return torch.tensor([fitness, fitness])