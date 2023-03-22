import numpy as np
import ray
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.utils import unbatch

from sample_pool import SamplePool
from generator import generate_organism, sample_from_samplepool
from organism import Organism, toGraph
from graphUtils import unbatch_nodes

@ray.remote
class GlobalState():
    def __init__(self, n, device, batch_size, with_global_node, food_amount, env_type):
        self.n = n
        self.device = device
        self.batch_size = batch_size
        self.with_global_node = with_global_node
        self.food_amount = food_amount
        self.env_type = env_type
        self.i = 0
        self.steps = 50
        self.pool_size = 1024
        pool_graphs = [generate_organism(self.n, self.device, self.with_global_node, self.food_amount, self.env_type).toGraph().x.detach().cpu().numpy()
                       for _ in range(self.pool_size)]
        self.sample_pool = SamplePool(x=pool_graphs)
        self.set_global_var()

        self.cell_threshold = 5
        self.food_threshold = 10
        self.in_population = 0

    def set_global_var(self):
        self.i += 1
        if self.i % 2000 == 0:
            if self.i > 3000:
                ...
            #if self.i % 1500 == 0:
            #    self.steps = 40
            else:
                self.steps += 10
            print(self.steps)
        self.time_steps = np.random.randint(self.steps, self.steps+10)
        #self.graphs = [generate_organism(self.n, self.device, self.with_global_node, self.food_amount, self.env_type).toGraph() for _ in range(self.batch_size)]

        self.batch = self.sample_pool.sample(self.batch_size)
        self.graphs = [toGraph(torch.tensor(x, device=self.device), self.device) for x in self.batch.x.values()]
        #self.graphs = torch.tensor(self.batch, device=self.device)
        self.in_population = 0

    def update_pool(self, graphs):
        self.in_population +=1

        #if locked correctly this should ensure only a single member of the population gets to commit their environment
        if self.in_population != 5: return #TODO use half of population size instead of hardcoded nr 5

        nodes = unbatch_nodes(graphs, self.batch_size)
        for i in range(len(nodes)):
            cells = nodes[i][:, 4] == 1
            food = nodes[i][:, 4] == 0
            if cells.sum() < self.cell_threshold or food.sum() < self.food_threshold:
                #called when we need to generate a new environment
                nodes[i] = generate_organism(self.n, self.device, self.with_global_node, self.food_amount, self.env_type).toGraph().x.detach().cpu().numpy()

        self.batch.x = nodes
        self.batch.commit()

    def get_global_var(self):
        return self.time_steps, self.graphs