import numpy as np
import ray
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.utils import unbatch
import random

from sample_pool import SamplePool
from generator import generate_organism
from organism import Organism, toGraph
from graphUtils import unbatch_nodes, cell_mask, food_mask

@ray.remote
class GlobalState():
    def __init__(self, settings):
        self.settings = settings
        self.device = settings.device
        self.batch_size = settings.batch_size
        
        self.steps = settings.train_config.timesteps
        self.i = 0

        if settings.train_config.with_samplepool:
            self.pool_size = 1024
            pool_graphs = [generate_organism(settings).toGraph().x.detach().cpu().numpy()
                            for _ in range(self.pool_size)]
            self.sample_pool = SamplePool(x=pool_graphs)
        self.set_global_var()

        self.cell_threshold = settings.n // 5 #if below threshold - generate new env instead of committing
        self.food_threshold = settings.food_envs[0].food_amount // 10 # -||-
        self.in_population = 0

    def set_global_var(self):
        self.i += 1
        if self.i % 100 == 0:
            self.steps += 10
            print(self.steps)
        self.time_steps = np.random.randint(self.steps, self.steps+20)

        if self.settings.train_config.with_samplepool:
            self.batch = self.sample_pool.sample(self.batch_size)
            self.graphs = [toGraph(torch.tensor(x, device=self.device), self.device) for x in self.batch.x.values()]
        else: 
            random_number = random.randint(0, len(self.settings.food_envs)-1)
            food_env = self.settings.food_envs[random_number]
            self.graphs = [generate_organism(self.settings).toGraph(food_env) for _ in range(self.batch_size)]
        self.graphs = DataLoader(self.graphs, batch_size=self.settings.batch_size)
        self.in_population = 0

    def update_pool(self, graphs):
        self.in_population +=1

        #if locked correctly this should ensure only a single member of the population gets to commit their environment
        if self.in_population != self.settings.train_config.popsize//2: return #TODO use half of population size instead of hardcoded nr 5

        nodes = unbatch_nodes(graphs, self.batch_size)
        for i in range(len(nodes)):
            node = torch.tensor(nodes[i], device=self.device)
            cells = cell_mask(node)
            #TODO change cells here to use graph util and change graphutils to take x directly....
            food = food_mask(node)
            if cells.sum() < self.cell_threshold or food.sum() < self.food_threshold:
                #called when we need to generate a new environment
                nodes[i] = generate_organism(self.settings).toGraph().x.detach().cpu().numpy()

        if self.settings.train_config.with_samplepool:
            self.batch.x = nodes
            self.batch.commit()

    def get_global_var(self):
        return self.time_steps, self.graphs