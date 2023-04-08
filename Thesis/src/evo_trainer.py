from evotorch.algorithms import PGPE, SNES, XNES, CMAES
from evotorch.logging import PandasLogger, StdOutLogger
from evotorch.algorithms import GeneticAlgorithm
from evotorch.operators import GaussianMutation, SimulatedBinaryCrossOver
import torch
import numpy as np
import ray
import time
import math

from GNCAConv import Conv
from global_state import GlobalState
from evo_problem import Custom_NEProblem

class Evo_Trainer():
    def __init__(self, settings, popsize=200):
        cpu = torch.device('cpu')
        self.settings = settings
        global_var = GlobalState.remote(settings)
        ray.get(global_var.set_global_var.remote())

        self.problem = Custom_NEProblem(
            settings=settings,
            global_var=global_var,
            device=cpu,
            objective_sense=['max', 'max', 'max', 'max'],
            network=Conv,
            network_args={'settings' : settings},
            num_actors='max',
            num_gpus_per_actor = 'max',
        )

        self.distribution_searcher = CMAES(
            self.problem,
            stdev_init=torch.tensor(0.04, dtype=torch.float),
            popsize=popsize,
            limit_C_decomposition=False,
            obj_index=0,
        )

        self.population_searcher = GeneticAlgorithm(
            self.problem,
            popsize=popsize,
            operators=[
                SimulatedBinaryCrossOver(self.problem, tournament_size=4, cross_over_rate=1.0, eta=8),
                GaussianMutation(self.problem, stdev=0.02),
            ],
        )

        self.searcher=self.distribution_searcher
        #self.searcher=self.population_searcher

        def before_epoch():
            ray.get(global_var.set_global_var.remote())
            
        self.searcher.before_step_hook.append(before_epoch)
        self.logger = StdOutLogger(self.searcher)
        self.logger = PandasLogger(self.searcher)
        self.logger_df = None
        self.trained_network = None

    def train(self, n=1000, name='test1'):
        n1 = n
        t = math.ceil(n/1000)
        for _ in range(t):
            x = n1 if n1<1000 else 1000
            self.searcher.run(x)
            self.logger_df = self.logger.to_dataframe()
            self.logger_df.to_csv('../logger/' + name + '.csv')
            self.trained_network = self.problem.parameterize_net(self.searcher.status['center'][0])
            #self.trained_network = self.problem.parameterize_net(self.searcher.status['best'][0])
            torch.save(self.trained_network.state_dict(), '../models/' + name + '.pth')
            n1 -= x

    def visualize_training(self):
        logger_df = self.logger_df.groupby(np.arange(len(self.logger_df))).mean()
        logger_df.plot(y='mean_eval')

    def get_trained_network(self):
        return self.trained_network
