from evotorch.algorithms import PGPE, SNES, XNES, CMAES
from evotorch.logging import PandasLogger, StdOutLogger
from evotorch.algorithms import GeneticAlgorithm
from evotorch.operators import GaussianMutation, SimulatedBinaryCrossOver
import torch
import numpy as np
import ray
import time
import math
import pickle as Pickle
import dill

from GNCAConv import Conv
from global_state import GlobalState
from evo_problem import Custom_NEProblem
from enums import *
from evo_logger import Custom_Logger

class Evo_Trainer():
    def __init__(self, settings, online_tracker):
        cpu = torch.device('cpu')
        self.settings = settings
        self.online_tracker = online_tracker
        self.global_var = GlobalState.remote(settings)
        ray.get(self.global_var.set_global_var.remote())

        self.problem = Custom_NEProblem(
            settings=settings,
            global_var=self.global_var,
            device=cpu,
            objective_sense=['max'],
            network=Conv,
            network_args={'settings' : settings},
            num_actors='max',
            num_gpus_per_actor = 'max',
        )

        self.searcher = None
        if settings.train_config.problem_searcher == ProblemSearcher.GeneticAlgorithm:
            self.searcher = GeneticAlgorithm(
                self.problem,
                popsize=settings.train_config.popsize,
                operators=[
                    SimulatedBinaryCrossOver(self.problem, tournament_size=4, cross_over_rate=1.0, eta=8),
                    GaussianMutation(self.problem, stdev=settings.train_config.stdev),
                ],
            )
        else:
            self.searcher = CMAES(
                self.problem,
                stdev_init=torch.tensor(settings.train_config.stdev, dtype=torch.float),
                popsize=settings.train_config.popsize,
                limit_C_decomposition=False,
                obj_index=0,
            )
            
        self.searcher.before_step_hook.append(self.before_epoch)
        self.logger = StdOutLogger(self.searcher)
        self.logger = Custom_Logger(self.searcher, self.online_tracker)
        self.logger_df = None
        self.trained_network = Conv(settings=settings)

    def before_epoch(self):
        ray.get(self.global_var.set_global_var.remote())

    def parameterize_net(self):
        if self.settings.train_config.problem_searcher == ProblemSearcher.GeneticAlgorithm:
            self.trained_network = self.problem.parameterize_net(self.searcher.status['best'][0])
        else:
            self.trained_network = self.problem.parameterize_net(self.searcher.status['center'][0])

    def train(self, n=1000, name=None):
        if name is None:
            name = self.settings.train_config.name
        n1 = n
        t = math.ceil(n/1000)
        for _ in range(t):
            x = n1 if n1<1000 else 1000
            self.searcher.run(x)
            self.logger_df = self.logger.to_dataframe()
            self.logger_df.to_csv('../logger/' + name + '.csv')
            self.parameterize_net()
            torch.save(self.trained_network.state_dict(), '../models/' + name + '.pth')
            n1 -= x
        #self.save_searcher(name)

    def visualize_training(self):
        logger_df = self.logger_df.groupby(np.arange(len(self.logger_df))).mean()
        logger_df.plot(y='mean_eval')

    def get_trained_network(self):
        self.parameterize_net()
        return self.trained_network

    def save_searcher(self, name=None):
        problem = self.searcher.problem
        if name is None:
            name = self.settings.train_config.name
        with open(r'../searcher/' + name + '.pkl', 'wb') as file:
            dic = self.searcher.__dict__.copy()
            print(dic)
            del dic['problem']
            dill.dump(dic, file)
            #Pickle.dump(self.searcher, file)
            #pd.to_pickle(self.searcher, file)

    def load_searcher(self, name=None):
        if name is None:
            name = self.settings.train_config.name
        with open(r'../searcher/' + name + '.pkl', 'rb') as file:
            #dic = Pickle.load(file)
            searcher = dill.load(file)
            searcher.problem = self.problem
            self.searcher = searcher

            #if self.settings.train_config.problem_searcher == ProblemSearcher.GeneticAlgorithm:
            #    self.searcher = GeneticAlgorithm(**dic)
            #else:
            #    self.searcher = CMAES(**dic)

#TODO add something for actually saving the problem in such a way we can reinitialize it and continue
#possibly we could also change the settings in regarding to change the population and such in the population
#to match the next training session
