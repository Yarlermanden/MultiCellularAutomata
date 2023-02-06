from evotorch.tools import dtype_of, device_of
from evotorch.neuroevolution import NEProblem
from evotorch.algorithms import PGPE, SNES, XNES, CMAES
from evotorch.logging import PandasLogger, StdOutLogger
import torch
import numpy as np
import ray

from generator import generate_organism
from GNCAmodel import GNCA

@ray.remote
class GlobalVarActor():
    def __init__(self):
        self.time_steps = self.set_global_var()

    def set_global_var(self):
        self.time_steps = np.random.randint(50, 100)

    def get_global_var(self):
        return self.time_steps

def before_epoch():
    ray.get(global_var.set_global_var.remote())

global_var = GlobalVarActor.remote()
ray.get(global_var.set_global_var.remote())

class Custom_NEProblem(NEProblem):
    def __init__(self, n, **kwargs):
        super(Custom_NEProblem, self).__init__(**kwargs)
        self.n = n

    def _evaluate_network(self, network: torch.nn.Module):
        steps = ray.get(global_var.get_global_var.remote())
        organism = generate_organism(self.n, self.device)
        graph = organism.toGraph()

        graph, velocity_bonus, border_cost, food_reward, dead_cost = network(graph, steps)

        fitness = velocity_bonus.sum()/2 - border_cost + food_reward*30 - dead_cost
        if torch.isnan(fitness): #TODO if this turned out to be the fix - should investigate why any network returns nan
            fitness = -1000
        return fitness

class Evo_Trainer():
    def __init__(self, n, device, popsize=10):
        self.problem = Custom_NEProblem(
            n=n,
            device=device,
            objective_sense='max',
            network=GNCA,
            network_args={'device': device},
            num_actors='max',
        )
        self.device = device
        self.searcher = CMAES(
            self.problem,
            stdev_init=torch.tensor(0.1),
            popsize=popsize,
        )
        self.searcher.before_step_hook.append(before_epoch)

        self.logger = StdOutLogger(self.searcher)
        self.logger = PandasLogger(self.searcher)
        self.logger_df = None
        self.trained_network = None

    def train(self, n=1000, name='test1'):
        self.searcher.run(n)
        self.logger_df = self.logger.to_dataframe()
        self.logger_df.to_csv('../logger/' + name + '.csv')
        self.trained_network = self.problem.parameterize_net(self.searcher.status['center'])
        torch.save(self.trained_network.state_dict(), '../models/' + name + '.pth')

    def visualize_training(self):
        logger_df = self.logger_df.groupby(np.arange(len(logger_df))).mean()
        logger_df.plot(y='mean_eval')

    def get_trained_network(self):
        return self.trained_network
