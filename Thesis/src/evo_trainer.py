from evotorch.tools import dtype_of, device_of
from evotorch.neuroevolution import NEProblem
from evotorch.algorithms import PGPE, SNES, XNES, CMAES
from evotorch.logging import PandasLogger, StdOutLogger
import torch
import numpy as np

from generator import generate_organism
from GNCAmodel import GNCA


class Custom_NEProblem(NEProblem):
    def __init__(self, n, device, **kwargs):
        super(Custom_NEProblem).__init__(kwargs)
        self.n = n
        self.device = device

    def _evaluate_network(self, network: torch.nn.Module):
        organism = generate_organism(self.n, self.device)
        graph = organism.toGraph()

        graph = network(graph, 1)

        #compute fitness
        #distance cost
        #velocity bonus
        #position cost

        return 0 #return fitness

class Evo_Trainer():
    def __init__(self, device):
        self.device = device

        self.problem = Custom_NEProblem(
            objective_sense='max',
            network=GNCA,
            network_args={'device': self.device},
            device=self.device,
            num_actors='max',
        )

        self.searcher = CMAES(
            self.problem,
            stdev_init=torch.tensor(0.1),
            popsize=10,
        )
        #searcher.before_step_hook.append(before)

        self.logger = StdOutLogger(self.searcher)
        self.logger = PandasLogger(self.searcher)
        self.logger_df = None

    def train(self, n=1000):
        self.searcher.run(n)
        self.logger_df = self.logger.to_dataframe()
        self.logger_df.to_csv('../logger/evo1.csv')

    def visualize_training(self):
        logger_df = self.logger_df.groupby(np.arange(len(logger_df))).mean()
        logger_df.plot(y='mean_eval')

    def get_trained_network(self):
        trained_network = self.problem.parameterize_net(self.searcher.status['center'])
        torch.save(trained_network.state_dict(), '../models/evo1.pth')
