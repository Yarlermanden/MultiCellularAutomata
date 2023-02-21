from evotorch.tools import dtype_of, device_of
from evotorch.neuroevolution import NEProblem
from evotorch.algorithms import PGPE, SNES, XNES, CMAES
from evotorch.logging import PandasLogger, StdOutLogger
from evotorch.algorithms import GeneticAlgorithm
from evotorch.operators import GaussianMutation, SimulatedBinaryCrossOver
import torch
import numpy as np
import ray
from torch_geometric.utils import to_networkx
import networkx as nx

from generator import generate_organism
from GNCAmodel import GNCA
from GATConv import GATConv
from CGConv import CGConv1
from SpatioTemporalModel import SpatioTemporal
from evotorch.decorators import vectorized, on_aux_device

@ray.remote
class GlobalVarActor():
    def __init__(self, n, device):
        self.n = n
        self.device = device
        self.set_global_var()

    def set_global_var(self):
        #self.time_steps = np.random.randint(40, 50)
        self.time_steps = np.random.randint(80, 100)
        #self.time_steps = np.random.randint(150, 200)
        self.organism = generate_organism(self.n, self.device)

    def get_global_var(self):
        return self.time_steps, self.organism

class Custom_NEProblem(NEProblem):
    def __init__(self, n, global_var, **kwargs):
        super(Custom_NEProblem, self).__init__(**kwargs)
        self.n = n
        self.global_var = global_var

    @vectorized
    @on_aux_device
    def _evaluate_network(self, network: torch.nn.Module):
        steps, organism = ray.get(self.global_var.get_global_var.remote())
        #organism = generate_organism(self.n, self.device)
        graph = organism.toGraph()

        with torch.no_grad():
            graph, velocity_bonus, border_cost, food_reward, dead_cost, visible_food, food_avg_degree, mean_food_dist = network(graph, steps)

        #G = to_networkx(graph, to_undirected=True)
        #largest_component = max(nx.connected_components(G), key=len) #subgraph with organism
        #G2 = G.subgraph(largest_component) 
        #diameter = nx.diameter(G2) #shortest longest path

        #TODO add reward for the average degree of each visible food - encourage more nodes seeing the same food - hopefully going towards it...
        #fitness = (velocity_bonus.sum() + visible_food/10 + food_avg_degree*diameter*food_reward*100/(1+velocity_bonus.mean()*100)) / (1+dead_cost+border_cost + visible_food/100) - border_cost/4
        #fitness = velocity_bonus.sum() + food_reward*10*velocity_bonus.sum()/(1+border_cost/10+dead_cost/100)

        #fitness = (visible_food+food_reward*1000) / (1+velocity_bonus.mean()*100 + border_cost*10) 

        fitness = (food_reward**3)

        if torch.isnan(fitness): #TODO if this turned out to be the fix - should investigate why any network returns nan
            print("fitness function returned nan")
            print((food_reward, velocity_bonus.mean(), border_cost, dead_cost))
            fitness = -10000
        #return torch.tensor([fitness, velocity_bonus.sum(), -border_cost, food_reward, -dead_cost], dtype=torch.float)
        return torch.tensor([fitness, velocity_bonus.sum(), food_reward, dead_cost, visible_food, mean_food_dist], dtype=torch.float)

class Evo_Trainer():
    def __init__(self, n, device, wrap_around, popsize=None):
        global_var = GlobalVarActor.remote(n, device)
        ray.get(global_var.set_global_var.remote())
        self.wrap_around = wrap_around

        self.problem = Custom_NEProblem(
            n=n,
            global_var=global_var,
            device=device,
            objective_sense=['max', 'min', 'max', 'min', 'max', 'min'],
            network=CGConv1,
            #network=SpatioTemporal,
            #network=GATConv,
            network_args={'device': device, 'wrap_around': wrap_around},
            num_actors='max',
            num_gpus_per_actor = 'max',
        )
        #self.searcher = CMAES(
        #    self.problem,
        #    stdev_init=torch.tensor(0.1, dtype=torch.float),
        #    popsize=popsize,
        #    limit_C_decomposition=False,
        #    obj_index=0
        #)
        self.searcher = GeneticAlgorithm(
            self.problem,
            popsize=200,
            operators=[
                SimulatedBinaryCrossOver(self.problem, tournament_size=4, cross_over_rate=1.0, eta=8),
                GaussianMutation(self.problem, stdev=0.03),
            ],
        )

        def before_epoch():
            ray.get(global_var.set_global_var.remote())
            
        self.searcher.before_step_hook.append(before_epoch)
        self.logger = StdOutLogger(self.searcher)
        self.logger = PandasLogger(self.searcher)
        self.logger_df = None
        self.trained_network = None


    def train(self, n=1000, name='test1'):
        self.searcher.run(n)
        self.logger_df = self.logger.to_dataframe()
        self.logger_df.to_csv('../logger/' + name + '.csv')
        #self.trained_network = self.problem.parameterize_net(self.searcher.status['center'][0])
        self.trained_network = self.problem.parameterize_net(self.searcher.status['best'][0])
        torch.save(self.trained_network.state_dict(), '../models/' + name + '.pth')

    def visualize_training(self):
        logger_df = self.logger_df.groupby(np.arange(len(logger_df))).mean()
        logger_df.plot(y='mean_eval')

    def get_trained_network(self):
        return self.trained_network
