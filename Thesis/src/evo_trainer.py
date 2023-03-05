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
from torch_geometric.loader import DataLoader

from generator import generate_organism
from GNCAmodel import GNCA
from GATConv import GATConv
from CGConv import CGConv1
from SpatioTemporalModel import SpatioTemporal
from evotorch.decorators import vectorized, on_aux_device

@ray.remote
class GlobalVarActor():
    def __init__(self, n, device, batch_size):
        self.n = n
        self.device = device
        self.batch_size = batch_size
        self.set_global_var()

    def set_global_var(self):
        self.time_steps = np.random.randint(50, 60)
        #self.time_steps = np.random.randint(80, 100)
        #self.time_steps = np.random.randint(150, 200)
        self.graphs = [generate_organism(self.n, self.device).toGraph() for _ in range(self.batch_size)]

    def get_global_var(self):
        return self.time_steps, self.graphs

class Custom_NEProblem(NEProblem):
    def __init__(self, n, global_var, batch_size, **kwargs):
        super(Custom_NEProblem, self).__init__(**kwargs)
        self.n = n
        self.global_var = global_var
        self.batch_size = batch_size

    @vectorized
    @on_aux_device
    def _evaluate_network(self, network: torch.nn.Module):
        steps, graphs = ray.get(self.global_var.get_global_var.remote())
        #organism = generate_organism(self.n, self.device)
        loader = DataLoader(graphs, batch_size=self.batch_size)
        batch = next(iter(loader))

        with torch.no_grad():
            graph = network(batch, steps)

        #G = to_networkx(graph, to_undirected=True)
        #largest_component = max(nx.connected_components(G), key=len) #subgraph with organism
        #G2 = G.subgraph(largest_component) 
        #diameter = nx.diameter(G2) #shortest longest path

        #TODO add reward for the average degree of each visible food - encourage more nodes seeing the same food - hopefully going towards it...
        #fitness = (velocity_bonus.sum() + visible_food/10 + food_avg_degree*diameter*food_reward*100/(1+velocity_bonus.mean()*100)) / (1+dead_cost+border_cost + visible_food/100) - border_cost/4
        #fitness = velocity_bonus.sum() + food_reward*10*velocity_bonus.sum()/(1+border_cost/10+dead_cost/100)

        #fitness = (visible_food+food_reward*1000) / (1+velocity_bonus.mean()*100 + border_cost*10) 

        #norm = 0
        #for x in network.parameters():
        #    for param in x:
        #        norm += param.data.norm()

        food_reward = graph.food_reward.mean()
        velocity = graph.velocity.mean()
        fitness1 = (((food_reward)) / (1 + velocity*20))
        fitness2 = velocity
        fitness3 = graph.food_search_movement.mean()
        fitness = fitness1 + fitness3*5

        if torch.any(torch.isnan(food_reward)): #TODO if this turned out to be the fix - should investigate why any network returns nan
            print("fitness function returned nan")
            print((graph.food_reward, graph.velocity, graph.border_cost, graph.dead_cost))
        #return torch.tensor([fitness, velocity_bonus.sum(), food_reward, dead_cost, visible_food/1000, mean_food_dist/10], dtype=torch.float)
        #return torch.tensor([food_reward, visible_food/1000, mean_food_dist/10], dtype=torch.float)
        #return torch.tensor([food_reward*100, graph.food_avg_dist.mean()/1000, graph.visible_food.mean()/(20-food_reward), graph.velocity.mean()/(1+food_reward)], dtype=torch.float).cpu()

        #return torch.tensor([fitness, graph.food_avg_dist.mean()/1000, graph.visible_food.mean()/(20-food_reward), graph.velocity.mean()/(1+food_reward)], dtype=torch.float).cpu()
        #return torch.tensor([fitness1, fitness2])
        return torch.tensor([fitness, fitness2])

class Evo_Trainer():
    def __init__(self, n, device, batch_size, wrap_around, popsize=200):
        cpu = torch.device('cpu')
        global_var = GlobalVarActor.remote(n, cpu, batch_size)
        ray.get(global_var.set_global_var.remote())
        self.wrap_around = wrap_around

        self.problem = Custom_NEProblem(
            n=n,
            global_var=global_var,
            batch_size=batch_size,
            device=cpu,
            #objective_sense=['max', 'min', 'max', 'min'],
            objective_sense=['max', 'max'],
            network=CGConv1,
            #network=SpatioTemporal,
            #network=GATConv,
            network_args={'device': device, 'batch_size': batch_size, 'wrap_around': wrap_around},
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
