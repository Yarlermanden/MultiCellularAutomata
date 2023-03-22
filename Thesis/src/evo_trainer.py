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
from torch_geometric.data import Data
import time

from generator import generate_organism
from GNCAmodel import GNCA
from GNCAConv import Conv
from evotorch.decorators import vectorized, on_aux_device
from global_state import GlobalState

class Custom_NEProblem(NEProblem):
    def __init__(self, n, global_var, batch_size, **kwargs):
        super(Custom_NEProblem, self).__init__(**kwargs)
        self.n = n
        self.global_var = global_var
        self.batch_size = batch_size

    @vectorized
    @on_aux_device
    def _evaluate_network(self, network: torch.nn.Module):
        time1 = time.perf_counter()
        steps, graphs = ray.get(self.global_var.get_global_var.remote())
        loader = DataLoader(graphs, batch_size=self.batch_size) #TODO move this part into the global_state to not do multiple times
        batch = next(iter(loader))
        time2 = time.perf_counter()

        with torch.no_grad():
            graph = network(batch, steps)
        time3 = time.perf_counter()

        s_idx = 0
        diameters = []
        subgraph_counts = []
        if graph.edge_index.shape[0] == 2:
            for i in range(self.batch_size):
                e_idx = graph.subsize[i]
                nodes_in_batch = torch.nonzero(graph.x[s_idx:e_idx, 4] == 1) + s_idx
                edges_in_batch = torch.nonzero(torch.isin(graph.edge_index[1], nodes_in_batch)).view(-1)
                nodes = graph.x[nodes_in_batch]
                edges = graph.edge_index[:, edges_in_batch]
                if len(nodes) == 0:
                    diameters.append(0)
                    subgraph_counts.append(1)
                else:
                    data = Data(x=nodes, edge_index=edges)
                    G = to_networkx(data, to_undirected=True)
                    G1 = nx.connected_components(G)
                    largest_component = max(G1, key=len) #subgraph with organism
                    G2 = G.subgraph(largest_component) 
                    diameters.append(nx.diameter(G2)) #shortest longest path
                    x = len([0 for _ in enumerate(G1)])
                    if x == 0:
                        x = 1
                    subgraph_counts.append(x)
                s_idx = e_idx
        else:
            [subgraph_counts.append(1) for _ in range(self.batch_size)]
            diameters.append(0)
        diameters = torch.tensor(diameters, dtype=torch.float)
        subgraph_counts = torch.tensor(subgraph_counts, dtype=torch.float)

        food_reward = graph.food_reward.mean()
        velocity = graph.velocity.mean()
        fitness1 = (((food_reward)) / (1 + velocity*20))
        fitness2 = velocity
        fitness3 = graph.food_search_movement.mean() * 10
        fitness4 = diameters.mean()/self.n * (1+fitness1) #0-1 times fitness1
        fitness5 = (graph.x[:, 4] == 1).sum()/(self.n*self.batch_size) * 3 #cells alive ratio
        fitness = fitness1 + fitness3 + fitness4 + fitness5
        fitness /= subgraph_counts.mean()**0.5

        if torch.any(torch.isnan(food_reward)):
            print("fitness function returned nan")
            print((graph.food_reward, graph.velocity, graph.border_cost, graph.dead_cost))

        time4 = time.perf_counter()
        #print('setup: ', time2-time1)
        #print('all graph computations: ', time3-time2)
        #print('fitness computation: ', time4-time3)

        ray.get(self.global_var.update_pool.remote(graph))
        return torch.tensor([fitness, fitness2])

class Evo_Trainer():
    def __init__(self, n, device, batch_size, wrap_around, with_global_node, food_amount, env_type, popsize=200):
        cpu = torch.device('cpu')
        global_var = GlobalState.remote(n, cpu, batch_size, with_global_node, food_amount, env_type)
        ray.get(global_var.set_global_var.remote())
        self.wrap_around = wrap_around

        self.problem = Custom_NEProblem(
            n=n,
            global_var=global_var,
            batch_size=batch_size,
            device=cpu,
            #objective_sense=['max', 'min', 'max', 'min'],
            objective_sense=['max', 'max'],
            network=Conv,
            network_args={'device': device, 'batch_size': batch_size, 'wrap_around': wrap_around, 'with_global_node': with_global_node},
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
        self.trained_network = self.problem.parameterize_net(self.searcher.status['center'][0])
        #self.trained_network = self.problem.parameterize_net(self.searcher.status['best'][0])
        torch.save(self.trained_network.state_dict(), '../models/' + name + '.pth')

    def visualize_training(self):
        logger_df = self.logger_df.groupby(np.arange(len(self.logger_df))).mean()
        logger_df.plot(y='mean_eval')

    def get_trained_network(self):
        return self.trained_network
