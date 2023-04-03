from evotorch.neuroevolution import NEProblem
import torch
import ray
from torch_geometric.utils import to_networkx
import networkx as nx
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from evotorch.decorators import vectorized, on_aux_device

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
        loader = DataLoader(graphs, batch_size=self.batch_size) #TODO move this part into the global_state to not do multiple times
        batch = next(iter(loader))
        alive_start = (batch.x[:, 4] == 1).sum()

        with torch.no_grad():
            graph = network(batch, steps)

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
        alive_end = (graph.x[:, 4] == 1).sum()
        cell_ratio = alive_end/alive_start

        food_reward = graph.food_reward.mean()
        fitness = food_reward
        #fitness1 = food_reward * 2
        #fitness3 = graph.food_search_movement.mean() * 15
        ##fitness4 = diameters.mean()/self.n * (1+fitness1) #0-1 times fitness1 - encourage large diameter to tage complex shapes
        #fitness = fitness1 + fitness3
        #fitness /= subgraph_counts.mean() #The more subgraphs, the more it has split - encourage to stay together
        #fitness *= (1+cell_ratio)/5 #multiply by ratio kept alive [0-1] - encourage to stay alive

        if torch.any(torch.isnan(food_reward)):
            print("fitness function returned nan")
            print((graph.food_reward, graph.velocity, graph.border_cost, graph.dead_cost))

        ray.get(self.global_var.update_pool.remote(graph))
        return torch.tensor([fitness, fitness])