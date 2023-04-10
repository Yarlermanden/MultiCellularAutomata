import torch
from generator import *
import random
import torch_geometric
from torch_geometric import utils

def add_food(graph, food):
    '''Add food source as node to graph'''
    graph.x = torch.cat((graph.x, food))

#TODO refactor the below three methods to a single taking environment and switching on how to generate...
def add_random_food(graph, device, n=1, scale=1):
    '''Add n random food sources as nodes to the graph'''
    for _ in range(n):
        food = generate_food(device, scale)
        add_food(graph, food)

def add_circular_food(graph, device, n=1, scale=1, circles=1):
    for _ in range(n):
        food = generate_circular_food(device, scale, std_dev=0, circles=circles, radius=scale//2) #TODO implement with std... to randomize around circle
        add_food(graph, food)

def add_spiral_food(graph, device, n=1, scale=1, spirals=1):
    rotation = random.uniform(0, 2*np.pi)
    for _ in range(n):
        food = generate_spiral_food(device, scale, std_dev=0, spirals=spirals, rotation=rotation)
        add_food(graph, food)

def add_clusters_of_food(graph, device, n=1, cluster_size=20, std_dev=0.1, scale=1):
    '''Generates and adds n clusters of food to the graph'''
    for _ in range(n):
        cluster = generate_cluster(device, cluster_size, std_dev, scale)
        add_food(graph, cluster)

def add_global_node(graph, device):
    '''Adds a global node to the graph. 
    Call this before creating batches to ensure a global node exists in all batches'''
    hidden = [0,0,0,0,0]
    global_node = torch.tensor([[0, 0, 0, 0, 2, 0, *hidden]], dtype=torch.float, device=device)
    graph.x = torch.cat((graph.x, global_node))

def update_velocity(graph, acceleration, max_velocity, c_mask):
    '''Updates the velocity of the nodes given the acceleration and previous velocity'''
    #velocity = graph.x[:, 2:4] + acceleration #Seems to difficult to learn as it's simply just another complexity on top of basic movement
    velocity = acceleration
    velocity[c_mask] = torch.clamp(velocity[c_mask], -max_velocity, max_velocity)
    return velocity

def update_positions(graph, velocity, wrap_around, c_mask, scale):
    '''Updates the position of the nodes given the velocity and previous positions'''
    if wrap_around:
        positions = torch.remainder(graph.x[c_mask, :2] + velocity[c_mask] + scale, 2*scale) - scale
    else:
        positions = graph.x[c_mask, :2] + velocity[c_mask]
    return positions

def food_mask(graph):
    '''Used to mask away all cell nodes to only keep food'''
    mask = graph.x[:, 4] == 0
    return mask

def cell_mask(graph):
    '''Used to mask away all food nodes to only keep cell nodes'''
    mask = graph.x[:, 4] == 1
    return mask

def get_consume_food_mask(graph, consume_radius, consumption_edge_required):
    '''Consumes food if criteria is met and returns reward'''
    f_mask = food_mask(graph)
    edge_below_distance = torch.nonzero(graph.edge_attr[:, 0] < consume_radius).flatten()
    edges_pr_node = torch.bincount(graph.edge_index[0, edge_below_distance], minlength=graph.x.shape[0])
    edge_mask = edges_pr_node >= consumption_edge_required
    consumption_mask = torch.bitwise_and(f_mask, edge_mask)
    return consumption_mask

def get_island_cells_mask(graph, edges_to_stay_alive):
    '''Return mask of cells with less than required amount of edges'''
    c_mask = cell_mask(graph)
    cell_edge_indices = torch.nonzero(graph.edge_attr[:, 3] == 1).flatten()
    too_few_edges_mask = torch.bincount(graph.edge_index[0, cell_edge_indices], minlength=graph.x.shape[0]) < edges_to_stay_alive
    mask = torch.bitwise_and(c_mask, too_few_edges_mask)
    return mask

def get_dead_cells_mask(graph, energy_required):
    '''Returns mask of cells with less than required energy level'''
    c_mask = cell_mask(graph)
    e_mask = graph.x[:, 5] < energy_required
    mask = torch.bitwise_and(c_mask, e_mask)
    return mask

def compute_border_cost(graph):
    epsilon = 0.000001 #Used to prevent taking log of 0.0 resulting in nan values
    mask = graph.x[:, :2].abs() > 1
    border_costX = (graph.x[:, 0].abs()+epsilon).log() * mask[:,0].to(torch.float)
    border_costY = (graph.x[:, 1].abs()+epsilon).log() * mask[:,1].to(torch.float)
    graph.border_cost += (border_costX.sum() + border_costY.sum())

def unbatch_nodes(graphs, batch_size):
    '''Unbatches nodes and returns a list of list of nodes in the minibatch'''
    nodes = []
    s_idx = 0
    for batch_idx in range(batch_size):
        e_idx = s_idx + graphs.subsize[batch_idx]
        
        nodes_in_batch = graphs.x[s_idx:e_idx]
        nodes.append(nodes_in_batch.detach().cpu().numpy())
        s_idx = e_idx
    return nodes