import torch
from generator import generate_food
import random
import torch_geometric
from torch_geometric import utils

def add_food(graph, food):
    '''Add food source as node to graph'''
    graph.x = torch.cat((graph.x, food))

def add_random_food(graph, device, n=1):
    '''Add n random food sources as nodes to the graph'''
    for _ in range(n):
        food = generate_food(device)
        add_food(graph, food)

def update_velocity(graph, acceleration, max_velocity, c_mask):
    '''Updates the velocity of the nodes given the acceleration and previous velocity'''
    #TODO change back to accumulate velocity
    #velocity = graph.x[:, 2:4] + acceleration
    velocity = acceleration
    velocity[c_mask] = torch.clamp(velocity[c_mask], -max_velocity, max_velocity)
    return velocity

def update_positions(graph, velocity, wrap_around, c_mask):
    '''Updates the position of the nodes given the velocity and previous positions'''
    positions = torch.remainder(graph.x[c_mask, :2] + velocity[c_mask] + 1.0, 2.0) - 1.0
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
    '''Remove cells without any edges'''
    c_mask = cell_mask(graph)
    cell_edge_indices = torch.nonzero(graph.edge_attr[:, 3] == 1).flatten()
    zero_edge_mask = torch.bincount(graph.edge_index[0, cell_edge_indices], minlength=graph.x.shape[0]) < edges_to_stay_alive
    mask = torch.bitwise_and(c_mask, zero_edge_mask)
    return mask

def compute_border_cost(graph):
    epsilon = 0.000001 #Used to prevent taking log of 0.0 resulting in nan values
    mask = graph.x[:, :2].abs() > 1
    border_costX = (graph.x[:, 0].abs()+epsilon).log() * mask[:,0].to(torch.float)
    border_costY = (graph.x[:, 1].abs()+epsilon).log() * mask[:,1].to(torch.float)
    graph.border_cost += (border_costX.sum() + border_costY.sum())