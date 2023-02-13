import torch
from generator import generate_food
import random
import torch_geometric
from torch_geometric import utils

def add_edges(graph, radius, device):
    '''Add edges dynamically according to radius. '''
    edges = []
    edge_attributes = []
    radius_food = radius*4
    
    cell_indices = torch.nonzero(graph.x[:, 4] == 1).flatten()
    food_indices = torch.nonzero(graph.x[:, 4] == 0).flatten()

    def add_edge(i: int, j: int, with_food: bool):
        radius_to_use = radius
        cell_to_cell = 1
        if with_food:
            radius_to_use = radius_food
            cell_to_cell = 0
        dist = (graph.x[i]-graph.x[j])[:2].norm()
        if dist < radius_to_use:
            edges.append([i, j])
            edge_attribute = [dist, cell_to_cell]
            edge_attributes.append(edge_attribute)

    n = len(cell_indices)
    for i_i in range(n):
        for j in food_indices: #check distance to food sources
            add_edge(cell_indices[i_i], j, True)

        for i_j in range(i_i+1, n): #check distance to other cells
            add_edge(cell_indices[i_i], cell_indices[i_j], False)

    if len(edges) == 0:
        graph.x = graph.x[food_indices].view(food_indices.shape[0], graph.x.shape[1])
        return False
    edge_index = torch.tensor(edges, dtype=torch.long, device=device).T
    edge_attr = torch.tensor(edge_attributes, device=device)
    graph.edge_index, graph.edge_attr = utils.to_undirected(edge_index, edge_attr)
    return True

def add_food(graph, food):
    '''Add food source as node to graph'''
    graph.x = torch.cat((graph.x, food))

def add_random_food(graph, device, n=1):
    '''Add n random food sources as nodes to the graph'''
    for _ in range(n):
        food = generate_food(device)
        add_food(graph, food)

def update_velocity(graph, acceleration, max_velocity):
    '''Updates the velocity of the nodes given the acceleration and previous velocity'''
    velocity = graph.x[:, 2:4] + acceleration
    velocity = torch.clamp(velocity, -max_velocity, max_velocity)
    return velocity

def update_positions(graph, velocity):
    '''Updates the position of the nodes given the velocity and previous positions'''
    positions = graph.x[:, :2] + velocity
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
    cell_edge_indices = torch.nonzero(graph.edge_attr[:, 1] == 1).flatten()
    zero_edge_mask = torch.bincount(graph.edge_index[0, cell_edge_indices], minlength=graph.x.shape[0]) < edges_to_stay_alive
    mask = torch.bitwise_and(c_mask, zero_edge_mask)
    return mask
