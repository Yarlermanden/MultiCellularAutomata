import torch
from generator import generate_food
import random

def add_edges(graph, radius, device):
    '''Add edges dynamically according to radius. '''
    edges = []
    edge_attributes = []
    radius_food = radius*5
    
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
            edges.append([j, i])
            edge_attribute = [dist, cell_to_cell]
            edge_attributes.append(edge_attribute)
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
    graph.edge_index = torch.tensor(edges, dtype=torch.long, device=device).T
    graph.edge_attr = torch.tensor(edge_attributes, device=device)
    return True

def add_food(graph, food):
    '''Add food source as node to graph'''
    graph.x = torch.cat((graph.x, food))

def add_random_food(graph, device, n=1):
    '''Add n random food sources as nodes to the graph'''
    for _ in range(n):
        food = generate_food(device)
        add_food(graph, food)

def consume_food(graph, food_index, energy_required):
    '''Consumes the food source and convert it to regular cell-node'''
    graph.attr[0] += 1

    if graph.attr[0] == energy_required:
        graph.attr[0] -= energy_required
        graph.x[food_index, 4] = 1
    else: #remove
        graph.x = torch.cat((graph.x[:food_index], graph.x[food_index+1:]))