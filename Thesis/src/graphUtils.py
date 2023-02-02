import torch
from generator import generate_food
import random

def add_edges(graph, radius, device):
    '''Add edges dynamically according to radius. '''
    edges = []
    edge_attributes = []
    radius_food = radius*3

    #get list of all cell-nodes
    #for each - go through list of all nodes with index above this node and check for whether to add edge
        #this outer loop only iterate all cell-nodes 
        #and inner loop only iterates all nodes above - so only one way
        #this will miss all edges between food and a cell noce with higher index...

    n = len(graph.x)
    for i in range(n):
        isFood_i = graph.x[i, 4] == 0
        for j in range(i+1, n):
            isFood_j = graph.x[j, 4] == 0
            if isFood_i and isFood_j: 
                continue #two food sources cannot have edges to each other
            radius_to_use = radius
            cell_to_cell = 1
            if isFood_i or isFood_j:
                radius_to_use = radius_food #edge to food has longer radius
                cell_to_cell = 0
            distX = graph.x[i][0] - graph.x[j][0]
            distY = graph.x[i][1] - graph.x[j][1]
            dist = (distX**2 + distY**2)**0.5
            if dist < radius_to_use:
                edges.append([i, j])
                edges.append([j, i])
                edge_attribute = [dist, cell_to_cell]
                edge_attributes.append(edge_attribute)
                edge_attributes.append(edge_attribute)
    if len(edges) == 0:
        return False
    graph.edge_index = torch.tensor(edges, dtype=torch.long, device=device).T
    graph.edge_attr = torch.tensor(edge_attributes)
    return True

def add_food(graph, food):
    '''Add food source as node to graph'''
    graph.x = torch.cat((graph.x, food))

def add_random_food(graph, n=1):
    '''Add n random food sources as nodes to the graph'''
    for i in range(n):
        food = generate_food()
        add_food(graph, food)

#consume food
def consume_food(graph, food_index, energy_required=4):
    '''Consumes the food source and convert it to regular cell-node'''
    graph.attr[0] += 1

    if graph.attr[0] == energy_required:
        graph.attr[0] -= energy_required
        graph.x[food_index, 4] = 1
    else: #kill
        graph.x = torch.cat((graph.x[:food_index], graph.x[food_index+1:]))