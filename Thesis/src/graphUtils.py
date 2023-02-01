import torch
from generator import generate_food

def add_edges(graph, radius, device):
    '''Add edges dynamically according to radius. '''
    edges = []
    radius_food = radius*2

    n = len(graph.x)
    for i in range(n):
        isFood_i = graph.x[i, 4] == 0
        for j in range(i, n):
            isFood_j = graph.x[j, 4] == 0
            if isFood_i and isFood_j: 
                continue #two food sources cannot have edges to each other
            radius_to_use = radius
            if isFood_i or isFood_j:
                radius_to_use = radius_food #edge to food has longer radius
            distX = graph.x[i][0] - graph.x[j][0]
            distY = graph.x[i][1] - graph.x[j][1]
            dist = (distX**2 + distY**2)**0.5
            if dist < radius_to_use:
                edges.append([i, j])
                edges.append([j, i])
    if len(edges) == 0:
        return
    graph.edge_index = torch.tensor(edges, dtype=torch.long, device=device).T

def add_food(graph, food):
    '''Add food source as node to graph'''
    graph.x = torch.cat((graph.x, food))

def add_random_food(graph, n=1):
    '''Add n random food sources as nodes to the graph'''
    for i in range(n):
        food = generate_food()
        add_food(graph, food)

#consume food
def consume_food(food):
    '''Consumes the food source and convert it to regular cell-node'''
    food[4] = 1
    #also need to add new food source... - maybe not in here

