import torch
from generator import generate_food

def add_edges(graph, radius, device):
    '''Add edges dynamically according to radius. '''
    #TODO add so the radius is larger if one of them is food source
    edges = []

    #maybe just return adjacency matrix of edges
    n = len(graph.x)

    for i in range(n):
        for j in range(i, n):
            distX = graph.x[i][0] - graph.x[j][0]
            distY = graph.x[i][1] - graph.x[j][1]
            dist = (distX**2 + distY**2)**0.5
            if dist < radius:
                edges.append([i, j])
                edges.append([j, i])

    graph.edge_index = torch.tensor(edges, dtype=torch.long, device=device).T
    return graph

def add_food(graph, food):
    '''Add food source as node to graph'''
    graph.x = torch.cat((graph.x, food))

def add_random_food(graph, n=1):
    '''Add n random food sources as nodes to the graph'''
    for i in range(n):
        food = generate_food()
        add_food(graph, food)

#consume food
def consume_food(graph, food):
    '''Consumes the food source and convert it to regular cell-node'''
    food[4] = 1
    #also need to add new food source... - maybe not in here

