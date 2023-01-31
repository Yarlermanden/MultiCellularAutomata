import torch

def add_edges(graph, radius):
    '''Add edges dynamically according to radius. '''
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

    graph.edge_index = torch.tensor(edges).T
    return graph

#consume food
def consume_food(graph, food):
    #convert food to regular cell-node
    ...

    #also need to add new food source... - maybe not in here

