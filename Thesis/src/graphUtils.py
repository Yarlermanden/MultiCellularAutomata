import torch
from generator import *
import random
import torch_geometric
from torch_geometric import utils
from enums import *

def add_food(graph, food):
    '''Add food source as node to graph'''
    graph.x = torch.cat((graph.x, food))

def add_random_food(graph, settings, food_env):
    '''Add n random food sources as nodes to the graph'''
    for _ in range(food_env.food_amount):
        food = generate_food(settings.device, settings.scale)
        add_food(graph, food)
    for _ in range(food_env.wall_amount):
        wall = generate_food(settings.device, settings.scale)
        wall[0, 4] = NodeType.Wall
        add_food(graph, wall)

def add_clusters_of_food(graph, settings, food_env):
    '''Generates and adds n clusters of food to the graph'''
    for _ in range(food_env.clusters):
        cluster = generate_cluster(settings.device, food_env.cluster_size, 0.04, settings.scale)
        add_food(graph, cluster)
    for _ in range(food_env.wall_amount):
        wall = generate_food(settings.device, settings.scale, d=0.4)
        wall[0, 4] = NodeType.Wall
        add_food(graph, wall)

def add_circular_food(graph, settings, food_env):
    for _ in range(food_env.food_amount):
        food = generate_circular_food(settings.device, settings.scale, std_dev=0, circles=food_env.circles) #TODO implement with std... to randomize around circle
        add_food(graph, food)
    for _ in range(food_env.wall_amount):
        wall = generate_circular_food(settings.device, settings.scale, std_dev=0, circles=food_env.circles, a=0.5)
        wall[0, 4] = NodeType.Wall
        add_food(graph, wall)

def add_spiral_food(graph, settings, food_env):
    rotation = random.uniform(0, 2*np.pi)
    for _ in range(food_env.food_amount):
        food = generate_spiral_food(settings.device, settings.scale, std_dev=0, spirals=food_env.spirals, rotation=rotation)
        add_food(graph, food)
    walls = generate_spiral_walls(settings.device, settings.scale, food_env.wall_amount, spirals=food_env.spirals, rotation=rotation)
    graph.x = torch.cat((graph.x, walls))

def add_labyrinth_food(graph, settings, food_env):
    #TODO
    ...

def add_bottleneck_food(graph, settings, food_env):
    #TODO
    ...

def add_box_food(graph, settings, food_env):
    #TODO
    ...

def add_global_node(graph, device):
    '''Adds a global node to the graph. 
    Call this before creating batches to ensure a global node exists in all batches'''
    hidden = [0,0,0,0,0]
    global_node = torch.tensor([[0, 0, 0, 0, NodeType.GlobalCell, 0, *hidden]], dtype=torch.float, device=device)
    graph.x = torch.cat((graph.x, global_node))

def update_positions(graph, velocity, wrap_around, c_mask, scale):
    '''Updates the position of the nodes given the velocity and previous positions'''
    if wrap_around:
        positions = torch.remainder(graph.x[c_mask, :2] + velocity[c_mask] + scale, 2*scale) - scale
    else:
        positions = graph.x[c_mask, :2] + velocity[c_mask]
    return positions

def food_mask(nodes):
    '''Used to mask away everything but the food nodes'''
    return nodes[:, 4] == NodeType.Food

def cell_mask(nodes):
    '''Used to mask away everything bu the cell nodes'''
    return torch.bitwise_or(nodes[:, 4] == NodeType.Cell, nodes[:, 4] == NodeType.LongRadiusCell)

def wall_mask(nodes):
    '''Used to mask away everything but the wall nodes'''
    return nodes[:, 4] == NodeType.Wall

def get_consume_food_mask(graph, consume_radius):
    '''Consumes food if criteria is met and returns reward'''
    f_mask = food_mask(graph.x)
    food_val = graph.x[:, 5]
    edge_below_distance = torch.nonzero(graph.edge_attr[:, 0] < consume_radius).flatten() #TODO should be able to optimize this the same way as with walls - to don't bin count all other type of edges
    edges_pr_node = torch.bincount(graph.edge_index[0, edge_below_distance], minlength=graph.x.shape[0])
    edge_mask = edges_pr_node >= food_val
    consumption_mask = torch.bitwise_and(f_mask, edge_mask)
    return consumption_mask

def wall_damage(graph, settings):
    '''Finds cells within damage radius of a wall and reduces their energy accordingly'''
    w_edges = graph.edge_attr[:, 3] == EdgeType.WallToCell
    w_edge_below_distance = graph.edge_attr[w_edges, 0] < settings.radius_wall_damage
    if len(w_edge_below_distance) > 0:
        #cell_indices = graph.edge_index[1, edge_below_distance]
        cell_indices = graph.edge_index[1, w_edges][w_edge_below_distance]
        cell_indices = torch.unique(cell_indices, dim=0)

        c_edges_below_distance = torch.bitwise_and(graph.edge_attr[:, 3] == EdgeType.CellToCell, graph.edge_attr[:, 0] < settings.radius)
        cell_edge_indices = torch.nonzero(c_edges_below_distance).flatten()
        cell_edge_count = torch.bincount(graph.edge_index[0, cell_edge_indices], minlength=graph.x.shape[0])
        graph.x[cell_indices, 5] -= settings.wall_damage / torch.clamp(cell_edge_count[cell_indices], max=settings.max_degree)

def degree_below_radius(graph, settings):
    edge_below_distance = torch.nonzero(graph.edge_attr[:, 0] < settings.radius).flatten() #TODO should be able to optimize this the same way as with walls - to don't bin count all other type of edges
    degree = torch.bincount(graph.edge_index[1, edge_below_distance], minlength=graph.x.shape[0])
    return torch.clamp(degree, max=settings.max_degree)

def get_dead_cells_mask(graph, energy_required):
    '''Returns mask of cells with less than required energy level'''
    c_mask = cell_mask(graph.x)
    e_mask = graph.x[:, 5] < energy_required
    mask = torch.bitwise_and(c_mask, e_mask)
    return mask

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