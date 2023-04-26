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

def add_global_node(graph, device):
    '''Adds a global node to the graph. 
    Call this before creating batches to ensure a global node exists in all batches'''
    hidden = [0,0,0,0,0]
    global_node = torch.tensor([[0, 0, 0, 0, NodeType.GlobalCell, 0, *hidden]], dtype=torch.float, device=device)
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

def food_mask(nodes):
    '''Used to mask away everything but the food nodes'''
    return nodes[:, 4] == NodeType.Food

def cell_mask(nodes):
    '''Used to mask away everything bu the cell nodes'''
    return torch.bitwise_or(nodes[:, 4] == NodeType.Cell, nodes[:, 4] == NodeType.LongRadiusCell)

def wall_mask(nodes):
    '''Used to mask away everything but the wall nodes'''
    return nodes[:, 4] == NodeType.Wall

def get_consume_food_mask(graph, consume_radius, consumption_edge_required):
    '''Consumes food if criteria is met and returns reward'''
    f_mask = food_mask(graph.x)
    food_val = graph.x[:, 2]
    edge_below_distance = torch.nonzero(graph.edge_attr[:, 0] < consume_radius).flatten() #TODO should be able to optimize this the same way as with walls - to don't bin count all other type of edges
    edges_pr_node = torch.bincount(graph.edge_index[0, edge_below_distance], minlength=graph.x.shape[0])
    edge_mask = edges_pr_node >= food_val
    consumption_mask = torch.bitwise_and(f_mask, edge_mask)
    return consumption_mask

def wall_damage(graph, damage_radius, damage):
    '''Finds cells within damage radius of a wall and reduces their energy accordingly'''
    w_edges = graph.edge_attr[:, 3] == 4
    edge_below_distance = graph.edge_attr[w_edges, 0] < damage_radius
    if len(edge_below_distance) > 0:
        #cell_indices = graph.edge_index[1, edge_below_distance]
        cell_indices = graph.edge_index[1, w_edges][edge_below_distance]
        cell_indices = torch.unique(cell_indices, dim=0)

        cell_edge_indices = torch.nonzero(graph.edge_attr[:, 3] == 1).flatten()
        cell_edge_count = torch.bincount(graph.edge_index[0, cell_edge_indices], minlength=graph.x.shape[0])
        graph.x[cell_indices, 5] -= damage / cell_edge_count[cell_indices]

def cell_degree(graph, radius):
    c_mask = cell_mask(graph.x)
    edge_below_distance = torch.nonzero(graph.edge_attr[:, 0] < radius).flatten() #TODO should be able to optimize this the same way as with walls - to don't bin count all other type of edges
    degree = torch.bincount(graph.edge_index[1, edge_below_distance], minlength=graph.x.shape[0])
    return degree[c_mask]

def get_island_cells_mask(graph, edges_to_stay_alive):
    '''Return mask of cells with less than required amount of edges'''
    c_mask = cell_mask(graph.x)
    cell_edge_indices = torch.nonzero(graph.edge_attr[:, 3] == 1).flatten()
    too_few_edges_mask = torch.bincount(graph.edge_index[0, cell_edge_indices], minlength=graph.x.shape[0]) < edges_to_stay_alive
    mask = torch.bitwise_and(c_mask, too_few_edges_mask)
    return mask

def get_dead_cells_mask(graph, energy_required):
    '''Returns mask of cells with less than required energy level'''
    c_mask = cell_mask(graph.x)
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