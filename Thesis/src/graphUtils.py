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
        food = generate_random_food(settings.device, settings.scale)
        add_food(graph, food)
    for _ in range(food_env.wall_amount):
        wall = generate_random_food_universal(settings.device, settings.scale, d=0.6)
        wall[0, 4] = NodeType.Wall
        add_food(graph, wall)

def add_walls(graph, settings):
    amount = (settings.scale*2) // settings.radius_wall_damage
    x = np.linspace(-settings.scale, settings.scale, int(amount))
    zeros = np.zeros_like(x)
    y0 = zeros.copy()-settings.scale
    y1 = zeros.copy()+settings.scale

    walls = [generate_food(settings.device) for _ in range(int(amount))]
    walls = torch.stack(walls).squeeze()
    walls[:, 4] = NodeType.Wall

    x0_walls = walls.clone()
    x1_walls = walls.clone()
    y0_walls = walls.clone()
    y1_walls = walls.clone()

    x0_walls[:, :2] = torch.tensor([x, y0], device=settings.device).transpose(0, 1)
    x1_walls[:, :2] = torch.tensor([x, y1], device=settings.device).transpose(0, 1)
    y0_walls[:, :2] = torch.tensor([y0, x], device=settings.device).transpose(0, 1)
    y1_walls[:, :2] = torch.tensor([y1, x], device=settings.device).transpose(0, 1)

    walls = torch.cat((x0_walls, x1_walls, y0_walls, y1_walls), dim=0)
    graph.x = torch.cat((graph.x, walls), dim=0)

def add_clusters_of_food(graph, settings, food_env):
    '''Generates and adds n clusters of food to the graph'''
    for _ in range(food_env.clusters):
        cluster = generate_random_cluster(settings.device, food_env.cluster_size, 0.025, settings.scale)
        add_food(graph, cluster)
    for _ in range(food_env.wall_amount):
        wall = generate_random_food_universal(settings.device, settings.scale, d=0.8)
        wall[0, 4] = NodeType.Wall
        add_food(graph, wall)
    #add_walls(graph, settings)

def add_circular_food(graph, settings, food_env):
    for _ in range(food_env.food_amount):
        food = generate_circular_food(settings.device, settings.scale, std_dev=0, circles=food_env.circles) #TODO implement with std... to randomize around circle
        add_food(graph, food)
    for _ in range(food_env.wall_amount):
        wall = generate_random_food_universal(settings.device, settings.scale, d=0.8)
        wall[0, 4] = NodeType.Wall
        add_food(graph, wall)
    #add_walls(graph, settings)

    #abnormal_walls = food_env.wall_amount//10
    #for _ in range(food_env.wall_amount-abnormal_walls):
    #    wall = generate_circular_food(settings.device, settings.scale, std_dev=0, circles=food_env.circles, a=0.5)
    #    wall[0, 4] = NodeType.Wall
    #    add_food(graph, wall)
    #for _ in range(abnormal_walls):
    #    wall = generate_circular_food(settings.device, settings.scale, std_dev=0, circles=food_env.circles) #TODO implement with std... to randomize around circle
    #    wall[0, 4] = NodeType.Wall
    #    add_food(graph, wall)

def add_spiral_food(graph, settings, food_env):
    for _ in range(food_env.food_amount):
        food = generate_spiral_food(settings.device, settings.scale, std_dev=0, spirals=food_env.spirals)
        add_food(graph, food)
    abnormal_walls = food_env.wall_amount//20
    walls = generate_spiral_walls(settings.device, settings.scale, food_env.wall_amount-abnormal_walls, spirals=food_env.spirals)
    graph.x = torch.cat((graph.x, walls))
    for _ in range(abnormal_walls):
        wall = generate_spiral_food(settings.device, settings.scale, std_dev=0, spirals=food_env.spirals)
        wall[0, 4] = NodeType.Wall
        add_food(graph, wall)
    #add_walls(graph, settings)

def add_labyrinth_food(graph, settings, food_env):
    x_points = np.linspace(-0.8*settings.scale, 0.8*settings.scale, (food_env.grid_size*3)//2)
    y_points = np.linspace(-0.8*settings.scale, 0.8*settings.scale, food_env.grid_size)
    for i, x in enumerate(x_points):
        for j, y in enumerate(y_points):
            walls = generate_half_circle_wall(settings.device, x, y)
            add_food(graph, walls)
            cluster = generate_cluster(settings.device, food_env.cluster_size, 0.015, settings.scale, x, y)
            add_food(graph, cluster)
    #add_walls(graph, settings)

def add_bottleneck_food(graph, settings, food_env):
    for _ in range(food_env.food_amount):
        food = generate_bottleneck_food(settings.device, settings.scale)
        add_food(graph, food)
    walls = generate_bottleneck_walls(settings.device, settings.scale, food_env.wall_amount)
    add_food(graph, walls)
    #add_walls(graph, settings)

def add_box_food(graph, settings, food_env):
    #TODO
    ...

def add_grid_food(graph, settings, food_env):
    x_points = np.linspace(-0.8*settings.scale, 0.8*settings.scale, food_env.grid_size)
    y_points = np.linspace(-0.8*settings.scale, 0.8*settings.scale, food_env.grid_size)
    for i, x in enumerate(x_points):
        for j, y in enumerate(y_points):
            #even = ((i**2+j**2) % 4) == 0
            even = i % 2 == 1 and (i+j) % 2 == 0
            if even:
                cluster = generate_cluster(settings.device, food_env.cluster_size, 0.010, settings.scale, x, y)
                add_food(graph, cluster)
            else:
                wall = generate_food(settings.device, x, y)
                wall[0, 4] = NodeType.Wall
                add_food(graph, wall)
    #add_walls(graph, settings)

def add_food_grid_food(graph, settings, food_env):
    x_points = np.linspace(-0.8*settings.scale, 0.8*settings.scale, food_env.grid_size)
    y_points = np.linspace(-0.8*settings.scale, 0.8*settings.scale, food_env.grid_size)
    for i, x in enumerate(x_points):
        for j, y in enumerate(y_points):
            cluster = generate_cluster(settings.device, food_env.cluster_size, 0.015, settings.scale, x, y)
            add_food(graph, cluster)
    #add_walls(graph, settings)

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
        cell_edge_count = torch.bincount(graph.edge_index[1, cell_edge_indices], minlength=graph.x.shape[0])
        graph.x[cell_indices, 5] -= settings.wall_damage / torch.clamp(cell_edge_count[cell_indices], min=1, max=settings.max_degree)

def degree_below_radius(graph, settings):
    c_edges_below_distance = torch.bitwise_and(graph.edge_attr[:, 3] == EdgeType.CellToCell, graph.edge_attr[:, 0] < settings.radius)
    edge_below_distance = torch.nonzero(c_edges_below_distance).flatten()
    degree = torch.bincount(graph.edge_index[1, edge_below_distance], minlength=graph.x.shape[0])
    return torch.clamp(degree, min=1, max=settings.max_degree)

def get_dead_cells_mask(graph, energy_required):
    '''Returns mask of cells with less than required energy level'''
    c_mask = cell_mask(graph.x)
    e_mask = graph.x[:, 5] < energy_required
    mask = torch.bitwise_and(c_mask, e_mask)
    return mask

def breed(graph, settings):
    '''Creates new cells if enough energy'''
    s_idx = 0
    for i in range(settings.batch_size):
        e_idx = s_idx + graph.subsize[i]
        c_mask = cell_mask(graph.x[s_idx:e_idx])
        enough_energy_mask = graph.x[s_idx:e_idx, 5] >= settings.energy_required_to_replicate
        breeding_mask = torch.nonzero(torch.bitwise_and(c_mask, enough_energy_mask)) + s_idx
        if breeding_mask.any():
            graph.x[breeding_mask, 5] -= settings.energy_required_to_replicate // 2
            new_cells = graph.x[breeding_mask].clone().view(-1, graph.x.shape[1])
            new_cells_survive = torch.rand(new_cells[:,0].shape) > 0.7
            new_cells = new_cells[new_cells_survive].view(-1, graph.x.shape[1])
            if c_mask.sum() < 1.5*settings.n and new_cells.any():
                x_noise = (torch.rand(new_cells[:, 0].shape, device=settings.device)*2-1.0) * settings.noise
                y_noise = (torch.rand(new_cells[:, 1].shape, device=settings.device)*2-1.0) * settings.noise
                new_cells[:, 0] += x_noise
                new_cells[:, 1] += y_noise
                graph.subsize[i] += new_cells.shape[0]
                graph.x = torch.cat((graph.x[:e_idx], new_cells, graph.x[e_idx:]), dim=0)
                e_idx += new_cells.shape[0]
        s_idx = e_idx

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