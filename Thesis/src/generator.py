from cell import Cell
from organism import Organism
import random
import numpy as np
import torch
from enums import EnvironmentType

def get_random_point_within(d):
    return random.uniform(-d, d), random.uniform(-d,d)

def get_random_point_normal(center, std_dev):
    return np.random.normal(center, std_dev), np.random.normal(center, std_dev)

def generate_organism(n: int, device, with_global_node, food_amount, env_type, scale):
    '''Generate a random centered organism'''
    cells = []
    d = 0.04*scale
    d = d*2 if n > 200 else d
    for i in range(n):
        x,y = get_random_point_within(d)
        cell = Cell([x,y])
        cells.append(cell)
    organism = Organism(cells, device, with_global_node, food_amount, env_type, scale)
    return organism

def generate_food(device, scale, d=0.2):
    '''Generate a random food'''
    x,y = get_random_point_normal(0, d*scale)
    val = random.randint(1,3)
    hidden = [0,0,0,0,0]
    food = torch.tensor([[x,y, val, 0, 0, 0, *hidden]], device=device)
    return food

def generate_cluster(device, cluster_size, std_dev, scale):
    '''Generates a cluster of food with certain size and std_dev'''
    d = 0.8*scale
    x,y = get_random_point_within(d)
    cluster = []
    for _ in range(cluster_size):
        food = generate_food(device, scale, std_dev)
        cluster.append(food)
    cluster = torch.concat(cluster, dim=0)
    cluster[:, :2]+=torch.tensor([[x,y]], device=device)
    return cluster
