from cell import Cell
from organism import Organism
import random
import numpy as np
import torch
import math
from enums import *

def get_random_point_within(d):
    return random.uniform(-d, d), random.uniform(-d,d)

def get_random_point_normal(center, std_dev):
    return np.random.normal(center, std_dev), np.random.normal(center, std_dev)

def generate_organism(settings):
    '''Generate a random centered organism'''
    cells = []
    d = 0.04
    d = d*2 if settings.n > 200 else d
    for i in range(settings.n):
        x,y = get_random_point_within(d)
        cell = Cell([x,y])
        cells.append(cell)
    organism = Organism(cells, settings)
    return organism

def generate_food(device, scale, d=0.3):
    '''Generate a random food'''
    #TODO consider making this not able in spawning food right in the center...
    x,y = get_random_point_normal(0, d*scale)
    val = random.randint(1,3)
    hidden = [0,0,0,0,0]
    food = torch.tensor([[x,y, val, 0, NodeType.Food, 0, *hidden]], device=device)
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

def reverse_sum(num):
    '''Reverse of Kth Triangle number function'''
    n = 0
    i = 0
    while n < num:
        i += 1
        n += i
    return i

def sum(num):
    '''Kth Triangle number function'''
    n = 0
    for i in range(num):
        n += i
    return n

def generate_circular_food(device, scale, std_dev, circles, a=0):
    '''Generates food in circular patterns given std_dev, number of circles and radius'''
    s = sum(circles+1)
    s1 = random.randint(1, s)
    s2 = reverse_sum(s1) / circles
    #r = (a*scale/circles) + scale * (math.sqrt(random.uniform(0.8, 1)) * s2)
    r = (a*scale/circles) + scale * s2
    theta = random.uniform(0,1) * 2 * math.pi
    noise1 = random.uniform(-0.05, 0.05)
    noise2 = random.uniform(-0.05, 0.05)
    x = r * math.cos(theta) + noise1
    y = r * math.sin(theta) + noise2
    food = generate_food(device, scale)
    food[:, :2] = torch.tensor([x, y], device=device)
    return food

def generate_spiral_food(device, scale, std_dev, spirals, rotation=0):
    '''Generates food in a spiral pattern'''
    #TODO add random noise, which should be bigger the further out we are...
    b = 1 / (2 * np.pi)
    theta1 = math.sqrt(random.uniform(0, math.pow(spirals * 2 * np.pi, 2))) #combination of sqrt and pow increases odds of rolling high numbers
    r = b * theta1 * scale/spirals
    noise1 = random.uniform(-0.05, 0.05)
    noise2 = random.uniform(-0.05, 0.05)
    x = r * (np.cos(theta1+rotation) + noise1)
    y = r * (np.sin(theta1+rotation) + noise2)

    food = generate_food(device, scale)
    food[:, :2] = torch.tensor([x, y], device=device)
    return food

def generate_spiral_walls(device, scale, wall_amount, spirals, rotation):
    b = 1 / (2 * np.pi)
    thetas = np.linspace(0, spirals * 2 * np.pi, wall_amount)
    r = [(0.1*scale) + b * t * scale/spirals for t in thetas]
    x = [r[i] * np.cos(thetas[i]+rotation+144) for i in range(wall_amount)]
    y = [r[i] * np.sin(thetas[i]+rotation+144) for i in range(wall_amount)]

    walls = [generate_food(device, scale) for _ in range(wall_amount)]
    walls = torch.stack(walls).squeeze()
    walls[:, :2] = torch.stack((torch.tensor(x, device=device), torch.tensor(y, device=device)), dim=1)
    walls[:, 4] = NodeType.Wall
    return walls