from cell import Cell
from organism import Organism
import random
import numpy as np
import torch
import math
from enums import *

def get_random_point_within(d):
    x = random.uniform(-d,d)
    y = random.uniform(-d,d)
    min_d = d/50
    while(abs(x) < min_d and abs(y) < min_d):
        x = random.uniform(-d,d)
        y = random.uniform(-d,d)
    return x,y

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

def generate_food(device, x=0, y=0):
    val = random.randint(1,4)
    hidden = [0,0,0,0,0]
    food = torch.tensor([[x,y, 0, 0, NodeType.Food, val, *hidden]], device=device, dtype=torch.float)
    return food

def generate_random_food(device, scale, d=0.3):
    '''Generate a random food'''
    #TODO consider making this not able in spawning food right in the center...
    x,y = get_random_point_normal(0, d*scale)
    return generate_food(device, x, y)

def generate_random_food_universal(device, scale, d):
    x,y = get_random_point_within(d*scale)
    return generate_food(device, x, y)

def generate_cluster(device, cluster_size, std_dev, scale, x, y):
    cluster = []
    for _ in range(cluster_size):
        food = generate_random_food(device, scale, std_dev)
        cluster.append(food)
    cluster = torch.concat(cluster, dim=0)
    cluster[:, :2]+=torch.tensor([[x,y]], device=device)
    return cluster

def generate_random_cluster(device, cluster_size, std_dev, scale):
    '''Generates a cluster of food with certain size and std_dev'''
    d = 0.7*scale
    x,y = get_random_point_within(d)
    return generate_cluster(device, cluster_size, std_dev, scale, x, y)

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
    food = generate_food(device, x, y)
    return food

def generate_spiral_food(device, scale, std_dev, spirals):
    '''Generates food in a spiral pattern'''
    #TODO add random noise, which should be bigger the further out we are...
    b = 1 / (2 * np.pi)
    theta1 = math.sqrt(random.uniform(0, math.pow(spirals * 2 * np.pi, 2))) #combination of sqrt and pow increases odds of rolling high numbers
    r = b * theta1 * scale/spirals
    noise1 = random.uniform(-0.05, 0.05)
    noise2 = random.uniform(-0.05, 0.05)
    x = r * (np.cos(theta1) + noise1)
    y = r * (np.sin(theta1) + noise2)

    food = generate_food(device, x, y)
    return food

def generate_spiral_walls(device, scale, wall_amount, spirals):
    b = 1 / (2 * np.pi)
    thetas = np.linspace(0, spirals * 2 * np.pi, wall_amount)
    r = [(0.1*scale) + b * t * scale/spirals for t in thetas]
    x = [r[i] * np.cos(thetas[i]+144) for i in range(wall_amount)]
    y = [r[i] * np.sin(thetas[i]+144) for i in range(wall_amount)]

    walls = [generate_food(device) for _ in range(wall_amount)]
    walls = torch.stack(walls).squeeze()
    walls[:, :2] = torch.stack((torch.tensor(x, device=device), torch.tensor(y, device=device)), dim=1)
    walls[:, 4] = NodeType.Wall
    return walls

def generate_bottleneck_food(device, scale):
    type = random.randint(1, 100)
    if type < 30: #30% on start
        x = random.uniform(-0.35, 0.35) * scale
        y = random.uniform(-0.15, 0.15) * scale
    elif type < 32: #2% in bottleneck
        x = 0
        y = random.uniform(0.15, 0.25) * scale
    else: #65% in goal area
        x = random.uniform(-0.35, 0.35) * scale
        y = random.uniform(0.25, 0.55) * scale
    food = generate_food(device, x, y)
    return food

def generate_bottleneck_walls(device, scale, wall_amount):
    x = np.linspace(-0.35*scale, 0.35*scale, 8*scale)
    x_gap = np.concatenate([np.linspace(-0.35*scale, -0.03*scale, 4*scale), np.linspace(0.03*scale, 0.35*scale, 4*scale)])
    y = np.linspace(-0.15*scale, 0.55*scale, 8*scale)
    
    walls = [generate_food(device) for _ in range(wall_amount)]
    walls = torch.stack(walls).squeeze()
    walls[:, 4] = NodeType.Wall

    bottom_walls_y = np.zeros_like(x)-0.15*scale

    bottom_walls_y2 = np.zeros_like(x)+0.15*scale
    bottom_walls_y3 = np.zeros_like(x)+0.1833*scale
    bottom_walls_y4 = np.zeros_like(x)+0.2166*scale
    top_walls_y = np.zeros_like(x)+0.25*scale

    top_walls_y2 = np.zeros_like(x)+0.55*scale
    
    walls[0:8*scale, :2] = torch.tensor([x, bottom_walls_y], device=device).transpose(0, 1)
    walls[8*scale:16*scale, :2] = torch.tensor([x_gap, bottom_walls_y2], device=device).transpose(0, 1)
    walls[16*scale:24*scale, :2] = torch.tensor([x_gap, bottom_walls_y3], device=device).transpose(0, 1)
    walls[24*scale:32*scale, :2] = torch.tensor([x_gap, bottom_walls_y4], device=device).transpose(0, 1)
    walls[32*scale:40*scale, :2] = torch.tensor([x_gap, top_walls_y], device=device).transpose(0,1)
    walls[40*scale:48*scale, :2] = torch.tensor([x, top_walls_y2], device=device).transpose(0,1)

    left_walls_x = np.zeros_like(y)-0.35*scale
    right_walls_x = np.zeros_like(y)+0.35*scale

    walls[48*scale:56*scale, :2] = torch.tensor([left_walls_x, y], device=device).transpose(0,1)
    walls[56*scale:64*scale, :2] = torch.tensor([right_walls_x, y], device=device).transpose(0,1)

    return walls

def generate_half_circle_wall(device, x, y):
    #generate 5 or 7 walls in half circle
    walls = [generate_food(device) for _ in range(7)]
    walls = torch.stack(walls).squeeze()
    walls[:, 4] = NodeType.Wall
    coords = torch.tensor([[-0.04, -0.2], [0.05, -0.16], [0.12, -0.08], [0.16, 0], [0.12, 0.08], [0.05, 0.16], [-0.04, 0.2]], device=device, dtype = torch.float)
    walls[:, :2] = coords+torch.tensor([x, y], device=device)
    return walls

