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

def generate_food(device, scale, d=0.2):
    '''Generate a random food'''
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

def generate_circular_food(device, scale, std_dev, circles, radius):
    '''Generates food in circular patterns given std_dev, number of circles and radius'''
    s = sum(circles+1)
    s1 = random.randint(1, s)
    s2 = reverse_sum(s1) / circles
    r = radius * math.sqrt(random.uniform(0.8, 1)) * s2 * 0.40 * scale
    theta = random.uniform(0,1) * 2 * math.pi
    x = r * math.cos(theta)
    y = r * math.sin(theta)
    food = generate_food(device, scale)
    food[:, :2] = torch.tensor([x, y], device=device)
    return food

def generate_spiral_food(device, scale, std_dev, spirals, rotation=0, a=0):
    '''Generates food in a spiral pattern'''
    #TODO add random noise, which should be bigger the further out we are...
    b = 1 / (2 * np.pi)
    theta1 = math.sqrt(random.uniform(0, math.pow(spirals * 2 * np.pi, 2))) #combination of sqrt and pow increases odds of rolling high numbers
    r = (a*scale) + b * theta1 * scale/spirals
    x = r * np.cos(theta1+rotation)
    y = r * np.sin(theta1+rotation)

    food = generate_food(device, scale)
    food[:, :2] = torch.tensor([x, y], device=device)
    return food