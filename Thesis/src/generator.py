from cell import Cell
from organism import Organism
import random
import torch

def generate_organism(n: int, device):
    '''Generate a random centered organism'''
    cells = []
    d = 0.04
    for i in range(n):
        x = random.uniform(-d, d)
        y = random.uniform(-d, d)
        cell = Cell([x,y])
        cells.append(cell)
    organism = Organism(cells, device)
    return organism

def generate_food(device):
    '''Generate a random food'''
    d = 0.4
    x = random.uniform(-d, d)
    y = random.uniform(-d, d)
    #food = torch.tensor([[x,y, 0, 0, 0, 0, 0, 0, 0, 0]], device=device)
    val = random.randint(1,3)
    food = torch.tensor([[x,y, val, 0, 0]], device=device)
    return food