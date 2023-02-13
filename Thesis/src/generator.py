from cell import Cell
from organism import Organism
import random
import torch

#generate random organism
def generate_organism(n: int, device):
    cells = []
    for i in range(n):
        x = random.uniform(-0.1, 0.1)
        y = random.uniform(-0.1, 0.1)
        cell = Cell([x,y])
        cells.append(cell)
    organism = Organism(cells, device)
    return organism


#generate random food as a node
def generate_food(device):
    x = random.uniform(-1, 1)
    y = random.uniform(-1, 1)
    food = torch.tensor([[x,y, 0, 0, 0, 0]], device=device)
    return food