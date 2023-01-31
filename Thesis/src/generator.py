from cell import Cell
from organism import Organism
import random

#generate random organism
def generate_organism(n: int, device):
    cells = []
    for i in range(n):
        x = random.uniform(0, 0.1)
        y = random.uniform(0, 0.1)
        cell = Cell([x,y])
        cells.append(cell)
    organism = Organism(cells, device)
    return organism


#generate random food as a node
def generate_food():
    ...

