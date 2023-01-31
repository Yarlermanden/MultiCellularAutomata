from cell import Cell
from organism import Organism
import random

#generate random organism
def generate_organism(n: int):
    cells = []
    for i in range(n):
        x = random.uniform(0, 0.1)
        y = random.uniform(0, 0.1)
        cell = Cell([x,y])
        cells.append(cell)
    organism = Organism(cells)
    return organism


#generate random food as a node
def generate_food():
    ...

