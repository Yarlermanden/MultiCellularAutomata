class Organism():
    def __init__(self):
        self.cells = []

    def toGraph(self):
        #transforms all cells to nodes in a graph

        #each node consists of: posX, posY, velX, velY, isCell(1)
        ...

    def fromGraph(self, graph):
        #transforms all nodes in graph to cells
        ...