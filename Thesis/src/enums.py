from enum import Enum, IntEnum

class EnvironmentType(Enum):
    Centered = 1, #random noise around center
    Clusters = 2, #clusters of food
    Circular = 3, #with circles
    Spiral = 4, #with spirals
    Labyrinth = 5, 
    Bottleneck = 6,
    Box = 7,

class ModelType(Enum):
    LocalOnly = 1, #only normal cell nodes
    WithGlobalNode = 2, #including global cell node
    SmallWorld = 3, #some cells with longer edges
    LocalMedium = 4, #local but with longer cell radius
    LocalLarge = 5, #Local but with large cell radius - global
    Global = 6, #completely global - all cells have edges to all cells

class ProblemSearcher(Enum):
    CMAES = 0, #distribution based
    GeneticAlgorithm = 1, #population based

class NodeType(IntEnum):
    Food = 0, #regular food node
    Cell = 1 #regular cell node
    GlobalCell = 2, #cell node connected with all other cells
    LongRadiusCell = 3, #cell node with longer radius for cell connections
    Wall = 4, #wall/toxic node

class EdgeType(IntEnum):
    FoodToCell = 0,
    CellToCell = 1,
    GlobalAndCell = 2,
    #CellToGlobal = 3, #TODO maybe combine with the other one
    WallToCell = 4,
