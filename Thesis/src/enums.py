from enum import Enum

class EnvironmentType(Enum):
    Centered = 1, #random noise around center
    Clusters = 2, #clusters of food
    Circular = 3, #with circles
    Spiral = 4, #with spirals
    #TODO could even add a type hardcoded with food in spiral and walls around...
    WithWalls = 5, #some environment with walls 

class ModelType(Enum):
    LocalOnly = 1, #only normal cell nodes
    WithGlobalNode = 2, #including global cell node
    SmallWorld = 3, #some cells with longer edges
    AI = 4, #hardcoded

class ProblemSearcher(Enum):
    CMAES = 0, #distribution based
    GeneticAlgorithm = 1, #population based