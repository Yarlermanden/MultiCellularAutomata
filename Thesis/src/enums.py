from enum import Enum

class EnvironmentType(Enum):
    Centered = 1, #random noise around center
    Clusters = 2, #clusters of food
    WithWalls = 3, #some environment with walls 

class ModelType(Enum):
    LocalOnly = 1, #only normal cell nodes
    WithGlobalNode = 2, #including global cell node
    SmallWorld = 3, #some cells with longer edges
    AI = 4, #hardcoded