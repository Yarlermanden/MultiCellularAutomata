from enum import Enum

class EnvironmentType(Enum):
    Centered = 1,
    Clusters = 2,

class ModelType(Enum):
    LocalOnly = 1,
    WithGlobalNode = 2,
    AI = 3,
#potentially version with nodes pr cluster