from enums import *

class Food_Env(object):
    def __init__(self, food_amount, wall_amount, env_type):
        self.food_amount = food_amount
        self.wall_amount = wall_amount
        self.env_type = env_type
        #TODO std??

class Cluster_Env(Food_Env):
    def __init__(self, clusters, cluster_size, wall_amount=0):
        self.clusters = clusters
        self.cluster_size = cluster_size
        super().__init__(clusters*cluster_size, wall_amount, EnvironmentType.Clusters)

class Centered_Env(Food_Env):
    def __init__(self, food_amount, wall_amount=0):
        super().__init__(food_amount, wall_amount, EnvironmentType.Centered)

class Circle_Env(Food_Env):
    def __init__(self, circles, food_amount, wall_amount=0):
        self.circles = circles #number of circles
        super().__init__(food_amount, wall_amount, EnvironmentType.Circular)

class Spiral_Env(Food_Env):
    def __init__(self, spirals, food_amount, wall_amount=0):
        self.spirals = spirals
        super().__init__(food_amount, wall_amount, EnvironmentType.Spiral)

class Labyrinth_Env(Food_Env):
    #Testing environment to see if it can follow food and avoid walls
    #have a few places without food but still traped within walls and see if it can go forward to more food
    def __init__(self, food_amount, wall_amount=0):
        super().__init__(food_amount, wall_amount, EnvironmentType.Labyrinth)

class Bottleneck_Env(Food_Env):
    def __init__(self, food_amount, wall_amount=0):
        super().__init__(food_amount, wall_amount, EnvironmentType.Bottleneck)

class Box_Env(Food_Env):
    def __init__(self, food_amount, wall_amount=0):
        super().__init__(food_amount, wall_amount, EnvironmentType.Box)