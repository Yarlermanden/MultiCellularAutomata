from enums import *

class Food_Env(object):
    def __init__(self, food_amount, env_type):
        self.food_amount = food_amount
        self.env_type = env_type
        #TODO std??

class Cluster_Env(Food_Env):
    def __init__(self, clusters, cluster_size):
        self.clusters = clusters
        self.cluster_size = cluster_size
        super().__init__(clusters*cluster_size, EnvironmentType.Clusters)

class Centered_Env(Food_Env):
    def __init__(self, food_amount):
        super().__init__(food_amount, EnvironmentType.Centered)

class Circle_Env(Food_Env):
    def __init__(self, circles, food_amount):
        self.circles = circles #number of circles
        super().__init__(food_amount, EnvironmentType.Circular)

class Spiral_Env(Food_Env):
    def __init__(self, spirals, food_amount):
        self.spirals = spirals
        super().__init__(food_amount, EnvironmentType.Spiral)

class Labyrinth_Env(Food_Env):
    #consist of walls and food in such a pattern that the organism will be able to track through it
    #could even make it so energy isn't lost unless inside a wall
    ...