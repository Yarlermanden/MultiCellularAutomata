from enums import *

class Train_Config(object):
    def __init__(self, stdev, popsize, name, problem_searcher, with_samplepool, timesteps):
        self.stdev = stdev
        self.popsize = popsize
        self.name = name
        self.problem_searcher = problem_searcher
        self.with_samplepool = with_samplepool
        self.timesteps = timesteps

class Settings(object):
    def __init__(self, device, batch_size, n, food_env, scale, wrap_around, 
                 model_type, radius, radius_food_scale, consume_radius_scale,
                 consumption_edge_required, noise, energy_required_to_replicate, 
                 train_config, radius_wall_scale, radius_wall_damage_scale,
                 wall_damage, n2=None, radius_long_scale=2):
        self.train_config = train_config

        self.device = device
        self.batch_size = batch_size
        self.n = n #amount of cells
        self.n2 = n2
        if self.n2 is None:
            self.n2 = self.n // 10
        self.food_env = food_env #the environment used for food
        self.cluster_std = 0 #TODO

        self.scale = scale #environment scale
        self.wrap_around = wrap_around #whether the world wraps around

        self.model_type = model_type

        #Model functionality parameters - could be varied for testing robustness
        self.radius = radius #radius of cell to cell communication
        self.radius_food = radius*radius_food_scale #radius of cell to food communication
        self.consume_radius = radius*consume_radius_scale #radius of cell to consume food
        self.radius_long = radius*radius_long_scale #radius of special cell nodes to see other cells
        self.radius_wall = radius*radius_wall_scale #the radius for a cell to observe a wall
        self.radius_wall_damage = radius*radius_wall_damage_scale #the radius for a wall to effect a cell
        self.wall_damage = wall_damage #the amount of energy a wall reduces a cell
        self.consumption_edge_required = consumption_edge_required #amount of edges required to consume food
        self.noise = noise #amount of random noise added to cells at each time step
        self.energy_required_to_replicate = energy_required_to_replicate #required energy for splitting/replicating