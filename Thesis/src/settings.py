from enums import *

class Settings(object):
    def __init__(self, device, batch_size, n, food_env, scale, wrap_around, 
                 model_type, radius, radius_food_scale, consume_radius_scale,
                 consumption_edge_required, noise, energy_required_to_replicate):
        self.device = device
        self.batch_size = batch_size
        self.n = n #amount of cells
        self.food_env = food_env #the environment used for food
        self.cluster_std = 0 #TODO

        self.scale = scale #environment scale
        self.wrap_around = wrap_around #whether the world wraps around

        self.model_type = model_type

        #Model functionality parameters - could be varied for testing robustness
        self.radius = radius #radius of cell to cell communication
        self.radius_food = radius*radius_food_scale #radius of cell to food communication
        self.consume_radius = radius*consume_radius_scale #radius of cell to consume food
        self.consumption_edge_required = consumption_edge_required #amount of edges required to consume food
        self.noise = noise #amount of random noise added to cells at each time step
        self.energy_required_to_replicate = energy_required_to_replicate #required energy for splitting/replicating