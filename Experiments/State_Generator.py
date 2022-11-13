import torch

class State():
    def __init__(self, x, y, food):
        self.x = x #input | initial CA state
        self.y = y #target | CA state after i iterations
        self.food = food #food 

class Generator():
    def __init__(self, device, random_states):
        self.width = 16
        self.depth = 4
        self.device = device
        self.random_states = random_states

    def generate_moving_state(self, timesteps):
        #Generate the new state/x
        #generate the food source determining the direciton goal
        #generate the target from epoch and x and food
        return ...


    def generate_stationary_state(self):
        zeros = torch.zeros(self.width, self.width)
        if self.random_states:
            cell_state = torch.rand(5, 5, device=self.device) > 0.7
        else:
            cell_state = torch.tensor([
                [0, 0, 1, 0, 0],
                [0, 1, 1, 1, 0],
                [1, 1, 1, 1, 1],
                [0, 1, 1, 1, 0],
                [0, 0, 1, 0, 0]
            ])
        cell = zeros
        cell[6:11, 6:11] = cell_state
        state = torch.stack([cell, zeros, zeros, zeros])
        return State(state, state, zeros)