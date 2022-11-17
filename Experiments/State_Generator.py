import torch
import random
import numpy as np

class State():
    def __init__(self, x, y, food):
        self.x = x #input | initial CA state
        self.y = y #target | CA state after i iterations
        self.food = food #food 

class Generator():
    def __init__(self, device, random_states):
        self.width = 17
        self.depth = 4
        self.device = device
        self.random_states = random_states

    def generate_moving_state(self, timesteps, batch_size):
        #TODO: make faster... - try having it all in numpy until as late as possible

        #TODO some cells should register the food quicker than others - some should wait even longer before starting to move
        #TODO should try to make it less spread out - ensure one entity

        #Generate initial CA
        zeros = torch.zeros(batch_size, self.width, self.width, device=self.device)

        state = self.get_centered_CA(batch_size)
        state = torch.stack([state, zeros, zeros, zeros], 1)

        #Generate food map
        food_coord = self.get_random_food_coord(batch_size)
        food = zeros
        for i in range(len(food)):
            food[i, food_coord[i, 0], food_coord[i, 1]] = 1

        #Generate target CA
        target_ca = state[:, 0]
        for i in range(timesteps):
            if i > 1:
                target_ca = self.move_towards_food(target_ca, food_coord)
        return State(state, target_ca, food)

    def generate_ca_and_food(self, batch_size):
        zeros = torch.zeros(batch_size, self.width, self.width, device=self.device)
        food = self.get_random_food(batch_size)
        ca = self.get_centered_CA(batch_size)
        ca = torch.stack([ca, zeros, zeros, food], 1)
        return ca

    def generate_stationary_state(self, batch_size):
        zeros = torch.zeros(batch_size, self.width, self.width, device=self.device)
        cell = self.get_centered_CA(batch_size)
        state = torch.stack([cell, zeros, zeros, zeros], 1)
        food = zeros
        for i in range(len(food)):
            food[i, self.width//2, self.width//2] = 1
        return State(state, cell, food)

    def get_centered_CA(self, batch_size):
        zeros = torch.zeros(batch_size, self.width, self.width, device=self.device)
        if self.random_states:
            center_ca = torch.rand(5, 5, device=self.device) > 0.7
        else:
            center_ca = torch.tensor([
                [0, 0, 1, 0, 0],
                [0, 1, 1, 1, 0],
                [1, 1, 1, 1, 1],
                [0, 1, 1, 1, 0],
                [0, 0, 1, 0, 0]
            ], device=self.device)
        ca = zeros
        ca[:, 6:11, 6:11] = center_ca
        return ca

    def get_random_food(self, batch_size):
        zeros = torch.zeros(batch_size, self.width, self.width, device=self.device)
        food_coord = self.get_random_food_coord(batch_size)
        food = zeros
        for i in range(len(food)):
            food[i, food_coord[i, 0], food_coord[i, 1]] = 1
        return food

    def get_random_food_coord(self, batch_size): #food can be centered or not
        def random_num():
            from_edge = 4
            return np.random.randint(from_edge, self.width-1-from_edge, batch_size)
        x = random_num()
        y = random_num()
        food = np.stack([x, y], 1)
        return food

    def get_food_coord_from_food(self, food):
        return (food==torch.max(food)).nonzero()[:, 1:]

    def random_food_noncentered(self, batch_size):
        def random_outer_num(middle):
            lower = random.randint(1, middle-5)
            upper = random.randint(middle+5, self.width-2)
            coin_flip = random.randint(0, 1)
            return lower*coin_flip + upper*(1-coin_flip) #returns lower or upper

        middle = self.width // 2
        x = random_outer_num(middle)
        y = random_outer_num(middle)
        return x,y

    def move_towards_food(self, ca, food):
        #TODO: if no food, should stay stationary
        #TODO: if food but not detectable, stay stationary
        #TODO: in case it reaches food, remove food and instead increase cells

        #TODO fix the out of range - due to out of range
        #TODO fix the wraparound failure... - is due to -1

        new_ca = ca.clone()

        #generate random map of entire world with values 0 or 1 for whether to move or not
        move_mask = (torch.rand_like(ca) > 0.3).to(torch.float)

        for b, batch in enumerate(ca): #batch
            for i, row in enumerate(batch):
                for j, val in enumerate(row):
                    if val > 0.1 and move_mask[b, i, j] == 1: #allowed to move
                        #find direction to move in - almost always 3 neighboring cells closer than current
                        delta_x = food[b, 1] - j 
                        delta_y = food[b, 0] - i
                        moved_val = 0.25

                        #fix deltas to be -1, 0 or 1
                        if delta_x > 0:
                            delta_x = 1
                        elif delta_x < 0:
                            delta_x = -1

                        if delta_y > 0:
                            delta_y = 1
                        elif delta_y < 0:
                            delta_y = -1

                        new_ca[b][i][j] -= moved_val #assume it can move
                        if delta_x == 0: #line up in column
                            if new_ca[b][i+delta_y][j] < 0.9: #available
                                new_ca[b][i+delta_y][j] += moved_val
                            elif new_ca[b][i+delta_y][j+1] < 0.9: #priotize right over left...
                                new_ca[b][i+delta_y][j+1] += moved_val
                            elif new_ca[b][i+delta_y][j-1] < 0.9:
                                new_ca[b][i+delta_y][j-1] += moved_val
                            else:
                                #new_ca[b][i][j] = val #couldn't move
                                new_ca[b][i][j] += moved_val #couldn't move

                        elif delta_y == 0: #line up in row
                            if new_ca[b][i][j+delta_x] < 0.9: #most direct way
                                new_ca[b][i][j+delta_x] += moved_val
                            elif new_ca[b][i+1][j+delta_x] < 0.9: #priotize down over up
                                new_ca[b][i+1][j+delta_x] += moved_val
                            elif new_ca[b][i-1][j+delta_x] < 0.9:
                                new_ca[b][i-1][j+delta_x] += moved_val
                            else:
                                #new_ca[b][i][j] = val #couldn't move
                                new_ca[b][i][j] += moved_val #couldn't move

                        #neither lines up
                        #move to the corners available
                        else:
                            if new_ca[b][i+delta_y][j+delta_x] < 0.9: #direct corner
                                new_ca[b][i+delta_y][j+delta_x] += moved_val
                            elif new_ca[b][i][j+delta_x] < 0.9: #priotize moving in x over y
                                new_ca[b][i][j+delta_x] += moved_val
                            elif new_ca[b][i+delta_y][j] < 0.9:
                                new_ca[b][i+delta_y][j] += moved_val
                            else:
                                #new_ca[b][i][j] = val #couldn't move
                                new_ca[b][i][j] += moved_val #couldn't move

                    #else: #cell is dead or cell isn't allowed to move due to mask
        return new_ca

    #def update_cell(ca, i, j, delta_x, delta_y):
        
#TODO: Make a way of generating starting positions, which are part of previous sequences seen - like the article