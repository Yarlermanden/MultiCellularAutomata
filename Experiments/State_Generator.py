import torch
import random

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
        #TODO some cells should register the food quicker than others - some should wait even longer before starting to move
        #TODO should try to make it less spread out - ensure one entity

        #Generate initial CA
        zeros = torch.zeros(self.width, self.width)

        state = self.get_centered_CA()
        state = torch.stack([state, zeros, zeros, zeros])

        #Generate food map
        food_coord = self.random_food_noncentered()
        food = zeros
        food[food_coord[0], food_coord[1]] = 1

        #Generate target CA
        target_ca = state[0]
        for _ in range(timesteps):
            target_ca = self.move_towards_food(target_ca, food_coord)

        return State(state, target_ca, food)


    def generate_stationary_state(self):
        zeros = torch.zeros(self.width, self.width)
        cell = self.get_centered_CA()
        state = torch.stack([cell, zeros, zeros, zeros])
        food = zeros
        food[self.width//2, self.width//2] #not entirely centered
        return State(state, cell, food)

    def get_centered_CA(self):
        zeros = torch.zeros(self.width, self.width)
        if self.random_states:
            center_ca = torch.rand(5, 5, device=self.device) > 0.7
        else:
            center_ca = torch.tensor([
                [0, 0, 1, 0, 0],
                [0, 1, 1, 1, 0],
                [1, 1, 1, 1, 1],
                [0, 1, 1, 1, 0],
                [0, 0, 1, 0, 0]
            ])
        ca = zeros
        ca[6:11, 6:11] = center_ca
        return ca

    def random_food_noncentered(self):
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
        #TODO: Fix bug where food is located wrong - probably mix between i and j
        #TODO: in case it reaches food, remove food and instead increase cells

        #TODO fix the out of range - due to out of range
        #TODO fix the wraparound failure... - is due to -1

        #for indices, ca in enumerate(ca): - both dimensions at once?!
        #delta_x = food.x - indices.x
        #delta_y = food.y - indices.y
        #order by pythagoras: sqrt(delta_x^2 _ delta_y^2)

        new_ca = ca.clone()

        #generate random map of entire world with values 0 or 1 for whether to move or not
        move_mask = (torch.rand_like(ca) > 0.1).to(torch.float)

        #TODO: ensure no values are negative
        for i, row in enumerate(ca):
            for j, val in enumerate(row):
                if val > 0.1 and move_mask[i][j] == 1: #allowed to move
                    #find direction to move in - almost always 3 neighboring cells closer than current
                    delta_x = food[1] - j #TODO why should this be flipped?
                    delta_y = food[0] - i

                    moved_val = 0.2
                    current_val = new_ca[i][j]

                    #fix deltas to be -1, 0 or 1
                    if delta_x > 0:
                        delta_x = 1
                    elif delta_x < 0:
                        delta_x = -1

                    if delta_y > 0:
                        delta_y = 1
                    elif delta_y < 0:
                        delta_y = -1

                    new_ca[i][j] -= moved_val #assume it can move
                    if delta_x == 0: #line up in column
                        if new_ca[i+delta_y][j] < 1: #available
                            new_ca[i+delta_y][j] += moved_val
                        elif new_ca[i+delta_y][j+1] < 1: #priotize right over left...
                            new_ca[i+delta_y][j+1] += moved_val
                        elif new_ca[i+delta_y][j-1] < 1:
                            new_ca[i+delta_y][j-1] += moved_val
                        else:
                            new_ca[i][j] = current_val #couldn't move

                    elif delta_y == 0: #line up in row
                        if new_ca[i][j+delta_x] < 1: #most direct way
                            new_ca[i][j+delta_x] += moved_val
                        elif new_ca[i+1][j+delta_x] < 1: #priotize down over up
                            new_ca[i+1][j+delta_x] += moved_val
                        elif new_ca[i-1][j+delta_x] < 1:
                            new_ca[i-1][j+delta_x] += moved_val
                        else:
                            new_ca[i][j] = current_val #couldn't move

                    #neither lines up
                    #move to the corners available
                    else:
                        if new_ca[i+delta_y][j+delta_x] < 1: #direct corner
                            new_ca[i+delta_y][j+delta_x] += moved_val
                        elif new_ca[i][j+delta_x] < 1: #priotize moving in x over y
                            new_ca[i][j+delta_x] += moved_val
                        elif new_ca[i+delta_y][j] < 1:
                            new_ca[i+delta_y][j] += moved_val
                        else:
                            new_ca[i][j] = current_val #couldn't move

                #else: #cell is dead or cell isn't allowed to move due to mask
        return new_ca

    #def update_cell(ca, i, j, delta_x, delta_y):
        
