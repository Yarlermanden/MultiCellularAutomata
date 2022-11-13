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
        

        #TODO make method to do it iteratively - that way we can simulate how it would look....
        return State(state, target_ca, food)


    def generate_stationary_state(self):
        zeros = torch.zeros(self.width, self.width)
        #center_ca = self.get_centered_CA()
        #cell = zeros
        #cell[6:11, 6:11] = center_ca
        cell = self.get_centered_CA()
        state = torch.stack([cell, zeros, zeros, zeros])
        return State(state, state, zeros)

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
            lower = random.randint(0, middle-5)
            upper = random.randint(middle+5, self.width-1)
            coin_flip = random.randint(0, 1)
            return lower*coin_flip + upper*(1-coin_flip) #returns lower or upper

        middle = self.width // 2
        x = random_outer_num(middle)
        y = random_outer_num(middle)
        return x,y

    def move_towards_food(self, ca, food):
        #TODO fix the out of range
        #TODO fix the wraparound failure...

        #could try without ordering, but to order:
        #for indices, ca in enumerate(ca): - both dimensions at once?!
        #delta_x = food.x - indices.x
        #delta_y = food.y - indices.y
        #order by pythagoras: sqrt(delta_x^2 _ delta_y^2)

        new_ca = ca.clone()

        #generate random map of entire world with values 0 or 1 for whether to move or not
        move_mask = (torch.rand_like(ca) > 0.5).to(torch.float)

        #without ordering:
        for i, row in enumerate(ca):
            for j, val in enumerate(row):
                if val > 0 and move_mask[i][j] > 0: #allowed to move
                    #find direction to move in - almost always 3 neighboring cells closer than current
                    delta_x = food[0] - j #TODO check whether j and i are correct here
                    delta_y = food[1] - i

                    #fix deltas to be -1, 0 or 1
                    if delta_x > 0:
                        delta_x = 1
                    elif delta_x < 0:
                        delta_x = -1

                    if delta_y > 0:
                        delta_y = 1
                    elif delta_y < 0:
                        delta_y = -1

                    #if 0:     on the same coordinates - both 1 up and down is fine, if moving in the other coordinate as well
                    #if minus: need to subtract 1
                    #if plus:  need to add 1

                    new_ca[i][j] = 0 #assume it can move
                    if delta_x == 0: #line up in column
                        if new_ca[i+delta_y][j] == 0: #available
                            new_ca[i+delta_y][j] = 1
                        elif new_ca[i+delta_y][j+1] == 0: #priotize right over left...
                            new_ca[i+delta_y][j+1] = 1
                        elif new_ca[i+delta_y][j-1] == 0:
                            new_ca[i+delta_y][j-1] = 1
                        else:
                            new_ca[i][j] = 1 #couldn't move

                    elif delta_y == 0: #line up in row
                        if new_ca[i][j+delta_x] == 0: #most direct way
                            new_ca[i][j+delta_x] = 1
                        elif new_ca[i+1][j+delta_x] == 0: #priotize down over up
                            new_ca[i+1][j+delta_x] = 1
                        elif new_ca[i-1][j+delta_x] == 0:
                            new_ca[i-1][j+delta_x] = 1
                        else:
                            new_ca[i][j] = 1 #couldn't move

                    #neither lines up
                    #move to the corners available
                    else:
                        if new_ca[i+delta_y][j+delta_x] == 0: #direct corner
                            new_ca[i+delta_y][j+delta_x] =1
                        elif new_ca[i][j+delta_x] == 0: #priotize moving in x over y
                            new_ca[i][j+delta_x] = 1
                        elif new_ca[i+delta_y][j] == 0:
                            new_ca[i+delta_y][j] = 1
                        else:
                            new_ca[i][j] = 1 #couldn't move

                #else: #cell is dead or cell isn't allowed to move due to mask
        return new_ca

        #One approach
        #Get an ordered index tuple of indicies closest to the food source
        #iterate through them and stochastically determine whether each moves
        #if they move, pick one of the 3 cells closer to the food source and move to one of them
        #else stay
        #if blocked pick another one or stay

    #def update_cell(ca, i, j, delta_x, delta_y):
        
