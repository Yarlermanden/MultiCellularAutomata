import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from State_Generator import Generator, State

class Trainer():
    def __init__(
        self,
        model,
        device
    ):
        self.model = model
        self.device = device
        self.lr = 0.0001
        self.epochs = 6
        self.iterations = 10000 # pr epoch
        self.iterations_per_sample = 2000
        self.random_states = False
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
        self.generator = Generator(device, self.random_states)

    def train(self):
        losses_list = np.zeros(self.iterations * self.epochs // 10)

        # train to be stationary in one step


        #train to be stationary after many steps


        #train to understand moving towards the rewards



        for epoch in tqdm(range(self.epochs)):
            for i in range(self.iterations):
                if i % self.iterations_per_sample == 0:
                    state = self.generator.generate_stationary_state()
                #state = self.generator.generate_stationary_state()

                loss_item = self.train_step(state, 2)
                
                if i % 10 == 0:
                    losses_list[epoch*self.iterations//10 + i//10] = loss_item
            print(loss_item)
        return self.model, losses_list

    def train_step(self, state, steps): #timesteps to iterate
        self.optimizer.zero_grad()
        x_hat, food = self.model(state.x.clone(), state.food, steps)

        loss = self.criterion(x_hat[0], state.y[0])
        loss_item = loss.item()

        loss.backward()
        self.optimizer.step()
        return loss_item
