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
        self.epochs = 11
        self.epochs2 = 6
        self.iterations = 20000 # pr epoch
        self.iterations_per_sample = 2000
        self.random_states = False
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
        self.generator = Generator(device, self.random_states)

    def train(self):
        losses_list = np.zeros(self.iterations * (self.epochs+self.epochs2) // 10)
        lr = self.lr

        #train to be stationary in one step
        #train to be stationary after many steps
        for epoch in tqdm(range(self.epochs)):
            if epoch < 2:
                timesteps = 1
            elif epoch < 4:
                timesteps = 2
            elif epoch < 6:
                timesteps = 4
            elif epoch < 8:
                lr = self.lr/10
                timesteps = np.random.randint(5, 15) #random between...
            elif epoch < 10:
                lr = self.lr/30
                timesteps = np.random.randint(15, 40)
            elif epoch < 12:
                lr = self.lr/40
                timesteps = np.random.randint(40, 60)

            for i in range(self.iterations):
                if i % self.iterations_per_sample == 0:
                    state = self.generator.generate_stationary_state()

                loss_item = self.train_step(state, timesteps)
                
                if i % 10 == 0:
                    if i % 1000 == 0:
                        print(loss_item)
                    losses_list[(epoch*self.iterations + i)//10] = loss_item

        #train to understand moving towards the rewards
        #for epoch in tqdm(range(self.epochs)):
        for epoch in tqdm(range(self.epochs2)):
            if epoch < 2:
                timesteps = 5
            elif epoch < 4:
                lr = self.lr/10
                timesteps = 10
            elif epoch < 6:
                lr = self.lr/40
                timesteps = np.random.randint(15, 30)

            for i in range(self.iterations):
                state = self.generator.generate_moving_state(timesteps)

                loss_item = self.train_step(state, timesteps)

                if i % 10 == 0:
                    if i % 1000 == 0:
                        print(loss_item)
                    losses_list[(epoch*self.iterations + i)//10] = loss_item

        return self.model, losses_list

    def train_step(self, state, steps): #timesteps to iterate
        self.optimizer.zero_grad()
        x_hat, food = self.model(state.x.clone(), state.food.clone(), steps)

        loss = self.criterion(x_hat[0], state.y)
        loss_item = loss.item()

        loss.backward()
        self.optimizer.step()
        return loss_item
