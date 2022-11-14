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
        self.iterations = 30000 # pr epoch
        self.iterations_per_sample = 2000
        self.random_states = False
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
        self.generator = Generator(device, self.random_states)

    def train(self):
        losses_list = np.zeros(self.iterations * (self.epochs+self.epochs2) // 10)
        lr = self.lr

        for epoch in tqdm(range(self.epochs)): #Train stationary
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
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

            for i in range(self.iterations):
                if i % self.iterations_per_sample == 0:
                    state = self.generator.generate_stationary_state()

                loss_item = self.train_step(state, timesteps)
                
                if i % 10 == 0:
                    if i % 1000 == 0:
                        print(loss_item)
                    losses_list[(epoch*self.iterations + i)//10] = loss_item
            name = 'models/complex_ca_stationary_temp' + str(epoch) + '.pth'
            torch.save(self.model.state_dict(), name)
        torch.save(self.model.state_dict(), 'models/complex_ca5_stationary.pth')

        for epoch in tqdm(range(self.epochs2)): #Train moving
            if epoch < 2:
                lr = self.lr/5
                timesteps = 2
            elif epoch < 4:
                lr = self.lr/10
                timesteps = 5
            elif epoch < 6:
                lr = self.lr/40
                timesteps = np.random.randint(10, 15)
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

            for i in range(self.iterations):
                state = self.generator.generate_moving_state(timesteps//2)

                loss_item = self.train_step(state, timesteps)

                if i % 10 == 0:
                    if i % 1000 == 0:
                        print(loss_item)
                    losses_list[((self.epochs + epoch)*self.iterations + i)//10] = loss_item
            name = 'models/complex_ca_moving_temp' + str(epoch) + '.pth'
            torch.save(self.model.state_dict(), name)

        return self.model, losses_list

    def train_step(self, state, steps): #timesteps to iterate
        self.optimizer.zero_grad()
        x_hat, food, live_count = self.model(state.x.clone(), state.food.clone(), steps)

        loss = self.criterion(x_hat[0], state.y)
        loss2 = self.criterion(live_count, torch.sum(state.x[0:1])) #TODO ensure same count as for live_count
        loss2 = loss2/10
        loss = loss+loss2
        loss_item = loss.item()

        loss.backward()
        self.optimizer.step()
        return loss_item
