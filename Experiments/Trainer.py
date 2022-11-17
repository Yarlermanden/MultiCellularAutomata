import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from State_Generator import Generator, State
from SamplePool import SamplePool

class Trainer():
    def __init__(
        self,
        model,
        device
    ):
        self.model = model
        self.device = device
        self.lr = 0.0001
        self.epochs = 0 #11
        self.epochs2 = 100
        self.iterations = 2000
        self.iterations_per_sample = 250
        self.batch_size = model.batch_size
        self.random_states = False
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
        self.generator = Generator(device, self.random_states)
        self.pool_size = 1024

    def train(self):
        losses_list = np.zeros(self.iterations * (self.epochs+self.epochs2) // 10)
        lr = self.lr
        timesteps = 4
        #pool = SamplePool(x=self.generator.generate_moving_state(timesteps//2, self.batch_size))
        batch = self.generator.generate_ca_and_food(self.batch_size)
        pool = SamplePool(x=batch.detach().cpu().numpy()) #pool contains x and food

        #TODO: need to look more into the curriculum and how the model does. When doing badly ensure it still works on simpler stuff

        for epoch in tqdm(range(self.epochs2)): #Train moving
            if epoch < 2:
                lr = self.lr
                timesteps = 2
            elif epoch < 4:
                lr = self.lr/5
                timesteps = 4
            elif epoch < 6:
                timesteps = 6
            elif epoch < 8:
                timesteps = np.random.randint(5, 10)
            elif epoch < 15:
                timesteps = np.random.randint(5, 20)
            elif epoch < 30:
                lr = self.lr/10
                timesteps = np.random.randint(15, 25)
            elif epoch < 50:
                timesteps = np.random.randint(10, 40)
            else:
                timesteps = np.random.randint(10, 50)
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

            for i in range(self.iterations):
                #TODO find out if it's better to keep in tensor or np
                batch = pool.sample(self.batch_size)
                ca = torch.tensor(batch.x, device=self.device)
                #batch[:self.batch_size//2] = self.generate_moving_state(timesteps//2, self.batch_size//2) #replace half with original
                ca[:self.batch_size//2] = self.generator.generate_ca_and_food(self.batch_size//2) #replace half with original
                #should use timesteps to generate y efter retrieving from pool
                food = ca[:, 3]
                food_coord = self.generator.get_food_coord_from_food(food)

                target_ca = ca[:, 0]
                for i in range(timesteps//2):
                    if i > 1:
                        target_ca = self.generator.move_towards_food(target_ca, food_coord)

                #if i % 100 == 0:
                #    state = self.generator.generate_moving_state(timesteps//2, self.batch_size)
                #state = State(batch.x, target_ca)
                state = State(ca, target_ca, food)
                x_hat, loss_item = self.train_step(state, timesteps)

                if i % 10 == 0:
                    if i % 100 == 0:
                        print(loss_item)
                    losses_list[((self.epochs + epoch)*self.iterations + i)//10] = loss_item
                x_hat[:, 3] = food
                batch.x[:] = x_hat.detach().cpu().numpy()
                batch.commit()
            name = 'models/complex_ca_moving1_temp' + str(epoch) + '.pth'
            torch.save(self.model.state_dict(), name)

        return self.model, losses_list

    def train_step(self, state, steps): #timesteps to iterate
        self.optimizer.zero_grad()
        x_hat, food, live_count = self.model(state.x.clone(), state.food.clone(), steps)

        #loss = F.mse_loss(x_hat[0], state.y) 
        loss = self.criterion(x_hat[:, 0], state.y)
        loss2 = self.criterion(live_count, state.x[:, 0:1].sum(dim=(1,2,3))) #TODO ensure same count as for live_count
        #TODO ensure that loss2 works as expected
        loss2 = loss2/4
        loss = loss+loss2
        loss_item = loss.item()

        loss.backward()
        self.optimizer.step()
        return x_hat, loss_item
