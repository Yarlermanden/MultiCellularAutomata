import numpy as np

class SamplePool:
    def __init__(self, *, _parent=None, _parent_idx=None, **slots):
        self._parent = _parent
        self._parent_idx = _parent_idx
        self._slot_names = slots.keys()
        self._size = None
        for k, v in slots.items():
            if self._size is None:
                self._size = len(v)
            assert self._size == len(v)
            setattr(self, k, np.asarray(v))

    def sample(self, n):
        idx = np.random.choice(self._size, n, False)
        batch = {k: getattr(self, k)[idx] for k in self._slot_names}
        batch = SamplePool(**batch, _parent=self, _parent_idx=idx)
        return batch

    def commit(self):
        for k in self._slot_names:
            getattr(self._parent, k)[self._parent_idx] = getattr(self, k)
            
    
self.pool_size = 1024
batch = self.generator.generate_ca_and_food(self.pool_size)
pool = SamplePool(x=batch) #pool contains x and food

batch = pool.sample(self.batch_size)
ca = batch.x
ca[:self.batch_size//2] = self.generator.generate_ca_and_food(self.batch_size//2) #replace half with original

x_hat[:, 3] = food
batch.x[:] = x_hat.detach().cpu().numpy()
batch.commit()