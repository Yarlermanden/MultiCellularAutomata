import numpy as np
from operator import itemgetter

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
            keys = range(self._size)
            dic = dict(zip(keys, v))
            setattr(self, k, dic)
            #setattr(self, k, np.asarray(v))

    def sample(self, n):
        idx = np.random.choice(self._size, n, False)
        #batch = {k: getattr(self, k)[idx] for k in self._slot_names}
        #print('dic: ', [getattr(self, k) for k in self._slot_names])
        batch = {k: list(itemgetter(*idx)(getattr(self, k))) for k in self._slot_names} 
        batch = SamplePool(**batch, _parent=self, _parent_idx=idx)
        return batch

    def commit(self):
        for k in self._slot_names:
            #print('attr: ', getattr(self,k))
            #there must be some way of resizing the parent...
            #getattr(self._parent, k)[self._size] = self._size

            parent_vals = getattr(self._parent, k)
            child_vals = getattr(self, k)
            for i in range(self._size):
                #parent_vals[self._parent_idx[i]] = self._parent
                parent_vals[self._parent_idx[i]] = child_vals[i]
            #getattr(self._parent, k)[self._parent_idx] = child_vals[i]
            
#self.pool_size = 1024
#batch = self.generator.generate_ca_and_food(self.pool_size)
#pool = SamplePool(x=batch) #pool contains x and food
#
#batch = pool.sample(self.batch_size)
#ca = batch.x
#ca[:self.batch_size//2] = self.generator.generate_ca_and_food(self.batch_size//2) #replace half with original
#
#x_hat[:, 3] = food
#batch.x[:] = x_hat.detach().cpu().numpy()
#batch.commit()