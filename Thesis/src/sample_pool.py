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

    def sample(self, n):
        idx = np.random.choice(self._size, n, False)
        batch = {k: list(itemgetter(*idx)(getattr(self, k))) for k in self._slot_names} 
        batch = SamplePool(**batch, _parent=self, _parent_idx=idx)
        return batch

    def commit(self):
        for k in self._slot_names:
            parent_vals = getattr(self._parent, k)
            child_vals = getattr(self, k)
            for i in range(self._size):
                parent_vals[self._parent_idx[i]] = child_vals[i]