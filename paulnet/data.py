# Data Utils
import numpy as np
np.random.seed(42)

class Batcher(object):
    def __init__(self,x,y, batch_size=64):
        self.x = x
        self.y = y if len(y.shape)>1 else y.reshape(-1,1)
        self.bs = batch_size
        # Always shuffle

    def __iter__(self):
        batch = []
        for idx in np.random.permutation(range(self.x.shape[0])):
            batch.append(idx)
            if len(batch)==self.bs:
                yield (self.x[batch, :], self.y[batch,:])
                batch = []
        if len(batch)>0:
            yield (self.x[batch, :], self.y[batch,:])

    def __len__(self):
        return np.ceil(len(self.y)/self.bs)

class DataLoader(object):
    def __init__(self,x,y, batch_size=64, shuffle=True):
        self.x = x
        self.y = y if len(y.shape)>1 else y.reshape(-1,1)
        self.bs = batch_size

    def __iter__(self):
        raise NotImplementedError()

    def __len__(self):
        raise NotImplementedError()
