import random
import torch


class SampleQueue:
    queues = {}
    _n_stored = 0

    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def add(self, x, probs):
        assert x.shape[0] == probs.shape[0], 'x and probs should have same number of elements'
        x_len = x.shape[1]
        # print(x_len)

        if x_len not in self.queues:
            self.queues[x_len] = SampleQueueSingleLength(self.pad_idx)

        self.queues[x_len].add(x, probs)
        self._n_stored += x.shape[0]

    def pop(self, batch_size):
        x_len = random.sample(self.queues.keys(), 1)[0]

        batch = self.queues[x_len].pop(batch_size)
        self._n_stored -= batch[0].shape[0]

        if self.queues[x_len].empty:
            del self.queues[x_len]

        return batch

    @property
    def empty(self):
        return len(self.queues) == 0

    @property
    def n_stored(self):
        return self._n_stored


class SampleQueueSingleLength:
    x_queue = None
    probs_queue = None

    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def add(self, x, probs):
        assert x.shape[0] == probs.shape[0], 'x and probs should have same number of elements'

        if self.probs_queue is None:
            self.probs_queue = probs.clone()
            self.x_queue = x.clone()
        else:
            self.probs_queue = torch.cat([self.probs_queue, probs])
            self.x_queue = torch.cat([self.x_queue, x])

    def pop(self, batch_size):
        if self.probs_queue.shape[0] <= batch_size:
            return self._pop_empty()
        else:
            return self._pop_batch(batch_size)

    def _pop_empty(self):
        probs = self.probs_queue
        x = self.x_queue

        self.probs_queue = None
        self.x_queue = None

        return x, probs

    def _pop_batch(self, batch_size):
        probs, idxs = self.probs_queue.topk(batch_size)
        x = self.fetch_idxs(self.x_queue, idxs)

        self.x_queue = self.remove_idxs(self.x_queue, idxs)
        self.probs_queue = self.remove_idxs(self.probs_queue, idxs)

        return x, probs

    def fetch_idxs(self, x, idxs):
        return x[idxs]

    def remove_idxs(self, x, idxs):
        mask = torch.ones(self.probs_queue.shape[0]).bool()
        mask[idxs] = False
        if len(x.shape) == 2:
            return x[mask.unsqueeze(-1).repeat(1, x.shape[1])].reshape(-1, x.shape[1])
        else:
            return x[mask]

    @property
    def empty(self):
        return self.probs_queue is None
