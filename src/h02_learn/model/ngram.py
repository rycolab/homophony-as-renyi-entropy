import torch

from util import constants
from .base import BaseLM


class NgramLM(BaseLM):
    # pylint: disable=abstract-method,not-callable,too-many-instance-attributes,unused-argument
    name = 'ngram'

    def __init__(self, alphabet, embedding_size, hidden_size,
                 nlayers, dropout):
        super().__init__(alphabet, embedding_size, hidden_size,
                         nlayers, dropout)

        self.n = 5
        self.ngram = Ngram(self.n, self.alphabet_size, self.pad_idx, self.eos_idx)

    def fit(self, trainloader, devloader, train_info):

        for x, y in trainloader:
            x = torch.cat([x[:, :1], y], 1)
            self.fit_batch(x)

    def fit_batch(self, x):
        with torch.no_grad():
            for word in x:
                self.ngram.add_string(word)

    def forward(self, x):
        log_probs = torch \
            .zeros((x.shape[0], x.shape[1], self.alphabet_size)) \
            .to(device=constants.device)
        for i, word in enumerate(x):
            log_prob = self.ngram.forward(word)
            log_probs[i, :log_prob.shape[0]] = log_prob

        return log_probs

    def forward_single_char(self, x, context=None):
        log_probs = torch \
            .zeros((x.shape[0], self.alphabet_size)) \
            .to(device=constants.device)

        if context is not None:
            assert x.shape[0] == context.shape[0]
            x = torch.cat([context, x], axis=-1)
            x = x[:, -self.n + 1:]

        for i, word in enumerate(x):
            if self.eos_idx in word or self.pad_idx in word:
                log_prob = torch.zeros(self.alphabet_size) \
                    .to(device=constants.device)
                x[i, -1] = self.eos_idx
            else:
                log_prob = self.ngram.get_log_probs(word.tolist())

            log_probs[i] = log_prob
        return log_probs, x

    def save(self, path):
        fname = self.get_name(path)
        torch.save({
            'kwargs': self.get_args(),
            'ngram': self.ngram,
            'model_state_dict': self.state_dict(),
        }, fname)

    @classmethod
    def load(cls, path):
        checkpoints = cls.load_checkpoint(path)
        model = cls(**checkpoints['kwargs'])
        model.load_ngram(checkpoints['ngram'])
        model.load_state_dict(checkpoints['model_state_dict'])
        return model

    def load_ngram(self, ngram):
        self.ngram = ngram


class Ngram:
    # pylint: disable=too-many-instance-attributes

    def __init__(self, n, alphabet_size, pad_idx, eos_idx, laplace_smoothing=0.01):
        super().__init__()
        self.alphabet_size = alphabet_size
        self.pad_idx = pad_idx
        self.eos_idx = eos_idx
        self.laplace_smoothing = laplace_smoothing
        self.n = n
        self.context_size = n - 1

        self.counts = {}
        self.base_count = torch.FloatTensor(self.alphabet_size).zero_()

    def add_string(self, x):
        char_list = x.tolist()

        for pos, char in enumerate(char_list):
            if pos == 0:
                continue

            context_start = max(pos - self.context_size, 0)
            context = char_list[context_start:pos]

            self.count_entry(context, char)

            if self._is_last_char(char):
                break

    def count_entry(self, context, char):
        context = self.pad_context(context)

        if context not in self.counts:
            self.counts[context] = torch.FloatTensor(self.alphabet_size).zero_()

        self.counts[context][char] += 1

    def pad_context(self, context):
        assert len(context) <= self.context_size, \
            "Context should not be larger than max"

        if len(context) < self.context_size:
            pad_str = ['PAD'] * (self.context_size - len(context))
            context = pad_str + context
        return tuple(context)

    def forward(self, x):
        log_probs = []

        # Make every word be padded
        x = x.tolist()
        for pos, char in enumerate(x):
            if self._is_last_char(char):
                break
            # (char == self.pad_idx) or (char == self.eos_idx):

            context_start = max(pos - self.context_size + 1, 0)
            context = x[context_start:pos + 1]

            # context = self.pad_context(context)
            log_prob = self.get_log_probs(context)
            log_probs += [log_prob]

        log_probs = torch.cat(log_probs, dim=0)

        return log_probs

    def get_log_probs(self, context):
        context = self.pad_context(context)

        # counts = node['counts']
        if context in self.counts:
            counts = self.counts[context]
        else:
            counts = self.base_count

        probs = self.get_probs(counts)

        log_probs = torch.log(probs)
        log_probs = log_probs.reshape(1, -1)
        return log_probs

    def get_probs(self, counts):
        return (counts + self.laplace_smoothing) / \
            (counts.sum() + self.laplace_smoothing * self.alphabet_size)

    def _is_last_char(self, x):
        return x in (self.pad_idx, self.eos_idx)

    def __str__(self):
        return 'NgramLM(%d)' % (len(self.counts))
