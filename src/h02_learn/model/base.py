import copy
import math
from abc import ABC
import torch
from torch import nn
from torch.nn import functional as F

from util.queue import SampleQueue
from util import constants


class BaseLM(nn.Module, ABC):
    # pylint: disable=abstract-method,not-callable,too-many-instance-attributes,too-many-public-methods
    name = 'base'
    criterion_cls = nn.CrossEntropyLoss

    def __init__(self, alphabet, embedding_size, hidden_size,
                 nlayers, dropout):
        super().__init__()
        self.alphabet = alphabet

        self.best_state_dict = None
        self.alphabet_size = len(self.alphabet)
        self.pad_idx = alphabet.PAD_IDX
        self.eos_idx = alphabet.EOS_IDX
        self.sos_idx = alphabet.SOS_IDX

        self.nlayers = nlayers
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.dropout_p = dropout

        self.criterion = self.criterion_cls(ignore_index=self.pad_idx) \
            .to(device=constants.device)
        self.criterion_full = self.criterion_cls(
            ignore_index=self.pad_idx, reduction='none') \
            .to(device=constants.device)

    def evaluate(self, evalloader):
        dev_loss, n_instances = 0, 0
        for x, y in evalloader:
            y_hat = self(x)
            loss = self.get_loss(y_hat, y)

            batch_size = y.shape[0]
            dev_loss += (loss * batch_size)
            n_instances += batch_size

        return (dev_loss / n_instances).item()

    def get_loss(self, y_hat, y):
        return self.criterion(
            y_hat.reshape(-1, y_hat.shape[-1]),
            y.reshape(-1)) / math.log(2)

    def get_loss_full(self, y_hat, y):
        loss_per_char = self.criterion_full(
            y_hat.reshape(-1, y_hat.shape[-1]),
            y.reshape(-1)) \
            .reshape_as(y) / math.log(2)

        loss = loss_per_char.sum(-1)
        return loss

    def set_best(self):
        self.best_state_dict = copy.deepcopy(self.state_dict())

    def recover_best(self):
        self.load_state_dict(self.best_state_dict)

    def save(self, path):
        fname = self.get_name(path)
        torch.save({
            'kwargs': self.get_args(),
            'model_state_dict': self.state_dict(),
        }, fname)

    def get_args(self):
        return {
            'nlayers': self.nlayers,
            'hidden_size': self.hidden_size,
            'embedding_size': self.embedding_size,
            'dropout': self.dropout_p,
            'alphabet': self.alphabet,
        }

    @classmethod
    def load(cls, path):
        checkpoints = cls.load_checkpoint(path)
        model = cls(**checkpoints['kwargs'])
        model.load_state_dict(checkpoints['model_state_dict'])
        return model

    @classmethod
    def load_checkpoint(cls, path):
        fname = cls.get_name(path)
        return torch.load(fname, map_location=constants.device)

    @classmethod
    def get_name(cls, path):
        return '%s/model.tch' % (path)

    def get_probabilities(self, batch_size, delta=0.001, print_every=100):
        samples, lengths, full_probs, n_samples, count = [], [], 0, 0, 0
        queue = SampleQueue(self.pad_idx)

        x, probs = self.get_initial_input()
        queue.add(x, probs)

        while not queue.empty:
            x, probs = queue.pop(batch_size)
            count += 1

            x, probs = self.run_prob_batch(x, probs)
            x, probs, finished_samples = self.get_finished(x, probs)

            if finished_samples is not None:
                samples += [finished_samples.cpu()]
                lengths += [torch.ones_like(finished_samples).cpu() * (x.shape[1] - 1)]
                full_probs += finished_samples.sum().item()
                n_samples += finished_samples.shape[0]

            x, probs = self.filter_probs(x, probs, min_prob=1e-8)
            if x.shape[0] > 0:
                queue.add(x, probs)

            if (count % print_every) == 0:
                print('Full probs: %.4f # sampled: %d # stored: %d' %
                      (full_probs, n_samples, queue.n_stored))

            if full_probs >= 1 - delta:
                break

        samples = torch.cat(samples)
        lengths = torch.cat(lengths)
        return samples, lengths

    def get_initial_input(self):
        x = torch.LongTensor(
            [[self.alphabet.SOS_IDX]]) \
            .to(device=constants.device)

        probs = self.get_probs(x, first=True).squeeze(0)

        valid_idxs = probs > 0
        idx = torch.arange(0, probs.shape[-1]).long().to(device=constants.device)
        idx = idx[valid_idxs]
        probs = probs[valid_idxs]

        x = self.extend_input(x, idx)
        return x, probs

    def run_prob_batch(self, x, probs):
        probs_batch = self.get_probs(x)

        idx = torch.arange(0, probs_batch.shape[-1]).long().to(device=constants.device)
        x_ext = self.extend_input(x, idx)
        probs_ext = self.extend_probs(probs, probs_batch)

        return x_ext, probs_ext

    def get_probs(self, x, first=False):
        logits = self(x)
        logits = self.mask_logits(logits[:, -1, :], first=first)
        probs = F.softmax(logits, dim=-1)

        return probs

    @staticmethod
    def extend_input(x, y):
        x_ext = x.unsqueeze(-1).repeat(1, 1, y.shape[0])
        y_ext = y.reshape(1, 1, y.shape[0]).repeat(x.shape[0], 1, 1)

        x_res = torch.cat([x_ext, y_ext], dim=1)
        x_res = x_res.transpose(1, 2).reshape(-1, x.shape[1] + 1)
        return x_res

    @staticmethod
    def extend_probs(x, y):
        x_ext = x.unsqueeze(-1).repeat(1, y.shape[1])
        x_res = x_ext * y
        return x_res.reshape(-1)

    def get_finished(self, x, probs):
        x, probs = self.filter_probs(x, probs)

        samples = None
        ended = (x[:, -1] == self.eos_idx) | (x[:, -1] == self.pad_idx)
        if ended.any():
            samples = probs[ended]
            x = x[~ended].reshape(-1, x.shape[-1])
            probs = probs[~ended]

        return x, probs, samples

    @staticmethod
    def filter_probs(x, probs, min_prob=0):
        valid_idxs = (probs > min_prob)
        x = x[valid_idxs]
        probs = probs[valid_idxs]

        return x, probs

    def mask_logits(self, logits, first=False):
        logits[:, self.sos_idx] = -float('inf')
        logits[:, self.pad_idx] = -float('inf')
        if first:
            logits[:, self.eos_idx] = -float('inf')

        return logits

    def sample(self, n_samples, max_mass=1e-10):
        x = torch.LongTensor(
            [[self.sos_idx] for _ in range(n_samples)]) \
            .to(device=constants.device)

        samples, first, context = [], True, None
        word_logprobs = torch.zeros(
            x.shape[0], dtype=torch.float).to(device=constants.device)

        while True:
            logits, context = self.forward_single_char(x[:, -1:], context)
            # la = self(x)
            # assert (logits == la[:, -1]).all()

            logits = self.mask_logits(logits, first)
            probs = F.softmax(logits, dim=-1)

            first = False

            y = probs.multinomial(1)
            x = torch.cat([x, y], dim=-1)

            logprobs = F.log_softmax(logits, dim=-1)
            word_logprobs += torch.gather(logprobs, -1, y).squeeze()

            y = y.squeeze(-1)
            ended = (y == self.eos_idx) | (y == self.pad_idx) | \
                (word_logprobs < math.log(max_mass))

            if ended.any():
                x, context, word_logprobs, samples = self.pop_finished_samples(
                    x, context, word_logprobs, ended, samples)

            if ended.all():
                break

        assert len(samples) == n_samples
        return [''.join(item) for item in samples]

    def pop_finished_samples(self, x, context, word_logprobs, ended, samples):
        samples += [self.alphabet.idx2word(item[1:-1].cpu().numpy()) for item in x[ended]]

        return x[~ended], context[~ended], word_logprobs[~ended], samples
