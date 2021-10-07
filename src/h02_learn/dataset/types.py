import torch

from .base import BaseDataset


class TypeDataset(BaseDataset):

    def process_train(self, data):
        folds_data = data[0]
        self.alphabet = data[1]

        self.words = [instance['tgt'] for fold in self.folds for instance in folds_data[fold]]
        self.word_train = [torch.LongTensor(self.get_word_idx(word)) for word in self.words]
        self.n_instances = len(self.word_train)
