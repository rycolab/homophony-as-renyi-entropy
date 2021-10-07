from abc import ABC, abstractmethod
from torch.utils.data import Dataset


class BaseDataset(Dataset, ABC):
    # pylint: disable=no-member

    def __init__(self, data, folds):
        self.data = data
        self.folds = folds
        self.process_train(data)
        self._train = True

    @abstractmethod
    def process_train(self, data):
        pass

    def get_word_idx(self, word):
        return [self.alphabet.char2idx('SOS')] + \
            self.alphabet.word2idx(word) + \
            [self.alphabet.char2idx('EOS')]

    def __len__(self):
        return self.n_instances

    def __getitem__(self, index):
        return (self.word_train[index],)
