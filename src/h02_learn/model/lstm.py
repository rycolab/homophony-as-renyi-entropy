from torch import nn
import torch.nn.functional as F

from .nn import BaseNNLM


class LstmLM(BaseNNLM):
    # pylint: disable=arguments-differ,too-many-instance-attributes
    name = 'lstm-lm'

    def __init__(self, alphabet, embedding_size, hidden_size,
                 nlayers, dropout):
        super().__init__(alphabet, embedding_size, hidden_size,
                         nlayers, dropout)

        self.embedding = nn.Embedding(self.alphabet_size, embedding_size)

        self.lstm = nn.LSTM(embedding_size, hidden_size, nlayers,
                            dropout=(dropout if nlayers > 1 else 0),
                            batch_first=True)

        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_size, embedding_size)
        self.out = nn.Linear(embedding_size, self.alphabet_size)

        # Tie weights
        self.out.weight = self.embedding.weight

    def forward(self, x):
        x_emb = self.get_embeddings(x)

        c_t, _ = self.lstm(x_emb)
        c_t = self.dropout(c_t).contiguous()

        hidden = F.relu(self.linear(c_t))
        logits = self.out(hidden)
        return logits

    def forward_single_char(self, x, context):
        x_emb = self.get_embeddings(x)

        c_t, context = self.lstm(x_emb, context)
        c_t = self.dropout(c_t).contiguous()

        hidden = F.relu(self.linear(c_t))
        logits = self.out(hidden)
        return logits[:, -1], context

    def get_embeddings(self, instance):
        emb = self.dropout(self.embedding(instance))

        return emb

    def pop_finished_samples(self, x, context, word_logprobs, ended, samples):
        samples += [self.alphabet.idx2word(item[1:-1].cpu().numpy()) for item in x[ended]]

        x, word_logprobs = x[~ended], word_logprobs[~ended]
        context = (context[0][:, ~ended], context[1][:, ~ended])
        return x, context, word_logprobs, samples
