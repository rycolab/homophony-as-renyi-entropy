import torch
from torch import optim

from .base import BaseLM


class BaseNNLM(BaseLM):
    # pylint: disable=abstract-method,not-callable,too-many-instance-attributes
    name = 'base-nn'

    def evaluate(self, evalloader):
        self.eval()
        with torch.no_grad():
            result = super().evaluate(evalloader)
        self.train()
        return result

    def fit(self, trainloader, devloader, train_info):
        optimizer = optim.Adam(self.parameters())

        while not train_info.finish:
            for x, y in trainloader:
                loss = self.train_batch(x, y, optimizer)
                train_info.new_batch(loss)

                if train_info.eval:
                    dev_loss = self.evaluate(devloader)

                    if train_info.is_best(dev_loss):
                        self.set_best()
                    elif train_info.finish:
                        break

                    train_info.print_progress(dev_loss)

        self.recover_best()

    def train_batch(self, x, y, optimizer):
        optimizer.zero_grad()
        y_hat = self(x)
        loss = self.get_loss(y_hat, y)
        loss.backward()
        optimizer.step()

        return loss.item()
