import os
import sys
from tqdm import tqdm
import torch

sys.path.append('./src/')
from h02_learn.dataset import get_data_loaders
from h02_learn.model import LstmLM, NgramLM
from util import argparser
from util import util
from util import constants


def get_args():
    # Data
    argparser.add_argument('--batch-size', type=int, default=512)
    # Models
    argparser.add_argument('--checkpoints-path', type=str, required=True)
    argparser.add_argument('--model-type', type=str, required=True)

    args = argparser.parse_args()
    args.model_path = os.path.join(args.checkpoints_path, args.model_type)
    return args


def load_model(fpath, model_type):
    if model_type in ['lstm']:
        model_cls = LstmLM
    elif model_type in ['ngram']:
        model_cls = NgramLM
    else:
        raise ValueError('Invalid model type requested %s' % model_type)

    return model_cls.load(fpath).to(device=constants.device)


def eval_per_word(dataloader, model, alphabet):
    # pylint: disable=too-many-locals
    model.eval()
    pad_idx = alphabet.char2idx('PAD')

    words, losses, lengths = [], [], []
    dev_loss, n_instances = 0, 0
    for x, y in tqdm(dataloader, desc='Evaluating per word'):
        y_hat = model(x)
        loss = model.get_loss_full(y_hat, y)

        dev_loss += loss.sum(-1)
        n_instances += y.shape[0]
        losses += [loss]
        words += [
            ''.join(alphabet.idx2word(item.tolist()))
            for item in y
        ]
        lengths += [(y != pad_idx).sum(-1)]

    losses = torch.cat(losses)
    lengths = torch.cat(lengths)

    results = {
        'losses': losses.cpu().numpy(),
        'words': words,
        'lengths': lengths.cpu().numpy(),
    }

    return results, (dev_loss / n_instances).item()
    # return results, (dev_loss / lengths.sum()).item()


def eval_all(model_path, dataloader, model_type):
    # pylint: disable=too-many-locals
    trainloader, devloader, testloader, alphabet = dataloader
    model = load_model(model_path, model_type)
    model_name = model_path.split('/')[-1]

    train_res, train_loss = eval_per_word(trainloader, model, alphabet)
    dev_res, dev_loss = eval_per_word(devloader, model, alphabet)
    test_res, test_loss = eval_per_word(testloader, model, alphabet)

    print('Training loss: %.4f Dev loss: %.4f Test loss: %.4f' %
          (train_loss, dev_loss, test_loss))

    results = {
        'name': model_name,
        'train': train_res,
        'dev': dev_res,
        'test': test_res,
    }

    return results


def main():
    args = get_args()
    folds = [list(range(8)), [8], [9]]

    dataloader = get_data_loaders(
        args.data_file, folds, args.batch_size)

    with torch.no_grad():
        results = eval_all(args.model_path, dataloader, args.model_type)

    results_file = '%s/losses.pckl' % (args.model_path)
    util.write_data(results_file, results)


if __name__ == '__main__':
    main()
