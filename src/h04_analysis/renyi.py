# pylint: disable=too-many-locals

import os
import sys
import pandas as pd
import numpy as np

sys.path.append('./src/')
from util import argparser
from util import util


def get_args():
    # Data
    argparser.add_argument("--raw-file", type=str)
    # Models
    argparser.add_argument('--checkpoints-path', type=str, required=True)

    args = argparser.parse_args()
    return args


def get_entropies(probs):
    renyi = - np.log((probs ** 2).sum()) / np.log(2)
    shannon = - np.log(probs).mean() / np.log(2)
    return renyi, shannon


def _get_real_renyi(freqs):
    match = (freqs * (freqs - 1)) / 2
    n_items = freqs.sum()

    return - np.log(match.sum() * 2 / (n_items * (n_items - 1))) / np.log(2)



def get_real_renyi(df):
    freqs = df.homo.values

    match = (freqs * (freqs - 1)) / 2
    n_items = freqs.sum()

    return - np.log(match.sum() * 2 / (n_items * (n_items - 1))) / np.log(2)


def get_results(model_path):
    results_file = '%s/renyi.pckl' % (model_path)
    results = util.read_data(results_file)

    results_file = '%s/losses.pckl' % (model_path)
    real_results = util.read_data(results_file)

    results_file = '%s/samples.pckl' % (model_path)
    bootstrap_results = util.read_data(results_file)
    renyis_bootstrap = bootstrap_results['renyis']

    shannon_cross = real_results['test']['losses'].mean()
    train_cross = real_results['train']['losses'].mean()

    renyi = results['renyi']
    shannon = results['shannon']

    probs = results['probs']
    delta = 1e-8
    max_error = (1 - probs.sum()) * delta / (probs ** 2).sum()

    return shannon_cross, shannon, renyi, train_cross, renyis_bootstrap, max_error


def main():
    args = get_args()

    df = pd.read_csv(args.raw_file, sep='\t')
    renyi_data = get_real_renyi(df)
    print('Real\nRenyi: %.4f. # instances: %d' % (renyi_data.item(), df.homo.sum()))

    lstm_path = os.path.join(args.checkpoints_path, 'lstm')
    shannon_cross, shannon, renyi, lstm_train, lstm_bootstrap, lstm_max_error = \
        get_results(lstm_path)
    print('LSTM\nCross: %.4f Shannon: %.4f Renyi: %.4f Train Cross: %.4f' %
          (shannon_cross, shannon, renyi, lstm_train))

    ngram_path = os.path.join(args.checkpoints_path, 'ngram')
    ngram_cross, ngram_shannon, ngram_renyi, ngram_train, ngram_bootstrap, ngram_max_error = \
        get_results(ngram_path)
    print('Ngram\nCross: %.4f Shannon: %.4f Renyi: %.4f Train Cross: %.4f' %
          (ngram_cross, ngram_shannon, ngram_renyi, ngram_train))


    print('~~~~~~~~$n$-gram & %.2f & %.2f & %.2f & %.2f & %.2f \\\\' %
          (ngram_train, ngram_cross, ngram_shannon, ngram_renyi, ngram_bootstrap.mean()))
    print('~~~~~~~~LSTM & %.2f & %.2f & %.2f & %.2f & %.2f \\\\' %
          (lstm_train, shannon_cross, shannon, renyi, lstm_bootstrap.mean()))
    print('~~~~~~~~Real & - & - & - & - & %.2f \\\\' % (renyi_data))

    print()
    lstm_p = (lstm_bootstrap < renyi_data).sum() / lstm_bootstrap.shape[0]
    lstm_p = min(lstm_p, 1 - lstm_p) + (1 / lstm_bootstrap.shape[0])
    ngram_p = (ngram_bootstrap < renyi_data).sum() / ngram_bootstrap.shape[0]
    ngram_p = min(ngram_p, 1 - ngram_p) + (1 / ngram_bootstrap.shape[0])
    print('Significance. LSTM: %.4f n-gram: %.4f' % (lstm_p, ngram_p))
    print('Bootstrap mean. LSTM: %.4f n-gram: %.4f' %
          (lstm_bootstrap.mean(), ngram_bootstrap.mean()))
    print('Renyi max error. LSTM: %.2e n-gram: %.2e' %
          (lstm_max_error.mean(), ngram_max_error.mean()))



if __name__ == '__main__':
    main()
