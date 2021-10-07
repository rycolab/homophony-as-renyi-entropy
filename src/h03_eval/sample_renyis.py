import os
import sys
from collections import Counter
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch

sys.path.append('./src/')
from h03_eval.eval import load_model
from util import argparser
from util import util


def get_args():
    # Data
    argparser.add_argument("--raw-file", type=str)
    argparser.add_argument('--batch-size', type=int, default=512)
    # Models
    argparser.add_argument('--checkpoints-path', type=str, required=True)
    argparser.add_argument('--model-type', type=str, required=True)
    # Bootstrap test
    argparser.add_argument('--n-samples', type=int, default=1000)

    args = argparser.parse_args()
    args.model_path = os.path.join(args.checkpoints_path, args.model_type)
    return args


def get_renyi_surprisal(hom_freqs):
    match = (hom_freqs * (hom_freqs - 1)) / 2
    n_items = hom_freqs.sum()

    return - np.log(match.sum() * 2 / (n_items * (n_items - 1))) / np.log(2)


def get_renyi(model, lex_size, batch_size):
    samples = []
    with tqdm(total=lex_size, desc='Sampling loop', leave=False) as tbar:
        while len(samples) < lex_size:
            batch_size = min(batch_size, lex_size - len(samples))
            samples += model.sample(batch_size)

            tbar.update(batch_size)

    sample_counts = Counter(samples)
    hom_counts = np.array(list(sample_counts.values()))
    renyi = get_renyi_surprisal(hom_counts)

    assert hom_counts.sum() == lex_size, \
        "Number of lstm samples should match lexicon size"
    return renyi


def bootstrap_renyi(model, n_samples, lex_size, batch_size, model_name, results_file):
    renyis = []
    for _ in tqdm(range(n_samples), desc='Bootstrap loop'):
        renyis += [get_renyi(model, lex_size, batch_size)]
        util.write_data(results_file, {
            'renyis': np.array(renyis),
            'name': model_name,
        })


def get_lex_size(raw_file):
    df = pd.read_csv(raw_file, sep='\t')
    freqs = df.homo.values
    return freqs.sum()


def main():
    args = get_args()
    lex_size = get_lex_size(args.raw_file)

    model = load_model(args.model_path, args.model_type)
    model = model.eval()
    model_name = args.model_path.split('/')[-1]

    results_file = '%s/samples.pckl' % (args.model_path)
    with torch.no_grad():
        bootstrap_renyi(model, args.n_samples, lex_size, args.batch_size, model_name, results_file)


if __name__ == '__main__':
    main()
