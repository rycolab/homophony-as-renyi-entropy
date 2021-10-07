import sys
import math
import torch

sys.path.append('./src/')
from h03_eval.eval import load_model, get_args
from util import util


def get_renyi(model, batch_size):
    probs, lengths = model.get_probabilities(batch_size)

    renyi = - (torch.log(torch.pow(probs, 2).sum()) / math.log(2)).item()
    shannon = - (torch.log(probs).mean() / math.log(2)).item()

    print('Renyi entropy: %.4f Shannon entropy: %.4f' %
          (renyi, shannon))

    return {
        'probs': probs.numpy(),
        'lengths': lengths.numpy(),
        'renyi': renyi,
        'shannon': shannon,
    }


def main():
    args = get_args()

    model = load_model(args.model_path, args.model_type)
    model = model.eval()
    model_name = args.model_path.split('/')[-1]

    with torch.no_grad():
        results = get_renyi(model, args.batch_size)
        results['name'] = model_name

    results_file = '%s/renyi.pckl' % (args.model_path)
    util.write_data(results_file, results)


if __name__ == '__main__':
    main()
