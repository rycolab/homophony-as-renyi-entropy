import sys
import os

sys.path.append('./src/')
from h02_learn.dataset import get_data_loaders
from h02_learn.model import LstmLM, NgramLM
from h02_learn.train_info import TrainInfo
from util import argparser
from util import util
from util import constants


def get_args():
    # Data
    argparser.add_argument('--batch-size', type=int, default=32)
    # Model
    argparser.add_argument('--nlayers', type=int, default=2)
    argparser.add_argument('--embedding-size', type=int, default=64)
    argparser.add_argument('--hidden-size', type=int, default=256)
    argparser.add_argument('--dropout', type=float, default=.33)
    argparser.add_argument('--model-type', type=str, required=True)
    # Optimization
    argparser.add_argument('--eval-batches', type=int, default=20)
    argparser.add_argument('--wait-epochs', type=int, default=5)
    # Save
    argparser.add_argument('--checkpoints-path', type=str)

    args = argparser.parse_args()
    args.wait_iterations = args.wait_epochs * args.eval_batches
    args.model_path = os.path.join(args.checkpoints_path, args.model_type)
    return args


def get_model(alphabet, args):
    if args.model_type in ['lstm']:
        model_cls = LstmLM
    elif args.model_type in ['ngram']:
        model_cls = NgramLM
    else:
        raise ValueError('Invalid model type requested %s' % args.model_type)

    return model_cls(
        alphabet, args.embedding_size, args.hidden_size,
        nlayers=args.nlayers, dropout=args.dropout) \
        .to(device=constants.device)


def eval_all(trainloader, devloader, testloader, model):
    train_loss = model.evaluate(trainloader)
    dev_loss = model.evaluate(devloader)
    test_loss = model.evaluate(testloader)

    print('Final Training loss: %.4f Dev loss: %.4f Test loss: %.4f' %
          (train_loss, dev_loss, test_loss))
    return (train_loss, dev_loss, test_loss)


def save_results(model, train_loss, dev_loss, test_loss, results_fname):
    args = model.get_args()
    del args['alphabet']
    results = [['name', 'train_loss', 'dev_loss', 'test_loss', 'alphabet_size'] +
               list(args.keys())]
    results += [[model.name, train_loss, dev_loss, test_loss, model.alphabet_size] +
                list(args.values())]
    util.write_csv(results_fname, results)


def save_checkpoints(model, train_loss, dev_loss, test_loss, model_path):
    model.save(model_path)
    results_fname = model_path + '/results.csv'
    save_results(model, train_loss, dev_loss, test_loss, results_fname)


def main():
    args = get_args()
    folds = [list(range(8)), [8], [9]]
    print(args)

    trainloader, devloader, testloader, alphabet = \
        get_data_loaders(args.data_file, folds, args.batch_size)
    print('Train size: %d Dev size: %d Test size: %d Alphabet size: %d' %
          (len(trainloader.dataset), len(devloader.dataset),
           len(testloader.dataset), len(alphabet)))

    train_info = TrainInfo(args.wait_iterations, args.eval_batches)
    model = get_model(alphabet, args)
    model.fit(trainloader, devloader, train_info)

    train_loss, dev_loss, test_loss = \
        eval_all(trainloader, devloader, testloader, model)

    save_checkpoints(model, train_loss, dev_loss, test_loss, args.model_path)


if __name__ == '__main__':
    main()
