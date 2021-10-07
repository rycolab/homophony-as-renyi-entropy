# pylint: disable=anomalous-backslash-in-string
import os
import sys
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from util import util


aspect = {
    'height': 5,
    'font_scale': 1.3,
    'labels': True,
    'name_suffix': '',
    'ratio': 1.625,
}

fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 6.4
fig_size[1] = 3.5
plt.rcParams["figure.figsize"] = fig_size

sns.set_palette("Set2")
sns.set_context("notebook", font_scale=aspect['font_scale'])
mpl.rc('font', family='serif', serif='Times New Roman')
mpl.rc('text', usetex=True)


def plot_heatmap(data, size, minx, fname):
    fig = plt.figure(figsize=fig_size)
    plt.imshow(data, cmap='viridis', origin='lower', extent=(0, size, 0, size))

    plt.gca().set_yticklabels(['$10^{%d}$' % x for x in np.arange(0, size+0.01, size / 5)])
    plt.gca().set_xticklabels(['%.3f' % x for x in np.arange(minx, 1.001, (1 - minx) / 4)])
    plt.colorbar()

    plt.xlabel('$p(x_0)$')
    plt.ylabel('\# instances ($n$)')
    if fname:
        fig.savefig(fname, bbox_inches='tight')
    else:
        plt.show()

    plt.close()


def plot_entropies(fpath):
    size = 20
    minx = .8
    entropies = np.zeros((size, size))
    renyi_entropies = np.zeros((size, size))

    x = np.arange(minx, 1, (1 - minx) / size)

    for i in range(0, size):
        n = (10**(i)) + 1

        entropies[i, :] = - x * np.log2(x) - (1 - x) * np.log2((1 - x) / n)
        renyi_entropies[i, :] = - np.log2(x**2 + (1/n)*(1 - x)**2)

    plot_heatmap(entropies, size, minx, fname=os.path.join(fpath, 'shannon.pdf'))
    plot_heatmap(renyi_entropies, size, minx, fname=os.path.join(fpath, 'renyi.pdf'))


def main():
    fpath = 'results/plots'
    util.mkdir(fpath)

    plot_entropies(fpath)


if __name__ == '__main__':
    main()
