# homophony-as-renyi-entropy

[![CircleCI](https://circleci.com/gh/tpimentelms/homophony-as-renyi-entropy.svg?style=svg&circle-token=bf53770e696c076f2b148e9817449f62c39ba984)](https://circleci.com/gh/tpimentelms/homophony-as-renyi-entropy)

This code accompanies the paper [On Homophony and Rényi Entropy](https://arxiv.org/abs/2109.13766) published in EMNLP 2021.

## Data

Download the [CELEX data](https://catalog.ldc.upenn.edu/LDC96L14) and place the raw `LDC96L14.tar.gz` file into `data/celex/raw/` path.
You can then extract its data with command:
```bash
$ make get_celex
```

## Install

To install dependencies run:
```bash
$ conda env create -f environment.yml
```

Activate the created conda environment with command:
```bash
$ source activate.sh
```

Finally, install the appropriate version of pytorch:
```bash
$ conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
# $ conda install pytorch torchvision cpuonly -c pytorch
```

## Preprocess data

To preprocess a language's data run:
```bash
$ make get_data MONOMORPHEMIC=True LANGUAGE=<language>
```
where language can be one of: `eng` (English), `deu` (German), or `nld` (Dutch).

## Train and Evaluate models

