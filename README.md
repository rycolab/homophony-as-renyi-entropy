# homophony-as-renyi-entropy

[![CircleCI](https://circleci.com/gh/rycolab/homophony-as-renyi-entropy.svg?style=svg&circle-token=bf53770e696c076f2b148e9817449f62c39ba984)](https://circleci.com/gh/rycolab/homophony-as-renyi-entropy)

This code accompanies the paper [On Homophony and Rényi Entropy (Pimentel et al., EMNLP 2021)](https://arxiv.org/abs/2109.13766).
It is a study of the pressures of homophony in language, analysing homophony through the lens of the Rényi collision entropy.

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

## Train models

To train a language's phonotactic model run:
```bash
$ make train MONOMORPHEMIC=True LANGUAGE=<language> MODEL=<model>
```
where model can be one of: `lstm`, or `ngram`.

## Evaluate models

There are three commands to evaluate the trained phonotactic models.
The first evaluates it on the test set to get its cross-entropy:
```bash
$ make eval MONOMORPHEMIC=True LANGUAGE=<language> MODEL=<model>
```

The second analyses all words with probability above a threshold delta to approximate its renyi entropy:
```bash
$ make get_renyi MONOMORPHEMIC=True LANGUAGE=<language> MODEL=<model>
```

Finally, the third samples artificial lexica from the language models' to run the null hypothesis test:
```bash
$ make sample_renyi MONOMORPHEMIC=True LANGUAGE=<language> MODEL=<model>
```


## Analyse models

Finally, to analyse the models and print results run:
```bash
$ make analyse MONOMORPHEMIC=True LANGUAGE=<language>
```


## Extra Information

#### Citation

If this code or the paper were usefull to you, consider citing it:


```bash
@inproceedings{pimentel-etal-2021-homophony,
    title = "On Homophony and Rényi Entropy",
    author = "Pimentel, Tiago and
    Meister, Clara and
    Teufel, Simone and
    Cotterell, Ryan",
    booktitle = "Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
    year = "2021",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/2109.13766",
}
```


#### Contact

To ask questions or report problems, please open an [issue](https://github.com/rycolab/homophony-as-renyi-entropy/issues).
