# homophony-as-renyi-entropy
This code accompanies the paper "On Homophony and Rényi Entropy".

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

And then install the appropriate version of pytorch:
```bash
$ conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
# $ conda install pytorch torchvision cpuonly -c pytorch
```


