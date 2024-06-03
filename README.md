# Novae benchmark

⚠️ **WARNING:** This repository is used to benchmark `novae` for the manuscript. It does **not** contain the code of novae, only the benchmark.

‼️ See the [official `novae` repository](https://github.com/MICS-Lab/novae).

## Setup

### Poetry setup

Poetry can be used for development. Yet, it will not include R. To train models, use the conda setup below.

At the root of the repository, run the following command line:

```sh
poetry install --all-extras
```

### Conda setup

For some algorithms (e.g., STAGATE/SEDR), R might be needed for clustering.

At the root of the repository, run the following command lines:

```sh
conda create -n novae_benchmark python=3.10 -y
pip install -e .
conda install -c conda-forge r-base rpy2 r-mclust
```
