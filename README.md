# Novae benchmark

This repository is used to benchmark `novae` for the manuscript (this does not contain the code of novae, only the benchmark)

See the [official `novae` repository](https://github.com/MICS-Lab/novae).

## Setup

### Poetry setup

```sh
poetry install --all-extras
```

### Conda setup

For some algorithms (e.g., STAGATE/SEDR), R might be needed for clustering.

```sh
pip install -e .
conda install conda-forge::r-base
conda install conda-forge::r-mclust
```
