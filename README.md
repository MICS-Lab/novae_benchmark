# Novae benchmark

⚠️ **WARNING:** This repository is used to benchmark `novae` for the manuscript. It does **not** contain the code of novae, only the benchmark.

‼️ See the [official `novae` repository](https://github.com/MICS-Lab/novae).

## Setup

### Poetry setup

Poetry can be used for development. Yet, it will not include R. To train models, use the conda setup below.

```sh
poetry install --all-extras
```

### Conda setup

For some algorithms (e.g., STAGATE/SEDR), R might be needed for clustering.

```sh
pip install -e .
conda install conda-forge::r-mclust
```
