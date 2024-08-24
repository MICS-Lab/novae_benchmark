import time
from pathlib import Path

import matplotlib.pyplot as plt
import scanpy as sc
import seaborn as sns
from anndata import AnnData


def timing(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Execution time: {execution_time} seconds")
        return execution_time

    return wrapper


title_dict = {
    "tissue": "Tissue",
    "technology": "Technology",
    "domain": "Domain",
}


@timing
def compute_and_save_umap(adata: AnnData, res_path: Path):
    print(f"{adata.n_obs=}")

    sc.pp.neighbors(adata, use_rep="novae_latent_corrected")
    sc.tl.umap(adata)

    colors = []
    for key in ["domain", "tissue", "technology", "novae_domains_15", "novae_domains_20", "novae_domains_25"]:
        if key in adata.obs:
            colors.append(key)

    for color in colors:
        sc.pl.umap(adata, color=color, show=False, title=title_dict.get(color, color))
        sns.despine(offset=10, trim=True)
        plt.savefig(res_path / f"umap_{adata.n_obs}_{color}.png", bbox_inches="tight", dpi=300)


def main():
    # data_path = Path("/Users/quentinblampey/dev/novae/data/results/dry-wood-40")
    # res_path = Path("/Users/quentinblampey/dev/novae_benchmark/figures")
    data_path = Path("/gpfs/workdir/blampeyq/novae/data/results/zany-night-17")
    res_path = Path("/gpfs/workdir/blampeyq/novae_benchmark/figures")

    adata_full = sc.read_h5ad(data_path / "adata_conc.h5ad")

    print("adata:", adata_full)

    for n_obs in [1_000, 1_000_000, 10_000_000, None]:
        if n_obs is not None and n_obs < adata_full.n_obs:
            adata = sc.pp.subsample(adata_full, n_obs=n_obs, copy=True)
        else:
            adata = adata_full

        compute_and_save_umap(adata, res_path)


if __name__ == "__main__":
    main()
