import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import scanpy as sc
import seaborn as sns
from anndata import AnnData

run_ID = f"ID_{random.randint(1, 100_000)}"
print(f"Run ID: {run_ID}")


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

    adata.write_h5ad(res_path / f"adata_{adata.n_obs}_{run_ID}.h5ad")

    colors = []
    for key in [
        "domain",
        "tissue",
        "technology",
        "novae_domains_15",
        "novae_domains_20",
        "novae_domains_22",
        "novae_domains_25",
    ]:
        if key in adata.obs:
            colors.append(key)

    for color in colors:
        sc.pl.umap(adata, color=color, show=False, title=title_dict.get(color, color))
        sns.despine(offset=10, trim=True)
        plt.savefig(res_path / f"umap_{adata.n_obs}_{color}_{run_ID}.png", bbox_inches="tight", dpi=300)


def main():
    # data_path = Path("/Users/quentinblampey/dev/novae/data/results/dry-wood-40")
    # res_path = Path("/Users/quentinblampey/dev/novae_benchmark/figures")
    data_path = Path("/gpfs/workdir/blampeyq/novae/data/results/still-surf-209")
    res_path = Path("/gpfs/workdir/blampeyq/novae_benchmark/figures")

    adata_full = sc.read_h5ad(data_path / "adata_conc.h5ad")

    print("adata:", adata_full)

    compute_and_save_umap(adata_full, res_path)


if __name__ == "__main__":
    main()
