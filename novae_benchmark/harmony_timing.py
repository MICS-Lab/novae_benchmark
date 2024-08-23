import time

import numpy as np
import scanpy as sc
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


@timing
def time_harmony(adata: AnnData):
    sc.external.pp.harmony_integrate(adata, key="slide_id", basis="SpaceFlow")


def main():
    # adata_full = sc.read_h5ad("/Users/quentinblampey/data/sandbox/colon_SpaceFlow_harmonized_30k.h5ad")
    adata_full = sc.read_h5ad("/gpfs/workdir/shared/prime/spatial/embeddings/colon_SpaceFlow_harmonized.h5ad")

    n_obs_list = []
    harmony_times = []

    for n_obs in [25_000, 100_000, 400_000, 1_600_000, 6_400_000]:
        if n_obs == adata_full.n_obs:
            adata = adata_full.copy()
        elif n_obs < adata_full.n_obs:
            adata = sc.pp.subsample(adata_full, n_obs=n_obs, copy=True)
        else:
            locs = np.random.choice(adata_full.obs.index, n_obs, replace=True)
            adata = adata_full[locs].copy()
            noise = np.random.randn(*adata.obsm["SpaceFlow"].shape) / 50
            adata.obsm["SpaceFlow"] = adata.obsm["SpaceFlow"] + noise

        n_obs_list.append(adata.n_obs)
        harmony_times.append(time_harmony(adata))

        print(f"{n_obs_list=}")
        print(f"{harmony_times=}")


if __name__ == "__main__":
    main()
