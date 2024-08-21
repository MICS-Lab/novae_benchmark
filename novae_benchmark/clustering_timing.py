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
def time_leiden(adata: AnnData):
    sc.pp.neighbors(adata, use_rep="SpaceFlow")
    sc.tl.leiden(adata)


@timing
def time_mclust(adata: AnnData):
    import rpy2.robjects as robjects

    robjects.r.library("mclust")

    import rpy2.robjects.numpy2ri

    rpy2.robjects.numpy2ri.activate()

    r_random_seed = robjects.r["set.seed"]
    r_random_seed(0)
    rmclust = robjects.r["Mclust"]

    res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(adata.obsm["SpaceFlow"].values), 7, "EEE")
    mclust_res = np.array(res[-2])

    adata.obs["mclust"] = mclust_res
    adata.obs["mclust"] = adata.obs["mclust"].astype("int").astype("category")


def main():
    # adata_full = sc.read_h5ad("/Users/quentinblampey/data/sandbox/colon_SpaceFlow_harmonized_30k.h5ad")
    adata_full = sc.read_h5ad("/gpfs/workdir/shared/prime/spatial/embeddings/colon_SpaceFlow_harmonized.h5ad")

    n_obs_list = []
    leiden_times = []
    mclust_times = []

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
        leiden_times.append(time_leiden(adata))
        try:
            mclust_times.append(time_mclust(adata))
        except Exception as e:
            print(f"mclust failed for {n_obs=}")
            print("Error:", e)
            mclust_times.append(None)

        print(f"{n_obs_list=}")
        print(f"{leiden_times=}")
        print(f"{mclust_times=}")


if __name__ == "__main__":
    main()
