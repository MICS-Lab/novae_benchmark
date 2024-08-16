import time

import anndata
import numpy as np
import pandas as pd
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
    sc.pp.neighbors(adata)
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

    res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(adata.X), 10, "EEE")
    mclust_res = np.array(res[-2])

    adata.obs["mclust"] = mclust_res
    adata.obs["mclust"] = adata.obs["mclust"].astype("int").astype("category")


def dummy_dataset(
    n_panels: int = 3,
    n_domains: int = 4,
    n_slides_per_panel: int = 1,
    xmax: int = 500,
    n_vars: int = 100,
    n_drop: int = 20,
    step: int = 20,
    panel_shift_lambda: float = 0.25,
    slide_shift_lambda: float = 0.5,
    domain_shift_lambda: float = 0.25,
    slide_ids_unique: bool = True,
) -> list[AnnData]:
    """Creates a dummy dataset, useful for debugging or testing.

    Args:
        n_panels: Number of panels. Each panel will correspond to one output `AnnData` object.
        n_domains: Number of domains.
        n_slides_per_panel: Number of slides per panel.
        xmax: Maximum value for the spatial coordinates (the larger, the more cells).
        n_vars: Maxmium number of genes per panel.
        n_drop: Number of genes that are randomly removed for each `AnnData` object. It will create non-identical panels.
        step: Step between cells in their spatial coordinates.
        panel_shift_lambda: Lambda used in the exponential law for each panel.
        slide_shift_lambda: Lambda used in the exponential law for each slide.
        domain_shift_lambda: Lambda used in the exponential law for each domain.
        slide_ids_unique: Whether to ensure that slide ids are unique.

    Returns:
        A list of `AnnData` objects representing a valid `Novae` dataset.
    """
    assert n_vars - n_drop - n_panels > 2

    spatial = np.mgrid[-xmax:xmax:step, -xmax:xmax:step].reshape(2, -1).T
    spatial = spatial[(spatial**2).sum(1) <= xmax**2]
    n_obs = len(spatial)

    domain = "domain_" + (np.sqrt((spatial**2).sum(1)) // (xmax / n_domains + 1e-8)).astype(int).astype(str)

    adatas = []

    var_names = np.array([f"g{i}" for i in range(n_vars)])

    lambdas_per_domain = np.random.exponential(1, size=(n_domains, n_vars))

    for panel_index in range(n_panels):
        adatas_panel = []
        panel_shift = np.random.exponential(panel_shift_lambda, size=n_vars)

        for slide_index in range(n_slides_per_panel):
            slide_shift = np.random.exponential(slide_shift_lambda, size=n_vars)

            adata = AnnData(
                np.zeros((n_obs, n_vars)),
                obsm={"spatial": spatial + panel_index + slide_index},  # ensure the locs are different
                obs=pd.DataFrame({"domain": domain}, index=[f"cell_{i}" for i in range(spatial.shape[0])]),
            )

            adata.var_names = var_names
            adata.obs_names = [f"c_{panel_index}_{slide_index}_{i}" for i in range(adata.n_obs)]

            slide_key = f"slide_{panel_index}_{slide_index}" if slide_ids_unique else f"slide_{slide_index}"
            adata.obs["slide_key"] = slide_key

            for i in range(n_domains):
                condition = adata.obs["domain"] == "domain_" + str(i)
                n_obs_domain = condition.sum()

                domain_shift = np.random.exponential(domain_shift_lambda, size=n_vars)
                lambdas = lambdas_per_domain[i] + domain_shift + slide_shift + panel_shift
                X_domain = np.random.exponential(lambdas, size=(n_obs_domain, n_vars))
                adata.X[condition] = X_domain.clip(0, 9)  # values should look like log1p values

            if n_drop:
                size = n_vars - n_drop - panel_index  # different number of genes
                var_indices = np.random.choice(n_vars, size=size, replace=False)
                adata = adata[:, var_indices].copy()

            adatas_panel.append(adata[: -1 - panel_index - slide_index].copy())  # different number of cells

        adata_panel = anndata.concat(adatas_panel)

        adatas.append(adata_panel)

    return adatas


def main():
    n_obs_list = []
    leiden_times = []
    mclust_times = []

    for xmax in [2_000, 3_000, 4_500, 7_000, 10_500, 15_000, 22_500]:
        adata = dummy_dataset(n_panels=1, n_drop=0, n_vars=64, n_domains=10, xmax=xmax)[0]

        n_obs_list.append(adata.n_obs)
        leiden_times.append(time_leiden(adata))
        # mclust_times.append(time_mclust(adata))

        print(f"{xmax=}")
        print(f"{n_obs_list=}")
        print(f"{leiden_times=}")
        print(f"{mclust_times=}")


if __name__ == "__main__":
    main()
