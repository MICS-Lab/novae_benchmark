import logging
from pathlib import Path

import warnings
import anndata

from typing import Iterable
from itertools import chain

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.sparse import SparseEfficiencyWarning, block_diag, csr_matrix, spmatrix
from scipy.spatial import Delaunay
from sklearn.metrics.pairwise import euclidean_distances

from . import spatial_neighbors

def _spatial_neighbor(
    adata: AnnData,
    radius: float | tuple[float, float] | None = None,
    set_diag: bool = False,
    percentile: float | None = None,
) -> tuple[csr_matrix, csr_matrix]:
    coords = adata.obsm["spatial"]

    assert coords.shape[1] == 2, f"adata.obsm['spatial'] has {coords.shape[1]} dimension(s). Expected 2."

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SparseEfficiencyWarning)
        Adj, Dst = _build_connectivity(
            coords,
            set_diag=set_diag,
        )

    if isinstance(radius, Iterable):
        minn, maxx = sorted(radius)[:2]  # type: ignore[var-annotated]
        mask = (Dst.data < minn) | (Dst.data > maxx)
        a_diag = Adj.diagonal()

        Dst.data[mask] = 0.0
        Adj.data[mask] = 0.0
        Adj.setdiag(a_diag)

    if percentile is not None:
        threshold = np.percentile(Dst.data, percentile)
        Adj[Dst > threshold] = 0.0
        Dst[Dst > threshold] = 0.0

    Adj.eliminate_zeros()
    Dst.eliminate_zeros()

    return Adj, Dst

def _build_connectivity(
    coords: np.ndarray,
    set_diag: bool = False,
) -> csr_matrix | tuple[csr_matrix, csr_matrix]:
    N = coords.shape[0]

    tri = Delaunay(coords)
    indptr, indices = tri.vertex_neighbor_vertices
    Adj = csr_matrix((np.ones_like(indices, dtype=np.float64), indices, indptr), shape=(N, N))

    # fmt: off
    dists = np.array(list(chain(*(
        euclidean_distances(coords[indices[indptr[i] : indptr[i + 1]], :], coords[np.newaxis, i, :])
        for i in range(N)
        if len(indices[indptr[i] : indptr[i + 1]])
    )))).squeeze()
    Dst = csr_matrix((dists, indices, indptr), shape=(N, N))
    # fmt: on

    # radius-based filtering needs same indices/indptr: do not remove 0s
    Adj.setdiag(1.0 if set_diag else Adj.diagonal())
    Dst.setdiag(0.0)

    return Adj, Dst


def dummy_dataset(
    n_obs_per_domain: int = 1000,
    n_vars: int = 100,
    n_drop: int = 20,
    n_domains: int = 4,
    n_panels: int = 3,
    n_slides_per_panel: int = 1,
    panel_shift_factor: float = 0.5,
    batch_shift_factor: float = 0.2,
    class_shift_factor: float = 2,
    slide_ids_unique: bool = True,
    compute_spatial_neighbors: bool = True,
) -> list[AnnData]:
    """Creates a dummy dataset, useful for debugging or testing.

    Args:
        n_obs_per_domain: Number of obs per domain or niche.
        n_vars: Number of genes.
        n_drop: Number of genes that are removed for each `AnnData` object. It will create non-identical panels.
        n_domains: Number of domains, or niches.
        n_panels: Number of panels. Each panel will correspond to one output `AnnData` object.
        n_slides_per_panel: Number of slides per panel.
        panel_shift_factor: Shift factor for each panel.
        batch_shift_factor: Shift factor for each batch.
        class_shift_factor: Shift factor for each niche.
        slide_ids_unique: Whether to ensure that slide ids are unique.
        compute_spatial_neighbors: Whether to compute the spatial neighbors graph.

    Returns:
        A list of `AnnData` objects representing a valid `Novae` dataset.
    """

    panels_shift = [panel_shift_factor * np.random.randn(n_vars) for _ in range(n_panels)]
    domains_shift = [class_shift_factor * np.random.randn(n_vars) for _ in range(n_domains)]
    loc_shift = [np.array([0, 10 * i]) for i in range(n_domains)]

    adatas = []

    for panel_index in range(n_panels):
        X_, spatial_, domains_, slide_ids_ = [], [], [], []
        var_names = np.array([f"g{i}" for i in range(n_vars)])

        slide_key = f"slide_{panel_index}_" if slide_ids_unique else "slide_"
        if n_slides_per_panel > 1:
            slides_shift = np.array([batch_shift_factor * np.random.randn(n_vars) for _ in range(n_slides_per_panel)])

        for domain_index in range(n_domains):
            cell_shift = np.random.randn(n_obs_per_domain, n_vars)
            slide_ids_domain_ = np.random.randint(0, n_slides_per_panel, n_obs_per_domain)
            X_domain_ = cell_shift + domains_shift[domain_index] + panels_shift[panel_index]

            if n_slides_per_panel > 1:
                X_domain_ += slides_shift[slide_ids_domain_]

            X_.append(X_domain_)
            spatial_.append(np.random.randn(n_obs_per_domain, 2) + loc_shift[domain_index])
            domains_.append(np.array([f"domain_{domain_index}"] * n_obs_per_domain))
            slide_ids_.append(slide_ids_domain_)

        X = np.concatenate(X_, axis=0).clip(0)

        if n_drop > 0:
            var_indices = np.random.choice(n_vars, size=n_vars - n_drop, replace=False)
            X = X[:, var_indices]
            var_names = var_names[var_indices]

        adata = AnnData(X=X)

        adata.obs_names = [f"c_{panel_index}_{i}" for i in range(adata.n_obs)]
        adata.var_names = var_names
        adata.obs["domain"] = np.concatenate(domains_)
        adata.obs["slide_key"] = (slide_key + pd.Series(np.concatenate(slide_ids_)).astype(str)).values
        adata.obsm["spatial"] = np.concatenate(spatial_, axis=0)

        if compute_spatial_neighbors:
            spatial_neighbors(adata, radius=[0, 3])

        adatas.append(adata)

    return adatas