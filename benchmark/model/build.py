import argparse

import numpy as np
import pandas as pd
import scanpy as sc
import STAGATE_pyG as STAGATE
import torch
from anndata import AnnData


def train(adata: AnnData, batch_key: str | None = None, rad_cutoff: float = 25):
    print("Using device:", "cuda:0" if torch.cuda.is_available() else "cpu")

    if batch_key is None:
        STAGATE.Cal_Spatial_Net(adata, rad_cutoff=rad_cutoff)
    else:
        adatas = [
            adata[adata.obs[batch_key] == b].copy()
            for b in adata.obs[batch_key].unique()
        ]
        for adata_ in adatas:
            print("Batch:", adata_.obs[batch_key][0])
            STAGATE.Cal_Spatial_Net(adata_, rad_cutoff=rad_cutoff)

        adata = sc.concat(adatas)
        adata.uns["Spatial_Net"] = pd.concat(
            [adata_.uns["Spatial_Net"] for adata_ in adatas]
        )
        print("\nConcatenated:", adata)

    adata = STAGATE.train_STAGATE(adata)
    return adata
