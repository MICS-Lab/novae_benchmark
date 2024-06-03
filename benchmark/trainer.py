import numpy as np
import pandas as pd
import scanpy as sc
import STAGATE_pyG as STAGATE
import torch
from anndata import AnnData


class Model:
    def __init__(self, model_name: str, hidden_dim: int) -> None:
        super().__init__()
        self.model_name = model_name
        self.hidden_dim = hidden_dim

    def train(self, adata: AnnData, batch_key: str | None, device: str = "cpu") -> None:
        # adata should be preprocessed: normalized + log1p
        raise NotImplementedError

    def inference(self, adata: AnnData) -> np.ndarray:
        assert self.model_name in adata.obsm.keys()

    def __call__(self, adata: AnnData) -> np.ndarray:
        self.train(adata)
        self.inference(adata)
        return adata.obsm[self.model_name]


class STAGATEModel(Model):
    RAD_CUTOFF = 25

    def train(self, adata: AnnData, batch_key: str | None, device: str = "cpu"):
        if batch_key is None:
            STAGATE.Cal_Spatial_Net(adata, rad_cutoff=self.RAD_CUTOFF)
        else:
            adatas = [
                adata[adata.obs[batch_key] == b].copy()
                for b in adata.obs[batch_key].unique()
            ]
            for adata_ in adatas:
                print("Batch:", adata_.obs[batch_key][0])
                STAGATE.Cal_Spatial_Net(adata_, rad_cutoff=self.RAD_CUTOFF)

            adata = sc.concat(adatas)
            adata.uns["Spatial_Net"] = pd.concat(
                [adata_.uns["Spatial_Net"] for adata_ in adatas]
            )
            print("\nConcatenated:", adata)

        adata = STAGATE.train_STAGATE(adata, key_added="STAGATE", device=device)
        return adata


MODEL_DICT = {
    "STAGATE": STAGATEModel,
}


def get_model(model_name: str, hidden_dim: int) -> Model:
    assert model_name in MODEL_DICT.keys()

    return MODEL_DICT[model_name](model_name, hidden_dim)
