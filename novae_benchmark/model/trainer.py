import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from sklearn.decomposition import PCA

from . import SEDR, STAGATE


class Model:
    def __init__(self, model_name: str, hidden_dim: int) -> None:
        super().__init__()
        self.model_name = model_name
        self.hidden_dim = hidden_dim

    def preprocess(self, adata: AnnData):
        adata.X = adata.layers["count"]
        sc.pp.normalize_total(adata)
        sc.pp.log1p(adata)
        return adata

    def train(self, adata: AnnData, batch_key: str | None, device: str = "cpu") -> None:
        raise NotImplementedError

    def inference(self, adata: AnnData) -> np.ndarray:
        assert self.model_name in adata.obsm.keys()

    def cluster(self, adata: AnnData, n_clusters: int):
        raise NotImplementedError

    def __call__(
        self, adata: AnnData, batch_key: str | None, n_clusters: int, device: str = "cpu"
    ) -> tuple[np.ndarray, pd.Series]:
        self.preprocess(adata)
        self.train(adata, batch_key, device)
        self.inference(adata)
        self.cluster(adata, n_clusters)
        return adata.obsm[self.model_name], adata.obs[self.model_name]


class STAGATEModel(Model):
    RAD_CUTOFF = 25

    def train(self, adata: AnnData, batch_key: str | None, device: str = "cpu"):
        if batch_key is None:
            STAGATE.Cal_Spatial_Net(adata, rad_cutoff=self.RAD_CUTOFF)
        else:
            adatas = [adata[adata.obs[batch_key] == b].copy() for b in adata.obs[batch_key].unique()]
            for adata_ in adatas:
                print("Batch:", adata_.obs[batch_key][0])
                STAGATE.Cal_Spatial_Net(adata_, rad_cutoff=self.RAD_CUTOFF)

            adata = sc.concat(adatas)
            adata.uns["Spatial_Net"] = pd.concat([adata_.uns["Spatial_Net"] for adata_ in adatas])
            print("\nConcatenated:", adata)

        adata = STAGATE.train_STAGATE(adata, key_added=self.model_name, device=device)
        return adata

    def cluster(self, adata: AnnData, n_clusters: int):
        STAGATE.mclust_R(adata, used_obsm=self.model_name, num_cluster=n_clusters)
        adata.obs[self.model_name] = adata.obs["m_clust"]


class SEDRModel(Model):
    def preprocess(self, adata: AnnData):
        adata.X = adata.layers["count"]
        sc.pp.normalize_total(adata, target_sum=1e6)
        sc.pp.scale(adata)

        adata_X = PCA(n_components=200, random_state=42).fit_transform(adata.X)
        adata.obsm["X_pca"] = adata_X

    def cluster(self, adata: AnnData, n_clusters: int):
        SEDR.mclust_R(adata, n_clusters, use_rep=self.model_name, key_added=self.model_name)

    def train(self, adata: AnnData, batch_key: str | None, device: str = "cpu"):
        graph_dict = SEDR.graph_construction(adata, 6)

        sedr_net = SEDR.Sedr(adata.obsm["X_pca"], graph_dict)
        using_dec = True
        if using_dec:
            sedr_net.train_with_dec()
        else:
            sedr_net.train_without_dec()
        sedr_feat, _, _, _ = sedr_net.process()
        adata.obsm[self.model_name] = sedr_feat


MODEL_DICT = {
    "STAGATE": STAGATEModel,
    "SEDR": SEDRModel,
}


def get_model(model_name: str, hidden_dim: int) -> Model:
    assert model_name in MODEL_DICT.keys()

    return MODEL_DICT[model_name](model_name, hidden_dim)
