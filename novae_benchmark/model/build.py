import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from sklearn.decomposition import PCA

from . import SEDR, STAGATE

DEFAULT_N_CLUSTERS = 7


class Model:
    def __init__(self, model_name: str, hidden_dim: int) -> None:
        super().__init__()
        self.model_name = model_name
        self.hidden_dim = hidden_dim

    def preprocess(self, adata: AnnData):
        """
        Preprocess the data before training the model. Raw counts can be found in `adata.layers["counts"]`
        """
        adata.X = adata.layers["counts"]
        sc.pp.normalize_total(adata)
        sc.pp.log1p(adata)

    def train(
        self, adata: AnnData, batch_key: str | None = None, device: str = "cpu", fast_dev_run: bool = False
    ) -> None:
        """
        Train the model. Use `fast_dev_run` to run only a few epochs (for testing purposes).
        """
        raise NotImplementedError

    def inference(self, adata: AnnData) -> np.ndarray:
        """
        Runs inference. The output should be stored in `adata.obsm[self.model_name]`.
        """
        assert self.model_name in adata.obsm.keys()

    def cluster(self, adata: AnnData, n_clusters: int = DEFAULT_N_CLUSTERS):
        """
        Clusters the data. The output should be stored in `adata.obs[self.model_name]`.
        """
        raise NotImplementedError

    def __call__(
        self,
        adata: AnnData,
        n_clusters: int = DEFAULT_N_CLUSTERS,
        batch_key: str | None = None,
        device: str = "cpu",
        fast_dev_run: bool = False,
    ) -> tuple[np.ndarray, pd.Series]:
        """
        Runs all steps, i.e preprocessing -> training -> inference -> clustering.

        Returns:
            A numpy array of shape (n_cells, hidden_dim) and a pandas Series with the cluster labels.
        """
        self.preprocess(adata)
        self.train(adata, batch_key=batch_key, device=device, fast_dev_run=fast_dev_run)
        self.inference(adata)
        self.cluster(adata, n_clusters)

        adata.obs[self.model_name] = adata.obs[self.model_name].astype("category")

        assert adata.obsm[self.model_name].shape[1] == self.hidden_dim
        assert len(adata.obs[self.model_name].cat.categories) == n_clusters

        return adata.obsm[self.model_name], adata.obs[self.model_name]


class STAGATEModel(Model):
    RAD_CUTOFF = 25

    def train(self, adata: AnnData, batch_key: str | None = None, device: str = "cpu", fast_dev_run: bool = False):
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

        adata = STAGATE.train_STAGATE(
            adata, key_added=self.model_name, device=device, n_epochs=2 if fast_dev_run else 1000
        )
        return adata

    def cluster(self, adata: AnnData, n_clusters: int = DEFAULT_N_CLUSTERS):
        STAGATE.mclust_R(adata, used_obsm=self.model_name, num_cluster=n_clusters)
        adata.obs[self.model_name] = adata.obs["mclust"]


class SEDRModel(Model):
    def preprocess(self, adata: AnnData):
        adata.X = adata.layers["counts"]
        sc.pp.normalize_total(adata, target_sum=1e6)
        sc.pp.scale(adata)

        adata_X = PCA(n_components=200, random_state=42).fit_transform(adata.X)
        adata.obsm["X_pca"] = adata_X

    def cluster(self, adata: AnnData, n_clusters: int):
        SEDR.mclust_R(adata, n_clusters, use_rep=self.model_name, key_added=self.model_name)

    def train(self, adata: AnnData, batch_key: str | None = None, device: str = "cpu", fast_dev_run: bool = False):
        graph_dict = SEDR.graph_construction(adata, 6)

        sedr_net = SEDR.Sedr(adata.obsm["X_pca"], graph_dict, device=device)
        using_dec = True
        if using_dec:
            sedr_net.train_with_dec(epochs=2 if fast_dev_run else 200)
        else:
            sedr_net.train_without_dec(epochs=2 if fast_dev_run else 200)
        sedr_feat, _, _, _ = sedr_net.process()
        adata.obsm[self.model_name] = sedr_feat


MODEL_DICT = {
    "STAGATE": STAGATEModel,
    "SEDR": SEDRModel,
}


def get_model(model_name: str, hidden_dim: int = 64) -> Model:
    assert model_name in MODEL_DICT.keys()

    return MODEL_DICT[model_name](model_name, hidden_dim)
