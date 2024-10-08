import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from sklearn.decomposition import PCA

from . import SEDR, STAGATE, SpaceFlow, cluster_utils, eval_utils, GraphST

DEFAULT_N_CLUSTERS = 7
DEFAULT_RADIUS_CLUSTERS = 50


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

    def cluster(self, adata: AnnData, n_clusters: int = DEFAULT_N_CLUSTERS, 
                method: str = "mclust", pca: bool = False):
        """
        Clusters the data. The output should be stored in `adata.obs[self.model_name]`.
        """
        cluster_utils.clustering(adata=adata, model_name=self.model_name, 
                                 n_clusters=n_clusters, method=method, pca=pca)
        
    def evaluate(self, adata: AnnData, batch_key: str | None = None, n_clusters: int = DEFAULT_N_CLUSTERS,
                 n_top_genes: int =3):
        self.model_performances = eval_utils.evaluate_latent(adatas=adata, obs_key=self.model_name, slide_key=batch_key,
                                          n_classes=n_clusters, n_top_genes=n_top_genes)


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
        print("--------------- {}: Preprocessing Started-------------------\n".format(self.model_name))
        self.preprocess(adata)
        print("--------------- {}: Preprocessing Finished-------------------\n".format(self.model_name))
        print("--------------- {}: Training Started-------------------\n".format(self.model_name))
        self.train(adata, batch_key=batch_key, device=device, fast_dev_run=fast_dev_run)
        print("--------------- {}: Training Finished-------------------\n".format(self.model_name))
        print("--------------- {}: Clustering Started-------------------\n".format(self.model_name))
        self.cluster(adata, n_clusters)
        print("--------------- {}: Clustering Finished-------------------\n".format(self.model_name))
        self.evaluate(adata, batch_key, n_clusters)
        print("--------------- {}: Evaluation completed-------------------\n".format(self.model_name))
        print(self.model_performances)


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


class SEDRModel(Model):
    def preprocess(self, adata: AnnData):
        adata.X = adata.layers["counts"]
        sc.pp.normalize_total(adata, target_sum=1e6)
        sc.pp.scale(adata)

        adata_X = PCA(n_components=200, random_state=42).fit_transform(adata.X)
        adata.obsm["X_pca"] = adata_X


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


class SpaceFlowModel(Model):
    N_TOP_GENES = 200

    def preprocess(self, adata: AnnData):
        pass

    def train(self, adata: AnnData, batch_key: str | None = None, device: str = "cpu", fast_dev_run: bool = False):
        spaceflow_net = SpaceFlow.Spaceflow(adata=adata)
        spaceflow_net.preprocessing_data(n_top_genes=self.N_TOP_GENES, batch_key=batch_key)
        spaceflow_embedding = spaceflow_net.train(z_dim=self.hidden_dim, epochs=2 if fast_dev_run else 1000)
        adata.obsm[self.model_name] = spaceflow_embedding


class GraphSTModel(Model):
    def train(self, adata: AnnData, batch_key: str | None = None, device: str = "cpu", fast_dev_run: bool = False):
        graphst_net = GraphST.Graphst(adata=adata, device=device, epochs=2 if fast_dev_run else 1000)
        adata = graphst_net.train()


MODEL_DICT = {
    "STAGATE": STAGATEModel,
    "SEDR": SEDRModel,
    "SpaceFlow": SpaceFlowModel,
    "GraphST": GraphSTModel,
}


def get_model(model_name: str, hidden_dim: int = 64) -> Model:
    assert model_name in MODEL_DICT.keys()

    return MODEL_DICT[model_name](model_name, hidden_dim)
