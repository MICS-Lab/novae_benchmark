import numpy as np
from anndata import AnnData

VALID_MODEL_NAMES = ["STAGATE", "SEDR", "GraphST", "SpaceFlow"]


class Model:
    def __init__(self, model_name: str, hidden_dim: int) -> None:
        super().__init__()
        assert model_name in VALID_MODEL_NAMES

        self.model_name = model_name
        self.hidden_dim = hidden_dim

        self.build_model()

    def build_model(self) -> None:
        raise NotImplementedError

    def train(self) -> None:
        raise NotImplementedError

    def inference(adata: AnnData) -> np.ndarray:
        raise NotImplementedError

    def __call__(self, adata: AnnData) -> np.ndarray:
        self.train()
        adata.obsm[self.model_name] = self.inference(adata)
        return adata.obsm[self.model_name]
