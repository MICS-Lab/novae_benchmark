from __future__ import annotations

import numpy as np
import scanpy as sc
from anndata import AnnData
from sklearn import metrics


ALL_METRICS = ["FIDE", "JSD", "SVG"]
ADJ = "spatial_distances"
EPS = 1e-8


def mean_fide_score(
    adatas: AnnData | list[AnnData], obs_key: str, slide_key: str = None, n_classes: int | None = None
) -> float:
    """Mean FIDE score over all slides. A low score indicates a great domain continuity.

    Args:
        adatas: An `AnnData` object, or a list of `AnnData` objects.
        {obs_key}
        {slide_key}
        n_classes: Optional number of classes. This can be useful if not all classes are predicted, for a fair comparision.

    Returns:
        The FIDE score averaged for all slides.
    """
    return np.mean(
        [fide_score(adata, obs_key, n_classes=n_classes) for adata in _iter_uid(adatas, slide_key=slide_key)]
    )


def fide_score(adata: AnnData, obs_key: str, n_classes: int | None = None) -> float:
    """F1-score of intra-domain edges (FIDE). A high score indicates a great domain continuity.

    Note:
        The F1-score is computed for every class, then all F1-scores are averaged. If some classes
        are not predicted, the `n_classes` argument allows to pad with zeros before averaging the F1-scores.

    Args:
        adata: An `AnnData` object
        {obs_key}
        n_classes: Optional number of classes. This can be useful if not all classes are predicted, for a fair comparision.

    Returns:
        The FIDE score.
    """
    adata.obs[obs_key] = adata.obs[obs_key].astype("category")

    i_left, i_right = adata.obsp[ADJ].nonzero()
    classes_left, classes_right = adata.obs.iloc[i_left][obs_key].values, adata.obs.iloc[i_right][obs_key].values

    where_valid = ~classes_left.isna() & ~classes_right.isna()
    classes_left, classes_right = classes_left[where_valid], classes_right[where_valid]

    f1_scores = metrics.f1_score(classes_left, classes_right, average=None)

    if n_classes is None:
        return f1_scores.mean()

    assert n_classes >= len(f1_scores), f"Expected {n_classes:=}, but found {len(f1_scores)}, which is greater"

    return np.pad(f1_scores, (0, n_classes - len(f1_scores))).mean()


# def jensen_shannon_divergence(adatas: AnnData | list[AnnData], obs_key: str, slide_key: str = None) -> float:
#     """Jensen-Shannon divergence (JSD) over all slides

#     Args:
#         adatas: One or a list of AnnData object(s)
#         {obs_key}
#         {slide_key}

#     Returns:
#         The Jensen-Shannon divergence score for all slides
#     """
#     distributions = [
#         adata.obs[obs_key].value_counts(sort=False).values
#         for adata in _iter_uid(adatas, slide_key=slide_key, obs_key=obs_key)
#     ]

#     return _jensen_shannon_divergence(np.array(distributions))

import numpy as np

def jensen_shannon_divergence(adatas: AnnData | list[AnnData], obs_key: str, slide_key: str = None) -> float:
    """Jensen-Shannon divergence (JSD) over all slides

    Args:
        adatas: One or a list of AnnData object(s)
        obs_key: The key in the `obs` DataFrame for the categorical variable of interest
        slide_key: The key in the `obs` DataFrame for the slide or batch variable

    Returns:
        The Jensen-Shannon divergence score for all slides
    """
    # Identify all possible categories across all AnnData objects
    all_categories = set()
    for adata in _iter_uid(adatas, slide_key=slide_key, obs_key=obs_key):
        all_categories.update(adata.obs[obs_key].cat.categories)

    # Convert categories set to a sorted list
    all_categories = sorted(all_categories)

    # Create the distributions, ensuring all categories are represented
    distributions = []
    for adata in _iter_uid(adatas, slide_key=slide_key, obs_key=obs_key):
        # Get the value counts, using the full list of categories
        value_counts = adata.obs[obs_key].value_counts(sort=False)
        distribution = np.zeros(len(all_categories))
        
        for i, category in enumerate(all_categories):
            if category in value_counts:
                distribution[i] = value_counts[category]
        
        distributions.append(distribution)
    
    # Convert to numpy array
    distributions = np.array(distributions)

    return _jensen_shannon_divergence(distributions)



def mean_svg_score(adata: AnnData | list[AnnData], obs_key: str, slide_key: str = None, n_top_genes: int = 3) -> float:
    """Mean SVG score over all slides. A high score indicates better niche-specific genes, or spatial variable genes.

    Args:
        adata: An `AnnData` object, or a list.
        {obs_key}
        {slide_key}
        {n_top_genes}

    Returns:
        The mean SVG score accross all slides.
    """
    return np.mean(
        [svg_score(adata, obs_key, n_top_genes=n_top_genes) for adata in _iter_uid(adata, slide_key=slide_key)]
    )


def svg_score(adata: AnnData, obs_key: str, n_top_genes: int = 3) -> float:
    """Average score of the top differentially expressed genes for each niche.

    Args:
        adata: An `AnnData` object
        {obs_key}
        {n_top_genes}

    Returns:
        The average SVG score.
    """
    sc.tl.rank_genes_groups(adata, groupby=obs_key)
    sub_recarray: np.recarray = adata.uns["rank_genes_groups"]["scores"][:n_top_genes]
    return np.mean([sub_recarray[field].mean() for field in sub_recarray.dtype.names])


def _jensen_shannon_divergence(distributions: np.ndarray) -> float:
    """Compute the Jensen-Shannon divergence (JSD) for a multiple probability distributions.

    The lower the score, the better distribution of clusters among the different batches.

    Args:
        distributions: An array of shape (B x C), where B is the number of batches, and C is the number of clusters. For each batch, it contains the percentage of each cluster among cells.

    Returns:
        A float corresponding to the JSD
    """
    distributions = distributions / distributions.sum(1)[:, None]
    mean_distribution = np.mean(distributions, 0)

    return _entropy(mean_distribution) - np.mean([_entropy(dist) for dist in distributions])


def _entropy(distribution: np.ndarray) -> float:
    """Shannon entropy

    Args:
        distribution: An array of probabilities (should sum to one)

    Returns:
        The Shannon entropy
    """
    return -(distribution * np.log(distribution + EPS)).sum()


def _iter_uid(adatas: AnnData | list[AnnData], slide_key: str | None = None, obs_key: str | None = None):
    if isinstance(adatas, AnnData):
        adatas = [adatas]

    if obs_key is not None:
        categories = set.union(*[set(adata.obs[obs_key].astype("category").cat.categories) for adata in adatas])
        for adata in adatas:
            adata.obs[obs_key] = adata.obs[obs_key].astype("category").cat.set_categories(categories)

    for adata in adatas:
        if slide_key is not None:
            for slide_id in adata.obs[slide_key].unique():
                yield adata[adata.obs[slide_key] == slide_id].copy()
        else:
            yield adata


def evaluate_latent(adatas: AnnData | list[AnnData],
                     obs_key: str, slide_key: str = None, 
                     n_classes: int | None = None, n_top_genes: int = 3):
    eval_dt = {}
    eval_dt["FIDE"] = mean_fide_score(adatas=adatas, obs_key=obs_key, slide_key=slide_key, n_classes=n_classes)
    eval_dt["JSD"] = jensen_shannon_divergence(adatas=adatas, obs_key=obs_key, slide_key=slide_key)
    #eval_dt["SVG"] = mean_svg_score(adata=adatas, obs_key=obs_key, slide_key=slide_key, n_top_genes=n_top_genes)
    return eval_dt



