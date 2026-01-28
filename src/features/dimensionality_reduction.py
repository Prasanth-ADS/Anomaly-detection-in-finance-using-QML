import pandas as pd
from sklearn.decomposition import PCA, KernelPCA
from src.utils.logger import get_logger

logger = get_logger("dim_reduction")

def apply_pca(X, n_components=10):
    """Applies PCA to reduce dimensionality."""
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    explained_variance = pca.explained_variance_ratio_.sum()
    logger.info(f"PCA reduced to {n_components} components. Explained Variance: {explained_variance:.4f}")
    return X_pca, pca

def apply_feature_selection(X, y, k=10):
    """Applies SelectKBest."""
    from sklearn.feature_selection import SelectKBest, f_classif
    selector = SelectKBest(f_classif, k=k)
    X_new = selector.fit_transform(X, y)
    logger.info(f"Selected Top {k} features.")
    return X_new, selector

if __name__ == "__main__":
    # Test stub
    pass
