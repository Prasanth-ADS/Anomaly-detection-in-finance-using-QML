import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from src.utils.logger import get_logger

logger = get_logger("dimensionality_reduction")

class DimensionalityReducer:
    def __init__(self, method='pca', n_components=4, feature_range=(0, 1)):
        """
        Args:
            method (str): 'pca' or 'autoencoder'
            n_components (int): Target dimension (qubits)
            feature_range (tuple): Normalization range, e.g., (0, 1) or (-np.pi, np.pi)
        """
        self.method = method
        self.n_components = n_components
        self.feature_range = feature_range
        self.scaler = MinMaxScaler(feature_range=feature_range)
        self.model = None

    def fit_transform(self, X):
        """
        Fit reduction model and transform data.
        """
        # 1. Fit PCA
        if self.method == 'pca':
            self.model = PCA(n_components=self.n_components)
            X_reduced = self.model.fit_transform(X)
            
            explained_variance = np.sum(self.model.explained_variance_ratio_)
            logger.info(f"PCA Explain Variance ({self.n_components} components): {explained_variance:.4f}")
            
        elif self.method == 'autoencoder':
            # Placeholder for AE logic or import from classical.autoencoder
            # For now, simplistic PCA fallback or error
             logger.warning("Autoencoder reduction not fully integrated in wrapper yet. Using PCA.")
             self.model = PCA(n_components=self.n_components)
             X_reduced = self.model.fit_transform(X)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        # 2. Normalize to feature_range (crucial for Quantum Embedding)
        X_scaled = self.scaler.fit_transform(X_reduced)
        return X_scaled

    def transform(self, X):
        """
        Apply existing reduction and scaling.
        """
        if self.model is None:
             raise RuntimeError("Must call fit_transform first.")
        
        X_reduced = self.model.transform(X)
        X_scaled = self.scaler.transform(X_reduced)
        return X_scaled
