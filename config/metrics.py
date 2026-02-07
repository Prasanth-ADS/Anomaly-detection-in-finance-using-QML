# Metrics Computation Module
"""
Unified metric computation functions for all models.
Ensures consistent evaluation across classical, neural, and QML models.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    mean_squared_error,
    mean_absolute_error,
)
from typing import Dict, Optional, Tuple
import time


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Compute classification metrics for anomaly detection.
    
    Parameters:
    -----------
    y_true : array-like
        True labels (0 = normal, 1 = anomaly)
    y_pred : array-like
        Predicted labels
    y_prob : array-like, optional
        Predicted probabilities for positive class
        
    Returns:
    --------
    dict : Dictionary of metric names and values
    """
    metrics = {}
    
    # Basic classification metrics
    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    metrics["precision"] = precision_score(y_true, y_pred, zero_division=0)
    metrics["recall"] = recall_score(y_true, y_pred, zero_division=0)
    metrics["f1_score"] = f1_score(y_true, y_pred, zero_division=0)
    
    # Confusion matrix for FPR
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics["fpr"] = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    metrics["tpr"] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    # ROC-AUC (requires probabilities)
    if y_prob is not None:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
        except ValueError:
            metrics["roc_auc"] = 0.5  # Default for single-class edge case
    else:
        metrics["roc_auc"] = None
    
    return metrics


def compute_reconstruction_metrics(
    X_original: np.ndarray,
    X_reconstructed: np.ndarray
) -> Dict[str, float]:
    """
    Compute reconstruction-based metrics for autoencoders.
    
    Parameters:
    -----------
    X_original : array-like
        Original input data
    X_reconstructed : array-like
        Reconstructed data from autoencoder
        
    Returns:
    --------
    dict : Dictionary of reconstruction metrics
    """
    metrics = {}
    
    # Per-sample reconstruction error
    reconstruction_errors = np.mean((X_original - X_reconstructed) ** 2, axis=1)
    
    metrics["mse"] = mean_squared_error(X_original, X_reconstructed)
    metrics["mae"] = mean_absolute_error(X_original, X_reconstructed)
    metrics["reconstruction_error_mean"] = np.mean(reconstruction_errors)
    metrics["reconstruction_error_std"] = np.std(reconstruction_errors)
    
    return metrics


def compute_generative_metrics(
    log_likelihoods: np.ndarray,
    y_true: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Compute metrics for generative models (GMM, VAE).
    
    Parameters:
    -----------
    log_likelihoods : array-like
        Log-likelihood scores for each sample
    y_true : array-like, optional
        True labels for class-wise analysis
        
    Returns:
    --------
    dict : Dictionary of generative metrics
    """
    metrics = {}
    
    metrics["log_likelihood_mean"] = np.mean(log_likelihoods)
    metrics["log_likelihood_std"] = np.std(log_likelihoods)
    metrics["negative_log_likelihood"] = -np.mean(log_likelihoods)
    
    if y_true is not None:
        # Separate by class
        normal_ll = log_likelihoods[y_true == 0]
        anomaly_ll = log_likelihoods[y_true == 1]
        
        if len(normal_ll) > 0:
            metrics["normal_ll_mean"] = np.mean(normal_ll)
        if len(anomaly_ll) > 0:
            metrics["anomaly_ll_mean"] = np.mean(anomaly_ll)
    
    return metrics


def compute_qml_metrics(
    n_qubits: int,
    circuit_depth: int,
    n_shots: int,
    training_time: float,
    inference_time: float
) -> Dict[str, float]:
    """
    Compute quantum-specific metrics.
    
    Parameters:
    -----------
    n_qubits : int
        Number of qubits used
    circuit_depth : int
        Depth of quantum circuit
    n_shots : int
        Number of measurement shots
    training_time : float
        Training time in seconds
    inference_time : float
        Inference time in seconds
        
    Returns:
    --------
    dict : Dictionary of QML-specific metrics
    """
    return {
        "n_qubits": n_qubits,
        "circuit_depth": circuit_depth,
        "n_shots": n_shots,
        "training_time_seconds": training_time,
        "inference_time_seconds": inference_time,
        "circuit_complexity": n_qubits * circuit_depth,
    }


def get_roc_curve_data(
    y_true: np.ndarray,
    y_prob: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Get ROC curve data for plotting.
    
    Returns:
    --------
    tuple : (fpr, tpr, thresholds)
    """
    return roc_curve(y_true, y_prob)


class Timer:
    """Context manager for timing operations."""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.elapsed = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, *args):
        self.end_time = time.time()
        self.elapsed = self.end_time - self.start_time


def anomaly_score_to_prediction(
    scores: np.ndarray,
    threshold: Optional[float] = None,
    percentile: float = 95
) -> Tuple[np.ndarray, float]:
    """
    Convert anomaly scores to binary predictions.
    
    Parameters:
    -----------
    scores : array-like
        Anomaly scores (higher = more anomalous)
    threshold : float, optional
        Fixed threshold. If None, uses percentile.
    percentile : float
        Percentile for automatic threshold (if threshold=None)
        
    Returns:
    --------
    tuple : (predictions, threshold_used)
    """
    if threshold is None:
        threshold = np.percentile(scores, percentile)
    
    predictions = (scores > threshold).astype(int)
    return predictions, threshold
