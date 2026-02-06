import numpy as np
from sklearn.metrics import (
    precision_recall_curve,
    auc,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    accuracy_score,
    mean_squared_error
)
from src.utils.logger import get_logger

logger = get_logger("metrics")

def get_classification_metrics(y_true, y_pred, y_prob=None):
    """
    Compute key classification metrics.
    
    Args:
        y_true (array): True labels (0 or 1)
        y_pred (array): Predicted labels (0 or 1)
        y_prob (array, optional): Predicted probabilities for class 1.
        
    Returns:
        dict: Dictionary containing metrics.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "fpr": fpr,
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }

    if y_prob is not None:
        # ROC-AUC
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
        except ValueError:
            metrics["roc_auc"] = 0.5 # Handle single class edge case

        # PR-AUC
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        metrics["pr_auc"] = auc(recall, precision)
    else:
        metrics["roc_auc"] = None
        metrics["pr_auc"] = None

    metrics["confusion_matrix"] = {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
    return metrics

def get_reconstruction_metrics(y_true, y_pred):
    """Metrics for reconstruction-based models (Autoencoders)."""
    return {
        "mse": mean_squared_error(y_true, y_pred)
    }

def print_metrics(metrics):
    """Helper to print metrics nicely."""
    logger.info("--- Evaluation Metrics ---")
    logger.info(f"PR-AUC:    {metrics.get('pr_auc', 'N/A')}")
    logger.info(f"ROC-AUC:   {metrics.get('roc_auc', 'N/A')}")
    logger.info(f"Recall:    {metrics['recall']:.4f}")
    logger.info(f"Precision: {metrics['precision']:.4f}")
    logger.info(f"F1-Score:  {metrics['f1']:.4f}")
    logger.info(f"Conf Mat:  {metrics['confusion_matrix']}")
    logger.info("--------------------------")
