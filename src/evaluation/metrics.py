from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, average_precision_score, confusion_matrix
)
import pandas as pd
from src.utils.logger import get_logger

logger = get_logger("metrics")

def calculate_metrics(y_true, y_pred, y_probs=None):
    """
    Calculates standard classification metrics.
    y_pred: Binary predictions
    y_probs: Probability scores (optional, required for AUC)
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0)
    }
    
    if y_probs is not None:
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_probs)
            metrics['pr_auc'] = average_precision_score(y_true, y_probs)
        except Exception as e:
            logger.warning(f"Could not calculate AUC: {e}")
            metrics['roc_auc'] = 0.0
            metrics['pr_auc'] = 0.0

    return metrics

def print_metrics(metrics):
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
