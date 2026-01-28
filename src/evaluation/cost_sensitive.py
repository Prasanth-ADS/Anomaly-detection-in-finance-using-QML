from sklearn.metrics import confusion_matrix
from src.utils.logger import get_logger

logger = get_logger("cost_sensitive")

def calculate_financial_loss(y_true, y_pred, cost_fn=100, cost_fp=1):
    """
    Calculates Expected financial loss.
    C_FN: Cost of False Negative (Fraud missed) - usually high.
    C_FP: Cost of False Positive (False Alarm) - usually low (admin cost).
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    total_loss = (fn * cost_fn) + (fp * cost_fp)
    
    logger.info(f"Loss Calc: FN={fn} * {cost_fn} + FP={fp} * {cost_fp} = {total_loss}")
    
    return total_loss

def compare_models_loss(results_dict, cost_fn=100, cost_fp=1):
    """
    Compares models based on financial loss.
    results_dict: {'model_name': (y_true, y_pred)}
    """
    losses = {}
    for model_name, (y_true, y_pred) in results_dict.items():
        losses[model_name] = calculate_financial_loss(y_true, y_pred, cost_fn, cost_fp)
    return losses
