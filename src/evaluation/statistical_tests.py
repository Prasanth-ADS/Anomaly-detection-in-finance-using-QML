from scipy import stats
import numpy as np
from src.utils.logger import get_logger

logger = get_logger("stat_tests")

def perform_ttest(scores_a, scores_b):
    """
    Paired t-test to check if model A is significantly different from model B.
    scores_a, scores_b: Lists/Arrays of metric scores (e.g., F1s across folds).
    """
    t_stat, p_val = stats.ttest_rel(scores_a, scores_b)
    logger.info(f"T-test: t={t_stat:.4f}, p={p_val:.4f}")
    return t_stat, p_val

def perform_wilcoxon(scores_a, scores_b):
    """
    Wilcoxon signed-rank test (non-parametric version of paired t-test).
    """
    try:
        w_stat, p_val = stats.wilcoxon(scores_a, scores_b)
        logger.info(f"Wilcoxon: w={w_stat:.4f}, p={p_val:.4f}")
        return w_stat, p_val
    except Exception as e:
        logger.warning(f"Wilcoxon failed (maybe identical scores?): {e}")
        return 0, 1.0

def bootstrap_confidence_interval(data, confidence=0.95, n_bootstraps=1000):
    """
    Calculates Bootstrap CI for a metric.
    """
    data = np.array(data)
    boot_means = []
    for _ in range(n_bootstraps):
        sample = np.random.choice(data, size=len(data), replace=True)
        boot_means.append(np.mean(sample))
    
    lower = np.percentile(boot_means, (1 - confidence) / 2 * 100)
    upper = np.percentile(boot_means, (1 + confidence) / 2 * 100)
    
    return lower, upper
