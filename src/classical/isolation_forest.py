from sklearn.ensemble import IsolationForest
from src.utils.logger import get_logger

logger = get_logger("isolation_forest")

class IsolationForestModel:
    def __init__(self, n_estimators=100, contamination='auto', random_state=42):
        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1
        )

    def fit(self, X):
        logger.info(f"Training Isolation Forest on shape {X.shape}")
        self.model.fit(X)

    def predict(self, X):
        """
        Returns -1 for anomaly, 1 for normal.
        We typically map this to 1 for anomaly, 0 for normal for metrics.
        """
        y_pred = self.model.predict(X)
        # Remap: -1 (anomaly) -> 1, 1 (normal) -> 0
        return (y_pred == -1).astype(int)

    def score(self, X):
        """Higher scores = more normal. Lower scores = more anomalous."""
        return self.model.decision_function(X)
