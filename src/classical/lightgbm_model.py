import lightgbm as lgb
from src.utils.logger import get_logger

logger = get_logger("lightgbm")

class LightGBMModel:
    def __init__(self, random_state=42, is_unbalance=True):
        self.model = lgb.LGBMClassifier(
            random_state=random_state,
            is_unbalance=is_unbalance, # Handle imbalance
            n_jobs=-1,
            verbosity=-1
        )

    def fit(self, X, y, X_val=None, y_val=None):
        logger.info(f"Training LightGBM on shape {X.shape}")
        callbacks = []
        # valid_sets logic if desired, though sklearn API handles fit internal
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]
