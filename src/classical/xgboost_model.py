import xgboost as xgb
from src.utils.logger import get_logger

logger = get_logger("xgboost")

class XGBoostModel:
    def __init__(self, scale_pos_weight=99, random_state=42):
        """
        Supervised model.
        scale_pos_weight helps with imbalance.
        """
        self.model = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='aucpr',
            scale_pos_weight=scale_pos_weight,
            seed=random_state,
            n_jobs=-1
        )

    def fit(self, X, y, X_val=None, y_val=None):
        logger.info(f"Training XGBoost on shape {X.shape}")
        eval_set = [(X_val, y_val)] if X_val is not None else None
        self.model.fit(X, y, eval_set=eval_set, verbose=False)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]
