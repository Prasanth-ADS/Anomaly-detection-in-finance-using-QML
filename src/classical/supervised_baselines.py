from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from src.utils.logger import get_logger

logger = get_logger("supervised_baselines")

class LogisticRegressionModel:
    def __init__(self, random_state=42, max_iter=1000, class_weight='balanced'):
        self.model = LogisticRegression(
            random_state=random_state,
            max_iter=max_iter,
            class_weight=class_weight, # Important for imbalance
            n_jobs=-1
        )

    def fit(self, X, y):
        logger.info(f"Training Logistic Regression on shape {X.shape}")
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]

class RandomForestModel:
    def __init__(self, n_estimators=100, random_state=42, class_weight='balanced'):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            class_weight=class_weight,
            n_jobs=-1
        )

    def fit(self, X, y):
        logger.info(f"Training Random Forest on shape {X.shape}")
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]
