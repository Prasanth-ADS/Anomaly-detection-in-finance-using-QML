from sklearn.svm import OneClassSVM
from src.utils.logger import get_logger

logger = get_logger("one_class_svm")

class OneClassSVMModel:
    def __init__(self, kernel='rbf', nu=0.1, gamma='scale'):
        self.model = OneClassSVM(
            kernel=kernel,
            nu=nu,
            gamma=gamma,
            verbose=False
        )

    def fit(self, X):
        logger.info(f"Training One-Class SVM on shape {X.shape}")
        self.model.fit(X)

    def predict(self, X):
        y_pred = self.model.predict(X)
        # Remap: -1 (anomaly) -> 1, 1 (normal) -> 0
        return (y_pred == -1).astype(int)
    
    def score(self, X):
        return self.model.decision_function(X)
