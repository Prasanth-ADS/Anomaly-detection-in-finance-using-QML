from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from src.utils.logger import get_logger

logger = get_logger("benchmarks")

class ClassicalBenchmarks:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self._initialize_models()

    def _initialize_models(self):
        # 1. Logistic Regression (Baseline)
        self.models['LogisticRegression'] = LogisticRegression(
            class_weight='balanced', 
            random_state=self.random_state,
            max_iter=1000
        )

        # 2. Linear SVM (Direct comparison to Q-SVM)
        # Probability=True needed for AUC
        self.models['SVM (Linear)'] = SVC(
            kernel='linear', 
            class_weight='balanced', 
            probability=True, 
            random_state=self.random_state
        )
        
        # 3. RBF SVM (Kernel comparison)
        self.models['SVM (RBF)'] = SVC(
            kernel='rbf', 
            class_weight='balanced', 
            probability=True, 
            random_state=self.random_state
        )

        # 4. Small MLP (Neural Net comparison)
        # Sized similarly to QNN (e.g., 4->8->4->1) to be "Fair" in param count?
        # Or just a standard small one.
        self.models['MLP (Small)'] = MLPClassifier(
            hidden_layer_sizes=(8, 4),
            max_iter=500,
            random_state=self.random_state
        )

    def train_and_evaluate(self, X_train, y_train, X_test):
        """
        Train all models and return predictions.
        Returns:
            dict: {model_name: {'pred': y_pred, 'prob': y_prob}}
        """
        results = {}
        for name, model in self.models.items():
            logger.info(f"Training {name}...")
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_test)[:, 1]
            else:
                y_prob = model.decision_function(X_test) # Fallback for some SVMs if prob=False
            
            results[name] = {
                'pred': y_pred,
                'prob': y_prob
            }
        return results
