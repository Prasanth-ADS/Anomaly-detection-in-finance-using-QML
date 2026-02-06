from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from src.utils.logger import get_logger

logger = get_logger("baseline_models")

# 1. Logistic Regression
class LogisticRegressionModel:
    def __init__(self, random_state=42, max_iter=1000, class_weight='balanced'):
        self.model = LogisticRegression(
            random_state=random_state,
            max_iter=max_iter,
            class_weight=class_weight,
            n_jobs=-1
        )

    def fit(self, X, y):
        logger.info(f"Training Logistic Regression on shape {X.shape}")
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]

# 2. SVM (RBF)
class SVMModel:
    def __init__(self, kernel='rbf', random_state=42, class_weight='balanced'):
        self.model = SVC(
            kernel=kernel,
            random_state=random_state,
            class_weight=class_weight,
            probability=True
        )

    def fit(self, X, y):
        logger.info(f"Training SVM ({self.model.kernel}) on shape {X.shape}")
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]

# 3. Random Forest
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

# 4. Classical MLP
class MLPModel:
    def __init__(self, hidden_layer_sizes=(64, 32), random_state=42, max_iter=500):
        self.model = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            random_state=random_state,
            max_iter=max_iter
        )

    def fit(self, X, y):
        logger.info(f"Training MLP on shape {X.shape}")
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]

# 5. Isolation Forest
class IsolationForestModel:
    def __init__(self, contamination=0.05, random_state=42):
        self.model = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1
        )

    def fit(self, X, y=None):
        logger.info(f"Training Isolation Forest on shape {X.shape}")
        self.model.fit(X)

    def predict(self, X):
        # Isolation Forest returns -1 for anomalies and 1 for normal
        preds = self.model.predict(X)
        return (preds == -1).astype(int)

    def decision_function(self, X):
        return -self.model.decision_function(X) # Higher score = more anomalous

# 6. Classical Autoencoder
class AE_Network(nn.Module):
    def __init__(self, input_dim, encoding_dim=4):
        super(AE_Network, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, encoding_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 16),
            nn.ReLU(),
            nn.Linear(16, input_dim),
            nn.Sigmoid() 
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class AutoEncoderModel:
    def __init__(self, input_dim, encoding_dim=4, lr=0.001, epochs=50, batch_size=32):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AE_Network(input_dim, encoding_dim).to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.epochs = epochs
        self.batch_size = batch_size

    def fit(self, X, y=None):
        # For pure anomaly detection, we only train on normal data
        if y is not None:
            X_normal = X[y == 0]
        else:
            X_normal = X
            
        X_tensor = torch.FloatTensor(X_normal).to(self.device)
        dataset = TensorDataset(X_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        self.model.train()
        for epoch in range(self.epochs):
            for batch in loader:
                inputs = batch[0]
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, inputs)
                loss.backward()
                self.optimizer.step()

    def predict_anomaly_score(self, X):
        X_tensor = torch.FloatTensor(X).to(self.device)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
            mse = torch.mean((outputs - X_tensor) ** 2, dim=1)
        return mse.cpu().numpy()

    def predict(self, X, threshold=None):
        scores = self.predict_anomaly_score(X)
        if threshold is None:
            threshold = np.percentile(scores, 95)
        return (scores > threshold).astype(int)
