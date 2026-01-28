import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from src.utils.logger import get_logger

logger = get_logger("autoencoder")

class AutoEncoder(nn.Module):
    def __init__(self, input_dim, encoding_dim=8):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, encoding_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim),
            # Output range depends on input normalization. 
            # If [0, 1], Sigmoid is good.
            nn.Sigmoid() 
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class AutoEncoderModel:
    def __init__(self, input_dim, encoding_dim=8, lr=0.001, epochs=50, batch_size=64):
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoEncoder(input_dim, encoding_dim).to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def fit(self, X, y=None):
        # We only train on 'Normal' data (Class 0) for anomaly detection usually,
        # OR we train on all and expect higher reconstruction error for anomalies.
        # Assuming X is just features.
        
        # Ensure tensor
        X_tensor = torch.FloatTensor(X).to(self.device)
        dataset = TensorDataset(X_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        logger.info(f"Training Autoencoder on {X.shape} using {self.device}")
        
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for batch in loader:
                inputs = batch[0]
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, inputs)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch [{epoch+1}/{self.epochs}], Loss: {total_loss/len(loader):.4f}")

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
            # Default to some percentile if not provided?
            # Or return scores.
            logger.warning("No threshold provided for Autoencoder prediction. Returning scores.")
            return scores
        return (scores > threshold).astype(int)
