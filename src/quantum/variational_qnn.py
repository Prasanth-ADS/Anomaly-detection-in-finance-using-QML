import pennylane as qml
from pennylane import numpy as np
import torch
import torch.nn as nn
from src.utils.logger import get_logger

logger = get_logger("vqnn")

class VQNNModel(nn.Module):
    def __init__(self, n_qubits=4, n_layers=2, backend="default.qubit"):
        super(VQNNModel, self).__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.dev = qml.device(backend, wires=n_qubits)
        
        # Define the QNode
        @qml.qnode(self.dev, interface='torch')
        def circuit(inputs, weights):
            # Data Encoding (Angle)
            qml.AngleEmbedding(inputs, wires=range(n_qubits))
            # Variational Layers
            qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
            # Measurement
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
            
        weight_shapes = {"weights": (n_layers, n_qubits, 3)}
        self.qnn = qml.qnn.TorchLayer(circuit, weight_shapes)
        
        # Classical post-processing -> Binary Classification
        self.fc = nn.Linear(n_qubits, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.qnn(x)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x

class VQNNTrainer:
    def __init__(self, n_qubits=4, n_layers=2, lr=0.01, epochs=20, batch_size=32):
        self.model = VQNNModel(n_qubits, n_layers)
        self.epochs = epochs
        self.batch_size = batch_size
        self.criterion = nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.n_qubits = n_qubits

    def fit(self, X, y):
        # Convert to headers
        if X.shape[1] != self.n_qubits:
             logger.warning(f"Feature dim {X.shape[1]} != Quibits {self.n_qubits}. Truncating/Padding needed.")
             # Simple truncation for demo
             X = X[:, :self.n_qubits]

        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y).reshape(-1, 1)
        
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        logger.info("Training VQNN...")
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for batch_x, batch_y in loader:
                self.optimizer.zero_grad()
                output = self.model(batch_x)
                loss = self.criterion(output, batch_y)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            if (epoch + 1) % 5 == 0:
                logger.info(f"Epoch {epoch+1}, Loss: {total_loss/len(loader):.4f}")

    def predict(self, X):
        if X.shape[1] != self.n_qubits:
             X = X[:, :self.n_qubits]
        X_tensor = torch.FloatTensor(X)
        self.model.eval()
        with torch.no_grad():
            preds = self.model(X_tensor)
        return (preds > 0.5).float().numpy().flatten()
