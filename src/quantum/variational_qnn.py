import pennylane as qml
from pennylane import numpy as np
import torch
import torch.nn as nn
from src.utils.logger import get_logger

logger = get_logger("vqnn")

class VQNNModel(nn.Module):
    def __init__(self, n_qubits=4, n_layers=2, backend="default.qubit", use_reuploading=True):
        """
        Variational Quantum Neural Network with Data Re-uploading support.
        
        Args:
            n_qubits: Number of qubits (components after PCA).
            n_layers: Number of (Encoding + Variational) layers for Re-uploading,
                      or just Variational layers for standard QNN.
            backend: Quantum backend name.
            use_reuploading: If True, inputs are re-encoded in every layer. 
                             If False, encoded once at start.
        """
        super(VQNNModel, self).__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.use_reuploading = use_reuploading
        self.dev = qml.device(backend, wires=n_qubits)
        
        # Define the QNode
        @qml.qnode(self.dev, interface='torch')
        def circuit(inputs, weights):
            """
            Args:
                inputs: (batch_size, n_qubits)
                weights: (n_layers, n_qubits, 3) or similar depending on Ansatz
            """
            # Reshape inputs if necessary (handled by PennyLane broadcasting usually, 
            # but TorchLayer expects specific signatures. 
            # Actually, TorchLayer automatically handles batching if designed right.
            
            # 1. Initial Encoding (Always needed)
            qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation='Y')
            
            # 2. Layers
            for i in range(n_layers):
                if use_reuploading and i > 0:
                    # Re-upload data (Mix features into state again)
                     qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation='Y')
                
                # Variational Block
                qml.StronglyEntanglingLayers(weights[i:i+1], wires=range(n_qubits))

            # Measurement (Expectation of PauliZ on all qubits or just one)
            # Measuring all provides more info -> post-process with Classical Linear
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
            
        weight_shapes = {"weights": (n_layers, n_qubits, 3)}
        self.qnn = qml.qnn.TorchLayer(circuit, weight_shapes)
        
        # Classical post-processing -> Binary Classification
        # Input: n_qubits expectations -> Output: 1 scalar (logit)
        self.fc = nn.Linear(n_qubits, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: (batch, n_qubits)
        q_out = self.qnn(x) # (batch, n_qubits)
        out = self.fc(q_out)
        out = self.sigmoid(out)
        return out

class VQNNTrainer:
    def __init__(self, model, lr=0.01, epochs=20, criterion=None):
        """
        Args:
           model: The VQNNModel instance.
        """
        self.model = model
        self.epochs = epochs
        # Weighted BCELoss is better for Imbalance, user can pass it.
        self.criterion = criterion if criterion else nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        # Data preparation
        X_tensor = torch.FloatTensor(X_train).to(self.device)
        y_tensor = torch.FloatTensor(y_train).reshape(-1, 1).to(self.device)
        
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        # Handle batch size inside ?? Or pass in __init__. 
        # Making it standard 32 for now as per prompt constraints involved simple laptop
        batch_size = 32 
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        logger.info(f"Starting QNN Training: {self.model.n_layers} Layers, Re-upload={self.model.use_reuploading}")
        self.model.train()
        
        history = {'loss': [], 'val_loss': []}
        
        for epoch in range(self.epochs):
            total_loss = 0
            for batch_x, batch_y in loader:
                self.optimizer.zero_grad()
                output = self.model(batch_x)
                loss = self.criterion(output, batch_y)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(loader)
            history['loss'].append(avg_loss)
            
            # Validation
            val_str = ""
            if X_val is not None and y_val is not None:
                val_loss = self.evaluate_loss(X_val, y_val)
                history['val_loss'].append(val_loss)
                val_str = f"| Val Loss: {val_loss:.4f}"
            
            if (epoch + 1) % 5 == 0 or epoch == 0:
                logger.info(f"Epoch {epoch+1}/{self.epochs} | Loss: {avg_loss:.4f} {val_str}")
                
        return history

    def evaluate_loss(self, X, y):
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).reshape(-1, 1).to(self.device)
        with torch.no_grad():
            preds = self.model(X_tensor)
            loss = self.criterion(preds, y_tensor)
        self.model.train()
        return loss.item()

    def predict_proba(self, X):
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        with torch.no_grad():
            preds = self.model(X_tensor)
        return preds.cpu().numpy().flatten()

    def predict(self, X, threshold=0.5):
        probs = self.predict_proba(X)
        return (probs > threshold).astype(float)
