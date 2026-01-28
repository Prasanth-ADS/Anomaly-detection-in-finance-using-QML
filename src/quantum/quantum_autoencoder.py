import pennylane as qml
from pennylane import numpy as np
import torch
import torch.nn as nn
from src.utils.logger import get_logger

logger = get_logger("quantum_autoencoder")

class QuantumAutoencoder(nn.Module):
    """
    A simple Quantum Autoencoder.
    We try to compress N qubits state into K latent qubits.
    """
    def __init__(self, n_qubits=4, n_latent=2, n_layers=2, backend="default.qubit"):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_latent = n_latent
        self.trash_qubits = list(range(n_latent, n_qubits)) # Qubits to measure/discard
        self.dev = qml.device(backend, wires=n_qubits)
        
        @qml.qnode(self.dev, interface='torch')
        def circuit(inputs, weights):
            # Encoder
            qml.AngleEmbedding(inputs, wires=range(n_qubits))
            qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
            
            # We want the trash qubits to be in state |0>
            # We measure the probability of trash qubits being |0>
            # Maximizing this prob = minimizing loss
            return qml.probs(wires=self.trash_qubits)
            
        weight_shapes = {"weights": (n_layers, n_qubits)}
        self.qnn = qml.qnn.TorchLayer(circuit, weight_shapes)

    def forward(self, x):
        # Output is probs of trash qubits. 
        # We want probability of '0...0' state on trash qubits to be 1.
        # Index 0 of the probs usually corresponds to the all-zero state for the measured wires.
        return self.qnn(x)

class QuantumAutoencoderTrainer:
    def __init__(self, n_qubits=4, n_latent=2, lr=0.01, epochs=20):
        self.model = QuantumAutoencoder(n_qubits, n_latent)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.epochs = epochs
        # Loss: We want to MAXIMIZE prob of 0 state. So MINIMIZE (1 - prob[0])
        # BUT `qml.probs` returns probabilities of all basis states of measured wires.
        # If trash qubits = 2, we have 4 states: 00, 01, 10, 11.
        # We want state 00 (index 0) to be 1.0. 
        
    def fit(self, X):
        X_tensor = torch.FloatTensor(X) # Ensure scaled to [0, pi] roughly
        dataset = torch.utils.data.TensorDataset(X_tensor)
        loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)
        
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for batch in loader:
                inputs = batch[0]
                self.optimizer.zero_grad()
                probs = self.model(inputs)
                
                # Probs shape: [batch, 2^len(trash_qubits)]
                # We want probs[:, 0] to be 1.
                prob_zero = probs[:, 0]
                loss = torch.mean(1 - prob_zero)
                
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            
            logger.info(f"QAE Epoch {epoch+1}, Loss: {total_loss/len(loader):.4f}")

    def score(self, X):
        """Returns anomaly score (reconstruction error proxy). 
        Here, 1 - probability of trash qubits being |0>.
        Higher = More Anomalous."""
        X_tensor = torch.FloatTensor(X)
        self.model.eval()
        with torch.no_grad():
            probs = self.model(X_tensor)
            prob_zero = probs[:, 0]
            scores = 1 - prob_zero
        return scores.numpy()
