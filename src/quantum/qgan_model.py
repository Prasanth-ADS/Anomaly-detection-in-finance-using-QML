import pennylane as qml
from pennylane import numpy as np
import time
from src.utils.logger import get_logger

logger = get_logger("qgan")

class QGANModel:
    def __init__(self, n_qubits=2, n_layers=2, backend="default.qubit"):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.dev = qml.device(backend, wires=n_qubits)
        
        # Generator params
        self.gen_params = np.random.uniform(low=-np.pi, high=np.pi, size=(n_layers, n_qubits, 3), requires_grad=True)
        # Discriminator (Classical for simplicity in hybrid setting)
        from sklearn.linear_model import LogisticRegression
        self.disc = LogisticRegression()

        @qml.qnode(self.dev)
        def generator_circuit(params):
            # Input is usually noise, but QGAN often uses variational state as distribution
            for layer in range(self.n_layers):
                for i in range(self.n_qubits):
                    qml.RX(params[layer, i, 0], wires=i)
                    qml.RY(params[layer, i, 1], wires=i)
                    qml.RZ(params[layer, i, 2], wires=i)
                for i in range(self.n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
            return qml.probs(wires=range(self.n_qubits))
            
        self.generator_circuit = generator_circuit

    def train(self, real_data, steps=50, lr=0.01):
        """
        Train QGAN on 1D/2D toy distributions.
        real_data: expected to be probabilities of states in 2^n_qubits
        """
        opt = qml.AdamOptimizer(stepsize=lr)
        
        def generator_loss(params):
            # Adversarial loss: minimize distance to real distribution
            fake_data = self.generator_circuit(params)
            return np.sum((fake_data - real_data)**2)

        start_time = time.time()
        for i in range(steps):
            self.gen_params, cost = opt.step_and_cost(generator_loss, self.gen_params)
            if (i+1) % 10 == 0:
                logger.info(f"QGAN Step {i+1}/{steps} | Loss: {cost:.4f}")
        
        self.train_time = time.time() - start_time
        logger.info(f"QGAN Training complete in {self.train_time:.2f}s")

    def generate(self):
        return self.generator_circuit(self.gen_params)
