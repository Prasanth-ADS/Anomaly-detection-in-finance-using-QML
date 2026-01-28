import pennylane as qml
from pennylane import numpy as np
from sklearn.svm import SVC
from src.utils.logger import get_logger

logger = get_logger("quantum_kernel")

class QuantumKernelModel:
    def __init__(self, n_qubits=4, backend="default.qubit"):
        self.n_qubits = n_qubits
        self.dev = qml.device(backend, wires=n_qubits)
        self.svm = SVC(kernel="precomputed") # We will compute the kernel matrix ourselves
        
    def feature_map(self, x):
        """Angle Encoding feature map."""
        qml.AngleEmbedding(x, wires=range(self.n_qubits))

    def kernel_circuit(self, x1, x2):
        """Quantum Kernel Circuit: <psi(x1)|psi(x2)> = |<0|U(x1)^dagger U(x2)|0>|^2"""
        # In PennyLane, we can just do the overlap test or simpler approach:
        # 1. State prep x1
        # 2. Adjoint State prep x2
        # 3. Measure all 0s
        
        # Note: PennyLane has specific kernel modules, keeping it explicit here for clarity or using qml.kernels
        pass 
    
    def compute_kernel_matrix(self, X_A, X_B):
        """
        Computes the kernel matrix using PennyLane's built-in kernel functionality
        to avoid manual loop inefficiency.
        """
        # Create a QNode for the kernel
        @qml.qnode(self.dev)
        def kernel(x1, x2):
            self.feature_map(x1)
            qml.adjoint(self.feature_map)(x2)
            return qml.probs(wires=range(self.n_qubits))

        # We take the probability of the all-zero state
        def kernel_func(x1, x2):
            return kernel(x1, x2)[0]

        logger.info(f"Computing Quantum Kernel Matrix: {X_A.shape[0]}x{X_B.shape[0]}")
        
        # Using qml.kernels.kernel_matrix (efficient if available) or simple loops
        # Check PennyLane version compatibility.
        # For simplicity in this scaffold:
        try:
            return qml.kernels.kernel_matrix(X_A, X_B, kernel_func)
        except:
            # Fallback simple loop (SLOW)
            K = np.zeros((X_A.shape[0], X_B.shape[0]))
            for i, x_a in enumerate(X_A):
                for j, x_b in enumerate(X_B):
                    K[i, j] = kernel_func(x_a, x_b)
            return K

    def fit(self, X_train, y_train):
        # Warning: Quantum Kernels are O(N^2), feasible only for small N (e.g., subset)
        if len(X_train) > 1000:
            logger.warning("Training set too large for Quantum Kernel. Subsampling to 500.")
            indices = np.random.choice(len(X_train), 500, replace=False)
            X_train = X_train[indices]
            y_train = y_train[indices]

        # Ensure features match n_qubits (PCA required beforehand)
        if X_train.shape[1] > self.n_qubits:
            logger.warning(f"Input features {X_train.shape[1]} > n_qubits {self.n_qubits}. Truncating.")
            X_train = X_train[:, :self.n_qubits]
            
        self.X_train = X_train
        kernel_matrix_train = self.compute_kernel_matrix(X_train, X_train)
        
        logger.info("Fitting SVC with Quantum Kernel...")
        self.svm.fit(kernel_matrix_train, y_train)

    def predict(self, X_test):
        if X_test.shape[1] > self.n_qubits:
            X_test = X_test[:, :self.n_qubits]
            
        kernel_matrix_test = self.compute_kernel_matrix(X_test, self.X_train)
        return self.svm.predict(kernel_matrix_test)
