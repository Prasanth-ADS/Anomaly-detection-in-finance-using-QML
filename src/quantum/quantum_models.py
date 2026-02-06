import torch
import torch.nn as nn

# 1. Variational Quantum Classifier (VQC) using TorchLayer for stability
class VQC_Net(nn.Module):
    def __init__(self, n_qubits=4, n_layers=3):
        super(VQC_Net, self).__init__()
        self.dev = qml.device("default.qubit", wires=n_qubits)
        
        @qml.qnode(self.dev, interface='torch')
        def circuit(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation='Y')
            qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
            return qml.expval(qml.PauliZ(0))
            
        weight_shapes = {"weights": (n_layers, n_qubits, 3)}
        self.vqc = qml.qnn.TorchLayer(circuit, weight_shapes)

    def forward(self, x):
        # Result shape (batch,)
        out = self.vqc(x)
        return (out + 1) / 2 # Map [-1, 1] to [0, 1]

class VQCModel:
    def __init__(self, n_qubits=10, n_layers=3):
        self.model = VQC_Net(n_qubits, n_layers)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.1)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def fit(self, X, y, steps=30, batch_size=32):
        X_t = torch.FloatTensor(X).to(self.device)
        y_t = torch.FloatTensor(y).reshape(-1, 1).to(self.device)
        dataset = torch.utils.data.TensorDataset(X_t, y_t)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        start_time = time.time()
        self.model.train()
        for i in range(steps):
            for bx, by in loader:
                self.optimizer.zero_grad()
                out = self.model(bx).reshape(-1, 1)
                loss = self.criterion(out, by)
                loss.backward()
                self.optimizer.step()
        self.train_time = time.time() - start_time

    def predict_proba(self, X):
        self.model.eval()
        with torch.no_grad():
            X_t = torch.FloatTensor(X).to(self.device)
            out = self.model(X_t)
        return out.cpu().numpy().flatten()

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) > threshold).astype(int)

# 2. Hybrid Quantum Neural Network (QNN)
class HybridQNN(nn.Module):
    def __init__(self, n_qubits=10, n_layers=2, backend="default.qubit"):
        super(HybridQNN, self).__init__()
        self.dev = qml.device(backend, wires=n_qubits)
        
        @qml.qnode(self.dev, interface='torch')
        def q_circuit(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation='Y')
            qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
        
        weight_shapes = {"weights": (n_layers, n_qubits, 3)}
        self.q_layer = qml.qnn.TorchLayer(q_circuit, weight_shapes)
        self.post_net = nn.Sequential(
            nn.Linear(n_qubits, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        q_out = self.q_layer(x)
        return self.post_net(q_out)

class QNNModel:
    def __init__(self, n_qubits=10, n_layers=2):
        self.model = HybridQNN(n_qubits, n_layers)
        self.criterion = nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def fit(self, X, y, epochs=20, batch_size=32):
        X_t = torch.FloatTensor(X).to(self.device)
        y_t = torch.FloatTensor(y).reshape(-1, 1).to(self.device)
        dataset = torch.utils.data.TensorDataset(X_t, y_t)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        start_time = time.time()
        self.model.train()
        for epoch in range(epochs):
            for bx, by in loader:
                self.optimizer.zero_grad()
                out = self.model(bx)
                loss = self.criterion(out, by)
                loss.backward()
                self.optimizer.step()
        self.train_time = time.time() - start_time

    def predict_proba(self, X):
        self.model.eval()
        with torch.no_grad():
            X_t = torch.FloatTensor(X).to(self.device)
            out = self.model(X_t)
        return out.cpu().numpy().flatten()

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) > threshold).astype(int)

# 3. Quantum Kernel SVM (QSVM)
from sklearn.svm import SVC

class QSVMModel:
    def __init__(self, n_qubits=4, backend="default.qubit"):
        self.n_qubits = n_qubits
        self.dev = qml.device(backend, wires=n_qubits)
        
        @qml.qnode(self.dev)
        def kernel_circuit(x1, x2):
            qml.AngleEmbedding(x1, wires=range(self.n_qubits), rotation='Z')
            qml.adjoint(qml.AngleEmbedding)(x2, wires=range(self.n_qubits), rotation='Z')
            return qml.probs(wires=range(self.n_qubits))
            
        self.kernel_circuit = kernel_circuit
        self.svc = SVC(kernel='precomputed', probability=True)

    def q_kernel(self, X1, X2):
        # Explicitly compute kernel matrix
        K = np.zeros((len(X1), len(X2)))
        for i, x1 in enumerate(X1):
            for j, x2 in enumerate(X2):
                K[i, j] = self.kernel_circuit(x1, x2)[0] # Prob of |0...0>
        return K

    def fit(self, X, y):
        start_time = time.time()
        K_train = self.q_kernel(X, X)
        self.X_train = X
        self.svc.fit(K_train, y)
        self.train_time = time.time() - start_time

    def predict(self, X):
        K_test = self.q_kernel(X, self.X_train)
        return self.svc.predict(K_test)

    def predict_proba(self, X):
        K_test = self.q_kernel(X, self.X_train)
        return self.svc.predict_proba(K_test)[:, 1]

# 4. Quantum Autoencoder
class QAEModel:
    def __init__(self, n_qubits=6, n_latent=2, n_layers=2, backend="default.qubit"):
        self.n_qubits = n_qubits
        self.n_latent = n_latent
        self.n_layers = n_layers
        self.n_trash = n_qubits - n_latent
        self.dev = qml.device(backend, wires=n_qubits)
        self.params = None
        
        @qml.qnode(self.dev)
        def circuit(params, x):
            # Input features into full register
            qml.AngleEmbedding(x, wires=range(self.n_qubits), rotation='Y')
            
            # Variational compression circuit
            for layer in range(self.n_layers):
                for i in range(self.n_qubits):
                    qml.RY(params[layer, i], wires=i)
                for i in range(self.n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
            
            # Swap test or measuring trash qubits
            # Here we want trash qubits to be in |0> state for compression
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_trash)]

        self.circuit = circuit

    def cost(self, params, X):
        # We want trash qubits to be |0>, so expval(Z) should be 1
        res = np.array([self.circuit(params, x) for x in X])
        return np.mean(1 - res)

    def fit(self, X, y=None, steps=30):
        # Only train on normal data
        if y is not None:
            X = X[y == 0]
        
        self.params = np.random.uniform(low=0, high=2*np.pi, size=(self.n_layers, self.n_qubits), requires_grad=True)
        opt = qml.AdamOptimizer(stepsize=0.1)
        
        start_time = time.time()
        for i in range(steps):
            self.params, c = opt.step_and_cost(lambda p: self.cost(p, X), self.params)
        self.train_time = time.time() - start_time

    def predict_anomaly_score(self, X):
        # Higher score = less overlap with |0> on trash = more anomalous
        res = np.array([self.circuit(self.params, x) for x in X])
        scores = np.mean(1 - res, axis=1)
        return scores

    def predict(self, X, threshold=None):
        scores = self.predict_anomaly_score(X)
        if threshold is None:
            threshold = np.percentile(scores, 95)
        return (scores > threshold).astype(int)
