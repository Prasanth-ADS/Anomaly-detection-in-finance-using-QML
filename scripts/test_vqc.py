import torch
import torch.nn as nn
import pennylane as qml
import numpy as np

def test_vqc():
    n_qubits = 4
    n_layers = 3
    dev = qml.device("default.qubit", wires=n_qubits)
    
    @qml.qnode(dev, interface='torch')
    def circuit(inputs, weights):
        # inputs shape: (batch_size, n_qubits)
        # weights shape: (n_layers, n_qubits, 3)
        qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation='Y')
        qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
        return qml.expval(qml.PauliZ(0))

    weight_shapes = {"weights": (n_layers, n_qubits, 3)}
    vqc_layer = qml.qnn.TorchLayer(circuit, weight_shapes)
    
    # Test with batch
    x = torch.randn(32, 4)
    try:
        out = vqc_layer(x)
        print(f"Success! Output shape: {out.shape}")
    except Exception as e:
        print(f"Failed with error: {e}")

if __name__ == "__main__":
    test_vqc()
