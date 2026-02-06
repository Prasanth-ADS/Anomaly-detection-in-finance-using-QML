# Quantum v.s. Classical Anomaly Detection in Finance

This repository contains a reproducible research pipeline comparing classical machine learning methods with Quantum Neural Network (QNN)-based anomaly detection for financial fraud detection.

## Structure

- `data/`: Raw and processed datasets.
- `notebooks/`: Jupyter notebooks for exploration and experiments.
- `src/`: Source code for the pipeline.
  - `classical/`: Classical ML models (Isolation Forest, SVM, XGBoost, Autoencoder).
  - `quantum/`: Quantum models (QSVM, VQNN).
  - `data/`: Data loading and preprocessing.
  - `features/`: Feature engineering.
  - `evaluation/`: Metrics and statistical tests.
- `results/`: output figures, tables, and logs.

## Setup

1. **Clone the repository**
2. **Install dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

### 1. Data Setup

Ensure `creditcard.csv` and `paysim.csv` are in `data/raw/`.
If using the default config, `creditcard.csv` is required.

### 2. Run the Full Pipeline

To run the end-to-end pipeline (Data Load -> Split -> Train Classical -> Train Quantum -> Eval):

```bash
python main.py --config src/config/config.yaml
```

### 3. Run Improved QML Benchmark

To run the detailed comparison between the Optimized QML model (Data Re-uploading) and Classical Baselines:

```bash
python benchmark_qml.py
```

### 4. Notebooks

Explore the notebooks in `notebooks/` for step-by-step analysis:

- `01_data_exploration.ipynb`: Data viz and distribution checks.
- `02_feature_engineering.ipynb`: PCA and feature selection.
- `03_classical_models.ipynb`: Training Isolation Forest, XGBoost, etc.
- `04_quantum_kernel.ipynb`: QSVM experiments.
- `05_variational_qnn.ipynb`: VQNN experiments.
- `06_results_analysis.ipynb`: Comparison plots and statistical tests.

### 5. Reproducibility

Run the verification script to ensure random seeds are working:

```bash
python verify_reproducibility.py
```

## Configuration

Edit `src/config/config.yaml` to change:

- `n_qubits`: Number of qubits for QNN (matched with PCA components).
- `n_layers`: Depth of VQNN.
- `backend`: Quantum backend (e.g., `default.qubit`, `qiskit.aer`).
