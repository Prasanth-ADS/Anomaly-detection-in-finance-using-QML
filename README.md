# Anomaly Detection in Financial Data using Quantum Machine Learning

A comprehensive **research-grade benchmark pipeline** comparing Classical ML, Neural Networks, and Quantum Machine Learning (QML) approaches for anomaly detection in financial transactions.

## 🎯 Project Overview

This project implements a complete, reproducible benchmarking framework that:

- Compares **8 Classical ML** models, **5 Neural Network** models, and **5 QML** models
- Uses the **Credit Card Fraud Detection** dataset (~284K transactions)
- Provides fair comparison with consistent preprocessing and evaluation
- Includes detailed visualizations and statistical analysis

## 📁 Project Structure

```
.
├── config/                     # Configuration files
│   ├── config.py              # Paths, constants, hyperparameters
│   └── metrics.py             # Unified metric functions
├── data/
│   ├── raw/                   # Original creditcard.csv
│   ├── processed/             # Cleaned data
│   ├── splits/                # Train/test splits
│   └── features/              # PCA-reduced features
├── models/                    # Saved model weights
├── results/                   # Metrics CSVs and summaries
├── figures/                   # All visualizations
│   └── circuit_diagrams/      # Quantum circuit visualizations
├── notebooks/                 # 12 Jupyter notebooks
└── requirements.txt           # Python dependencies
```

## 🔬 Notebooks Workflow

### Phase 1: Data Pipeline

| # | Notebook | Description |
|---|----------|-------------|
| 1 | `01_data_ingestion.ipynb` | Load raw data, handle missing values, stratified subsample |
| 2 | `02_data_engineering.ipynb` | Feature scaling, train/test split |
| 3 | `03_feature_engineering.ipynb` | Statistical features, importance screening |
| 4 | `04_dimensionality_reduction.ipynb` | PCA to 10 components |

### Phase 2: Tuning & Training

| # | Notebook | Description |
|---|----------|-------------|
| 5 | `05_hyperparameter_tuning.ipynb` | GridSearchCV for all model categories |
| 6 | `06_classical_models.ipynb` | 8 classical models (SVM, RF, IF, etc.) |
| 7 | `07_neural_models.ipynb` | 5 PyTorch models (MLP, AE, VAE, LSTM, OCNN) |
| 8 | `08_qml_models.ipynb` | 5 PennyLane models (VQC, Hybrid QNN, QSVM, QAE, QGAN) |

### Phase 3: Comparative Analysis

| # | Notebook | Description |
|---|----------|-------------|
| 9 | `09_classical_vs_neural.ipynb` | Classical vs Neural comparison |
| 10 | `10_neural_vs_qml.ipynb` | Neural vs QML with scalability analysis |
| 11 | `11_classical_vs_qml.ipynb` | Classical vs QML efficiency trade-offs |
| 12 | `12_unified_comparison.ipynb` | Final dashboard with all results |

## 🛠️ Installation

```bash
# Clone repository
git clone <repo-url>
cd Anomaly-detection-in-finance-using-QML

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 📊 Models Implemented

### Classical ML (Notebook 6)

- SVM (Linear & RBF kernels)
- Logistic Regression
- Random Forest
- Isolation Forest
- Gaussian Mixture Model
- Classical Autoencoder
- Classical MLP

### Neural Networks (Notebook 7)

- Deep MLP Classifier
- Deep Autoencoder
- Variational Autoencoder (VAE)
- LSTM Autoencoder
- Deep One-Class Neural Network

### Quantum ML (Notebook 8)

- Variational Quantum Classifier (VQC)
- Hybrid Quantum Neural Network
- Quantum Kernel SVM (QSVM)
- Quantum Autoencoder
- Quantum GAN (toy implementation)

## 🔧 Configuration

Key settings in `config/config.py`:

- `RANDOM_SEED = 42` - Reproducibility
- `SUBSAMPLE_SIZE = 2000` - Dataset size for tractable QML
- `ANOMALY_RATIO ≈ 5%` - Target anomaly percentage
- `N_COMPONENTS = 10` - PCA dimensions
- `N_QUBITS = 4` - Quantum circuit width

## 📈 Running the Pipeline

```bash
# Run notebooks in order
jupyter notebook notebooks/01_data_ingestion.ipynb
# ... continue through notebook 12
```

Or execute all cells programmatically:

```bash
jupyter nbconvert --execute --to notebook notebooks/*.ipynb
```

## 📋 Requirements

- Python 3.9+
- PyTorch 2.0+
- PennyLane 0.30+
- scikit-learn 1.3+
- See `requirements.txt` for full list

## ⚠️ QML Limitations

This study uses **4 qubits** on a **simulator** due to hardware constraints:

- Training is significantly slower than classical approaches
- Limited expressivity with shallow circuits
- No real quantum advantage demonstrated

For production anomaly detection, **classical models are recommended**.

## 📄 License

MIT License

## 🙏 Acknowledgments

- [Kaggle Credit Card Fraud Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- [PennyLane](https://pennylane.ai/) for QML framework
- [PyTorch](https://pytorch.org/) for neural networks
