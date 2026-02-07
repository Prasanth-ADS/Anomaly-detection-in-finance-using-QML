# Anomaly Detection Benchmark Pipeline - Configuration
"""
Global configuration for the anomaly detection benchmark.
This file defines constants, paths, and settings used across all notebooks.
"""

import os
from pathlib import Path

# =============================================================================
# RANDOM SEED - Fixed for reproducibility
# =============================================================================
RANDOM_SEED = 42

# =============================================================================
# PATHS
# =============================================================================
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SPLITS_DIR = DATA_DIR / "splits"
FEATURES_DIR = DATA_DIR / "features"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = BASE_DIR / "figures"

# Create directories if they don't exist
for dir_path in [PROCESSED_DIR, SPLITS_DIR, FEATURES_DIR, MODELS_DIR, RESULTS_DIR, FIGURES_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# =============================================================================
# DATASET CONFIGURATION
# =============================================================================
RAW_DATASET_PATH = RAW_DATA_DIR / "creditcard.csv"
TARGET_COLUMN = "Class"
MAX_SAMPLES = 2000
MAX_FEATURES = 10
ANOMALY_RATIO = 0.05  # Target 5% anomalies after oversampling
TEST_SIZE = 0.2

# =============================================================================
# HARDWARE CONSTRAINTS
# =============================================================================
MAX_RAM_GB = 16
MAX_GPU_VRAM_GB = 6

# =============================================================================
# QML CONSTRAINTS
# =============================================================================
QML_MAX_QUBITS = 8
QML_MIN_QUBITS = 2
QML_MAX_CIRCUIT_DEPTH = 6
QML_SHOTS = 1024
QML_USE_SIMULATOR = True

# =============================================================================
# MODEL TRAINING
# =============================================================================
EPOCHS_NEURAL = 100
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EARLY_STOPPING_PATIENCE = 10

# =============================================================================
# CROSS-VALIDATION
# =============================================================================
CV_FOLDS = 5

# =============================================================================
# OUTPUT FILES
# =============================================================================
OUTPUT_FILES = {
    # Notebook 1
    "cleaned_data": PROCESSED_DIR / "cleaned_data.csv",
    "data_summary": RESULTS_DIR / "data_summary.json",
    
    # Notebook 2
    "engineered_train": SPLITS_DIR / "engineered_train.csv",
    "engineered_test": SPLITS_DIR / "engineered_test.csv",
    "split_metadata": RESULTS_DIR / "split_metadata.json",
    "scaler": MODELS_DIR / "scaler.pkl",
    
    # Notebook 3
    "feature_train": FEATURES_DIR / "feature_engineered_train.csv",
    "feature_test": FEATURES_DIR / "feature_engineered_test.csv",
    "feature_report": RESULTS_DIR / "feature_report.json",
    
    # Notebook 4
    "pca_train": FEATURES_DIR / "pca_train.csv",
    "pca_test": FEATURES_DIR / "pca_test.csv",
    "pca_model": MODELS_DIR / "pca_model.pkl",
    
    # Notebook 5
    "best_params": RESULTS_DIR / "best_params.json",
    "tuning_results": RESULTS_DIR / "tuning_results.csv",
    
    # Notebook 6-8
    "classical_metrics": RESULTS_DIR / "classical_metrics.csv",
    "neural_metrics": RESULTS_DIR / "neural_metrics.csv",
    "qml_metrics": RESULTS_DIR / "qml_metrics.csv",
}

# =============================================================================
# METRICS TO COMPUTE
# =============================================================================
CLASSIFICATION_METRICS = [
    "accuracy",
    "precision", 
    "recall",
    "f1_score",
    "roc_auc",
    "fpr",  # False Positive Rate
]

RECONSTRUCTION_METRICS = [
    "mse",
    "mae",
    "reconstruction_error",
]

QML_SPECIFIC_METRICS = [
    "n_qubits",
    "circuit_depth",
    "n_shots",
    "training_time",
    "inference_time",
]
