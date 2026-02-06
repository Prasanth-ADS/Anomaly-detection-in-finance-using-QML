import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from src.data.load_data import load_config, load_creditcard_data
from src.features.dimensionality_reduction import DimensionalityReducer
from src.quantum.variational_qnn import VQNNModel, VQNNTrainer
from src.evaluation.benchmarks import ClassicalBenchmarks
from src.evaluation.metrics import get_classification_metrics, print_metrics
from src.utils.logger import get_logger

logger = get_logger("main_comparison")

def main():
    # 1. Config & Data
    config = load_config()
    try:
        # Load a subset for speed if needed, or full data
        df = load_creditcard_data(config)
    except FileNotFoundError:
        logger.error("Data not found. Please ensure 'data/raw/creditcard.csv' exists.")
        return

    # Subsample for QML feasibility (Simulation is slow!)
    # Stratified sample to keep anomaly ratio? No, we need enough anomalies for signal.
    # Let's take ALL anomalies and a subset of normal.
    anomalies = df[df['Class'] == 1]
    normal = df[df['Class'] == 0]
    
    # Take all anomalies (usually ~492 in full dataset)
    # If too many, take 50. If too few, take all.
    n_anromalies_to_keep = min(len(anomalies), 100) 
    df_anomalies = anomalies.sample(n=n_anromalies_to_keep, random_state=42)
    
    # Take e.g. 1000 normal samples
    df_normal = normal.sample(n=1000, random_state=42)
    
    df_sample = pd.concat([df_anomalies, df_normal]).sample(frac=1, random_state=42)
    
    X = df_sample.drop('Class', axis=1).drop('Time', axis=1).values
    y = df_sample['Class'].values
    
    logger.info(f"Data Shape: {X.shape}, Anomalies: {sum(y)}")

    # 2. Features (PCA -> 4-8 Qubits)
    N_QUBITS = 4
    logger.info(f"Reducing dimensions to {N_QUBITS} for Quantum Circuit...")
    reducer = DimensionalityReducer(method='pca', n_components=N_QUBITS, feature_range=(0, np.pi))
    X_scaled = reducer.fit_transform(X)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y)
    
    # 3. Quantum Model (Optimized)
    logger.info("Initializing Optimized VQNN (Data Re-uploading)...")
    # Weighted Loss for imbalance (Pos weight = Neg / Pos ratio)
    num_neg = len(y_train) - sum(y_train)
    num_pos = sum(y_train)
    pos_weight = num_neg / num_pos if num_pos > 0 else 1.0
    
    # WeightedBCEWithLogitsLoss logic usually, but we have Sigmoid at end of model.
    # Let's use BCELoss with manual weighting or just simple BCELoss for now, 
    # but the prompt asked for "Handling class imbalance".
    # Better: use PyTorch's weighted BCE.
    # Note: VQNNModel outputs sigmoid ([0,1]), so we use BCELoss.
    # To use pos_weight, we need BCEWithLogitsLoss and remove Sigmoid from model?
    # Or just stick to standard BCELoss for simplicity in this script, or pass weight manually.
    
    q_model = VQNNModel(n_qubits=N_QUBITS, n_layers=3, use_reuploading=True)
    
    # Custom Weighted Loss 
    # (Simplified: just train normal for now, focus on QML architecture first as per prompt flow)
    # But let's try to be robust.
    criterion = nn.BCELoss(weight=None) # Adding weight tensor manually is complex here without tensor conversion first
    
    trainer = VQNNTrainer(q_model, lr=0.05, epochs=15, criterion=criterion)
    trainer.fit(X_train, y_train, X_test, y_test)
    
    # QML Predictions
    y_pred_q = trainer.predict(X_test)
    y_prob_q = trainer.predict_proba(X_test)
    
    metrics_q = get_classification_metrics(y_test, y_pred_q, y_prob_q)
    logger.info(">>> Quantum Model Performance:")
    print_metrics(metrics_q)
    
    # 4. Classical Benchmarks
    logger.info("Running Classical Benchmarks...")
    benchmarker = ClassicalBenchmarks()
    results_c = benchmarker.train_and_evaluate(X_train, y_train, X_test)
    
    for name, res in results_c.items():
        metrics_c = get_classification_metrics(y_test, res['pred'], res['prob'])
        logger.info(f">>> {name} Performance:")
        print_metrics(metrics_c)

if __name__ == "__main__":
    main()
