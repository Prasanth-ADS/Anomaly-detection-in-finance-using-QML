import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import time

# Add project root to path
project_root = os.path.abspath(os.path.join(os.getcwd()))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.data.load_data import load_config, load_creditcard_data
from src.features.dimensionality_reduction import DimensionalityReducer
from src.classical.baseline_models import (
    LogisticRegressionModel, SVMModel, RandomForestModel, 
    MLPModel, IsolationForestModel, AutoEncoderModel
)
from src.quantum.quantum_models import VQCModel, QNNModel, QSVMModel, QAEModel
from src.evaluation.metrics import get_classification_metrics, print_metrics
from src.utils.seed import set_seed
from src.utils.logger import get_logger

logger = get_logger("research_benchmark")

def run_research_benchmark():
    config = load_config('src/config/config.yaml')
    set_seed(config['project']['seed'])
    
    # 1. Load and Limit Data (Constraint: 2,000 samples)
    df = load_creditcard_data(config)
    n_samples = config['data']['max_samples']
    n_anomalies = int(n_samples * config['data']['contamination'])
    
    anomalies = df[df['Class'] == 1].sample(n=min(len(df[df['Class'] == 1]), n_anomalies), random_state=42)
    normal = df[df['Class'] == 0].sample(n=n_samples - len(anomalies), random_state=42)
    df_mini = pd.concat([anomalies, normal]).sample(frac=1, random_state=42)
    
    X = df_mini.drop(['Class', 'Time'], axis=1).values
    y = df_mini['Class'].values
    
    logger.info(f"Dataset: {X.shape[0]} samples, {sum(y)} anomalies ({sum(y)/len(y)*100:.2f}%)")
    
    # 2. Preprocess (Constraint: 10 features max)
    n_features = config['features']['n_components']
    reducer = DimensionalityReducer(method='pca', n_components=n_features)
    X_pca = reducer.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_pca, y, test_size=config['data']['test_size'], random_state=42, stratify=y
    )
    
    results = []

    # --- Classical Baselines ---
    classical_models = {
        "Logistic Regression": LogisticRegressionModel(),
        "SVM (RBF)": SVMModel(kernel='rbf'),
        "Random Forest": RandomForestModel(),
        "MLP": MLPModel(),
        "Isolation Forest": IsolationForestModel(contamination=config['data']['contamination']),
        "Autoencoder": AutoEncoderModel(input_dim=n_features)
    }

    for name, model in classical_models.items():
        logger.info(f"Evaluating Classical: {name}")
        start_train = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_train
        
        start_inf = time.time()
        y_pred = model.predict(X_test)
        inf_time = time.time() - start_inf
        
        # Probabilities
        y_prob = None
        if hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(X_test)
        elif hasattr(model, 'decision_function'):
            y_prob = model.decision_function(X_test)
        elif hasattr(model, 'predict_anomaly_score'):
            y_prob = model.predict_anomaly_score(X_test)

        m = get_classification_metrics(y_test, y_pred, y_prob)
        m.update({"Model": name, "Type": "Classical", "Train Time": train_time, "Inf Time": inf_time})
        results.append(m)

    # --- Quantum Models (Simulator-Based) ---
    quantum_models = {
        "VQC (4-Qubit)": VQCModel(n_qubits=4),
        "Hybrid QNN": QNNModel(n_qubits=4),
        "QSVM": QSVMModel(n_qubits=4),
        "Quantum AE": QAEModel(n_qubits=6)
    }

    for name, q_model in quantum_models.items():
        logger.info(f"Evaluating Quantum: {name}")
        start_train = time.time()
        q_model.fit(X_train, y_train)
        train_time = time.time() - start_train
        
        start_inf = time.time()
        y_pred = q_model.predict(X_test)
        inf_time = time.time() - start_inf
        
        y_prob = None
        if hasattr(q_model, 'predict_proba'):
            y_prob = q_model.predict_proba(X_test)
        elif hasattr(q_model, 'predict_anomaly_score'):
            y_prob = q_model.predict_anomaly_score(X_test)

        m = get_classification_metrics(y_test, y_pred, y_prob)
        m.update({"Model": name, "Type": "Quantum", "Train Time": train_time, "Inf Time": inf_time})
        results.append(m)

    # 3. Final Report Generation
    df_res = pd.DataFrame(results)
    os.makedirs('results/research', exist_ok=True)
    df_res.to_csv('results/research/final_comparison.csv', index=False)
    
    # Plotting
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df_res, x='Model', y='pr_auc', hue='Type')
    plt.title("Constraint-Compliant Benchmark: PR-AUC Comparison")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('results/research/pr_auc_comparison.png')
    
    logger.info("Research Benchmark Complete. Results saved to results/research/")
    print("\nFinal Research Comparison (Sorted by PR-AUC):")
    print(df_res[['Model', 'Type', 'pr_auc', 'roc_auc', 'f1', 'fpr', 'accuracy']].sort_values('pr_auc', ascending=False))

if __name__ == "__main__":
    run_research_benchmark()
