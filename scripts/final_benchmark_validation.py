import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

# Add project root to path
project_root = os.path.abspath(os.path.join(os.getcwd()))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.data.load_data import load_config, load_creditcard_data
from src.features.dimensionality_reduction import DimensionalityReducer
from src.quantum.variational_qnn import VQNNModel, VQNNTrainer
from src.evaluation.benchmarks import ClassicalBenchmarks
from src.evaluation.metrics import get_classification_metrics
from src.evaluation.statistical_tests import perform_ttest
from src.utils.seed import set_seed
from src.utils.logger import get_logger

logger = get_logger("benchmark_validation")

def run_benchmark_validation(n_trials=5):
    config = load_config('src/config/config.yaml')
    
    # Load Data
    df = load_creditcard_data(config)
    
    # Subsample for specific test (ensure sufficient anomalies)
    anomalies = df[df['Class'] == 1]
    normal = df[df['Class'] == 0]
    
    n_anomalies = 100
    df_sample = pd.concat([
        anomalies.sample(n=n_anomalies, random_state=42),
        normal.sample(n=1000, random_state=42)
    ]).sample(frac=1, random_state=42)
    
    X = df_sample.drop(['Class', 'Time'], axis=1).values
    y = df_sample['Class'].values
    
    N_QUBITS = 4
    reducer = DimensionalityReducer(method='pca', n_components=N_QUBITS, feature_range=(0, np.pi))
    X_scaled = reducer.fit_transform(X)
    
    all_metrics = []
    
    seeds = [42, 123, 789, 456, 999][:n_trials]
    
    for trial, seed in enumerate(seeds):
        logger.info(f"--- Trial {trial+1}/{n_trials} (Seed {seed}) ---")
        set_seed(seed)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.3, random_state=seed, stratify=y
        )
        
        # Classical Baselines
        benchmarker = ClassicalBenchmarks(random_state=seed)
        c_results = benchmarker.train_and_evaluate(X_train, y_train, X_test)
        
        for name, res in c_results.items():
            metrics = get_classification_metrics(y_test, res['pred'], res['prob'])
            metrics['Model'] = name
            metrics['Trial'] = trial
            all_metrics.append(metrics)
            
        # Quantum VQNN
        logger.info("Training VQNN...")
        q_model = VQNNModel(n_qubits=N_QUBITS, n_layers=3, use_reuploading=True)
        trainer = VQNNTrainer(model=q_model, epochs=15)
        trainer.fit(X_train, y_train)
        
        y_pred_q = trainer.predict(X_test)
        y_prob_q = trainer.predict_proba(X_test)
        
        q_metrics = get_classification_metrics(y_test, y_pred_q, y_prob_q)
        q_metrics['Model'] = 'Quantum VQNN'
        q_metrics['Trial'] = trial
        all_metrics.append(q_metrics)
        
    df_results = pd.DataFrame(all_metrics)
    
    # Statistical Summary
    summary = df_results.groupby('Model')['pr_auc'].agg(['mean', 'std']).reset_index()
    
    # Compare each classical model with Quantum VQNN
    q_scores = df_results[df_results['Model'] == 'Quantum VQNN']['pr_auc'].values
    
    p_values = {}
    for model in df_results['Model'].unique():
        if model == 'Quantum VQNN':
            continue
        m_scores = df_results[df_results['Model'] == model]['pr_auc'].values
        _, p_val = perform_ttest(q_scores, m_scores)
        p_values[model] = p_val
        
    summary['p_value_vs_quantum'] = summary['Model'].map(p_values)
    
    # Save Results
    os.makedirs('results/benchmarks', exist_ok=True)
    summary.to_csv('results/benchmarks/statistical_validation.csv', index=False)
    
    # Plot Distribution
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df_results, x='Model', y='pr_auc', palette='viridis')
    plt.title("Benchmarking PR-AUC Stability (5 Trials)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('results/benchmarks/metric_stability.png')
    
    logger.info("Benchmark Validation Complete.")
    print("\nStatistical Validation Summary (PR-AUC):")
    print(summary)

if __name__ == "__main__":
    run_benchmark_validation(n_trials=5)
