import sys
import argparse
import pandas as pd
from src.utils.logger import get_logger, setup_logger
from src.utils.seed import set_seed
from src.data.load_data import load_config, load_creditcard_data
from src.data.preprocess import preprocess_creditcard
from src.data.split import split_data
from src.features.dimensionality_reduction import DimensionalityReducer
from src.classical.xgboost_model import XGBoostModel
from src.quantum.variational_qnn import VQNNModel, VQNNTrainer
from src.evaluation.metrics import get_classification_metrics, print_metrics

logger = get_logger("main_pipeline")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="src/config/config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config['project']['seed'])

    # 1. Load Data
    logger.info("Step 1: Loading Data...")
    try:
        df = load_creditcard_data(config)
    except Exception as e:
        logger.error(f"Cannot load data: {e}")
        return

    # 2. Preprocess
    logger.info("Step 2: Preprocessing...")
    df = preprocess_creditcard(df, config)
    
    X = df.drop(columns=['Class']).values
    y = df['Class'].values

    # 3. Feature Engineering / PCA
    logger.info("Step 3: Dimensionality Reduction...")
    n_components = config['features']['n_components']
    # Use new DimensionalityReducer
    reducer = DimensionalityReducer(method='pca', n_components=n_components)
    X_pca = reducer.fit_transform(X)

    # 4. Split
    logger.info("Step 4: Splitting Data...")
    
    # split_data expects a DataFrame with the target column
    df_pca = pd.DataFrame(X_pca)
    df_pca['Class'] = y
    
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(
        df_pca, target_col='Class',
        test_size=config['data']['test_size'], 
        val_size=config['data']['val_size']
    )
    # Convert back to numpy
    X_train = X_train.values
    X_val = X_val.values
    X_test = X_test.values
    y_train = y_train.values
    y_val = y_val.values
    y_test = y_test.values

    # 5. Train Classical Model (XGBoost)
    logger.info("Step 5: Training Classical Model (XGBoost)...")
    xgb = XGBoostModel()
    xgb.fit(X_train, y_train, X_val, y_val)
    y_pred_xgb = xgb.predict(X_test)
    probs_xgb = xgb.predict_proba(X_test)
    
    metrics_xgb = get_classification_metrics(y_test, y_pred_xgb, probs_xgb)
    logger.info("XGBoost Metrics:")
    print_metrics(metrics_xgb)

    # 6. Train Quantum Model (VQNN)
    # Note: Training VQNN on full dataset is slow. Subsampling for demo.
    logger.info("Step 6: Training Quantum Model (VQNN)...")
    logger.info("Subsampling training data for VQNN to 500 samples for speed...")
    X_train_q = X_train[:500]
    y_train_q = y_train[:500]
    X_test_q = X_test[:200]
    y_test_q = y_test[:200]
    
    # Instantiate Model first
    n_layers = config['quantum']['n_layers']
    q_model = VQNNModel(n_qubits=n_components, n_layers=n_layers, use_reuploading=True)
    
    vqnn = VQNNTrainer(model=q_model, epochs=5)
    vqnn.fit(X_train_q, y_train_q)
    y_pred_vqnn = vqnn.predict(X_test_q)
    
    metrics_vqnn = get_classification_metrics(y_test_q, y_pred_vqnn)
    logger.info("VQNN Metrics:")
    print_metrics(metrics_vqnn)

    logger.info("Pipeline Finished.")

if __name__ == "__main__":
    main()
