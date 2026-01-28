import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from src.utils.logger import get_logger

logger = get_logger("preprocess")

def clean_data(df):
    """Basic cleaning: remove duplicates, handle NaNs."""
    initial_shape = df.shape
    df = df.drop_duplicates()
    df = df.dropna()
    logger.info(f"Cleaned data: {initial_shape} -> {df.shape}")
    return df

def normalize_data(df, columns, method='minmax', feature_range=(0, 1)):
    """Normalizes specified columns."""
    if method == 'minmax':
        scaler = MinMaxScaler(feature_range=feature_range)
    elif method == 'standard':
        scaler = StandardScaler()
    else:
        raise ValueError("Unknown normalization method")
    
    df[columns] = scaler.fit_transform(df[columns])
    logger.info(f"Normalized {len(columns)} columns using {method}")
    return df, scaler

def preprocess_creditcard(df, config):
    """Specific preprocessing for credit card dataset."""
    # Drop Time as it's relative
    # Scale Amount and V features
    # V features are already PCA'd, but scaling -1 to 1 or 0 to 1 helps QNN
    
    df = clean_data(df)
    
    # Feature Engineering (Basic)
    # Usually we want 'Class' to be last or separate
    target_col = 'Class'
    feature_cols = [c for c in df.columns if c not in [target_col, 'Time']]
    
    # Normalize features to [0, 1] for quantum embedding (amplitude/angle encoding often needs specific ranges)
    # Angle encoding: [0, pi] or [0, 2pi]. Let's stick to [0, 1] then scale to pi later or here.
    df, scaler = normalize_data(df, feature_cols, method='minmax', feature_range=(0, 1))
    
    return df

if __name__ == "__main__":
    from src.data.load_data import load_config, load_creditcard_data
    config = load_config()
    df = load_creditcard_data(config)
    df_clean = preprocess_creditcard(df, config)
    print(df_clean.head())
