import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from src.utils.logger import get_logger
from src.utils.seed import set_seed

logger = get_logger("split")

def split_data(df, target_col='Class', test_size=0.2, val_size=0.1, random_state=42):
    """
    Splits data into Train, Validation, and Test sets.
    Ensures stratified split for imbalanced datasets.
    """
    set_seed(random_state)
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # First split: Train+Val vs Test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    
    # Adjust val_size to be relative to the original dataset size
    # or relative to the remaining (Train+Val) size.
    # If val_size is 0.1 of total, and test is 0.2, then val is 0.1/0.8 = 0.125 of temp.
    relative_val_size = val_size / (1 - test_size)
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=relative_val_size, stratify=y_temp, random_state=random_state
    )
    
    logger.info(f"Data Split: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}")
    return X_train, y_train, X_val, y_val, X_test, y_test

def get_kfold_splits(X, y, n_splits=5, random_state=42):
    """Returns a generator for Stratified K-Fold splits."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return skf.split(X, y)
