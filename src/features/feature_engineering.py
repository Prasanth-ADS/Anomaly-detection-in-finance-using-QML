import pandas as pd
import numpy as np
from src.utils.logger import get_logger

logger = get_logger("feature_engineering")

def add_rolling_features(df, window_sizes=[3, 7]):
    """
    Adds rolling mean and std features.
    Note: reliable only if data is time-ordered and grouped by entity (e.g., account).
    CreditCard dataset 'Time' is relative delta, and transactions are likely not account-grouped easily 
    without 'AccountID' which is missing. 
    for PaySim, we have 'step' and 'nameOrig'.
    """
    logger.warning("Rolling features require time-series context (e.g. PaySim). Skipping for standard CreditCard if IDs missing.")
    return df

def feature_engineer_paysim(df):
    """Specific feature engineering for PaySim."""
    # Example: Calculate transaction frequency per account
    # df['step_diff'] = df.groupby('nameOrig')['step'].diff()
    return df

def basic_feature_engineering(df):
    """Applies general feature engineering."""
    # Placeholder for z-scores if not already normalized
    return df
