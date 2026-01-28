import pandas as pd
import os
import yaml
from src.utils.logger import get_logger

logger = get_logger("load_data")

def load_config(config_path=None):
    if config_path is None:
        # Get the project root directory (assuming src/data/load_data.py structure)
        # file_dir = src/data
        # project_root = src/data/../../
        file_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(file_dir))
        config_path = os.path.join(project_root, "src", "config", "config.yaml")

    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    # Determine project root based on this script's location
    # src/data/load_data.py -> project_root is 2 levels up
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    
    # Update paths in config to be absolute
    if 'paths' in config:
        for key, path in config['paths'].items():
            # Only update if it's a relative path
            if path and not os.path.isabs(path):
                config['paths'][key] = os.path.join(project_root, path)

    return config

def load_creditcard_data(config):
    """Loads the Credit Card Fraud Detection dataset."""
    raw_path = config['paths']['raw_data']
    file_path = os.path.join(raw_path, "creditcard.csv")
    
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"File not found: {file_path}")
    
    logger.info(f"Loading Credit Card data from {file_path}")
    df = pd.read_csv(file_path)
    logger.info(f"Loaded Credit Card data with shape {df.shape}")
    return df

def load_paysim_data(config):
    """Loads the PaySim dataset."""
    raw_path = config['paths']['raw_data']
    file_path = os.path.join(raw_path, "paysim.csv")
    
    if not os.path.exists(file_path):
        logger.warning(f"File not found: {file_path}. Please download it.")
        return None
        
    logger.info(f"Loading PaySim data from {file_path}")
    df = pd.read_csv(file_path)
    logger.info(f"Loaded PaySim data with shape {df.shape}")
    return df

if __name__ == "__main__":
    config = load_config()
    try:
        load_creditcard_data(config)
    except Exception as e:
        print(e)
