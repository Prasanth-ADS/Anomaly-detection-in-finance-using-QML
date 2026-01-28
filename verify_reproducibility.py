import numpy as np
import torch
from src.utils.seed import set_seed
from src.classical.xgboost_model import XGBoostModel

def test_reproducibility():
    """
    Runs a small experiment twice to ensure results are identical.
    """
    X = np.random.rand(100, 5)
    y = np.random.randint(0, 2, 100)
    
    # Run 1
    set_seed(42)
    model1 = XGBoostModel(random_state=42)
    model1.fit(X, y)
    pred1 = model1.predict_proba(X)
    
    # Run 2
    set_seed(42)
    model2 = XGBoostModel(random_state=42)
    model2.fit(X, y)
    pred2 = model2.predict_proba(X)
    
    if np.array_equal(pred1, pred2):
        print("✅ Reproducibility Test Passed: XGBoost predictions are identical.")
    else:
        print("❌ Reproducibility Test Failed: Predictions differ.")
        
    # Check Torch/Quantum Seed (Simulated via simple torch random)
    set_seed(42)
    t1 = torch.rand(5)
    set_seed(42)
    t2 = torch.rand(5)
    
    if torch.all(t1.eq(t2)):
        print("✅ Reproducibility Test Passed: Torch RNG is identical.")
    else:
        print("❌ Reproducibility Test Failed: Torch RNG differs.")

if __name__ == "__main__":
    test_reproducibility()
