import random
import os
import numpy as np
import torch

def set_seed(seed=42):
    """Sets the seed for reproducibility."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # Note: PennyLane and Qiskit seeds depend on the specific backend/execution
    # and should be handled in their respective modules, but often rely on numpy.
    print(f"Random seed set to {seed}")
