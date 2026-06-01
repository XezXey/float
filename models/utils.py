import torch
import numpy as np
import os
import random

def seed_everything(seed: int = 42):
    # 1. Set Python core random seed
    random.seed(seed)
    
    # 2. Set Python hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # 3. Set NumPy seed
    np.random.seed(seed)
    
    # 4. Set PyTorch CPU and CUDA seeds
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # Multi-GPU setup
    
    # 5. Force CUDA to use deterministic algorithms
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # 6. Optional: Throws an error if an operation cannot be made deterministic
    # torch.use_deterministic_algorithms(True)
