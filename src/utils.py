import random
import numpy as np

def set_global_seed(seed: int = 42):
    """Fixe les graines pour rendre les runs (plus) reproductibles."""
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass
    random.seed(seed)
    np.random.seed(seed)
