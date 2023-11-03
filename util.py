import torch
import random
import numpy as np


# Seed 고정
def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

# Model Parameter 개수
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
