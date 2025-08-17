import torch


if torch.cuda.is_available():
    DEVICE = torch.device("cuda")

elif torch.mps.is_available():
    DEVICE = torch.device("mps")
    
else:
    DEVICE = torch.device("cpu")
    
BATCH_SIZE = 64
MAX_LENGTH = 128