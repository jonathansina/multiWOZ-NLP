import torch


if torch.cuda.is_available():
    DEVICE = torch.device("cuda")

elif torch.mps.is_available():
    DEVICE = torch.device("mps")
    
else:
    DEVICE = torch.device("cpu")
    
BATCH_SIZE = 64

MAX_LENGTH_ACTION = 128
MAX_LENGTH_RESPONSE = 128

MAX_TURNS = 2