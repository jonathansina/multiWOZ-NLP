import torch

MODEL_NAME = "google-t5/t5-small" # "google-t5/t5-small", "google/t5-efficient-mini"
    
BATCH_SIZE = 256

MAX_LENGTH_ENCODER_ACTION = 64
MAX_LENGTH_DECODER_ACTION = 32

MAX_LENGTH_ENCODER_RESPONSE = 64
MAX_LENGTH_DECODER_RESPONSE = 32

MAX_TURNS = 2

USE_TRAINED_MODEL = False


if torch.cuda.is_available():
    DEVICE = torch.device("cuda")

elif torch.mps.is_available():
    DEVICE = torch.device("mps")
    
else:
    DEVICE = torch.device("cpu")