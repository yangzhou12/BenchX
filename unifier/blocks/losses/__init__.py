# PyTorch built-in loss functions
from torch.nn import BCEWithLogitsLoss

# Custom losses
from .ConVIRTLoss import ConVIRTLoss
from .GLoRIALoss import GLoRIALoss, gloria_attention_fn, cosine_similarity
from .InfoNCELoss import InfoNCELoss
