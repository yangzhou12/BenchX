# PyTorch built-in loss functions
from torch.nn.modules.loss import *

# Custom losses
from .ConVIRTLoss import ConVIRTLoss
from .GLoRIALoss import GLoRIALoss, gloria_attention_fn, cosine_similarity
from .InfoNCELoss import InfoNCELoss
from .LabelSmoothingCrossEntropyLoss import LabelSmoothingCrossEntropy
