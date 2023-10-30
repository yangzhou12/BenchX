# PyTorch built-in loss functions
from torch.nn.modules.loss import *

# Custom losses
from .ConVIRTLoss import ConVIRTLoss
from .GLoRIALoss import GLoRIALoss, gloria_attention_fn, cosine_similarity
from .InfoNCELoss import InfoNCELoss
from .R2GenLMLoss import R2GenLMLoss
from .LabelSmoothingCrossEntropyLoss import LabelSmoothingCrossEntropy
