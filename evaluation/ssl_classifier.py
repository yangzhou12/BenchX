import torch
import torch.nn as nn


class SSLClassifier(nn.Module):
    def __init__(self, backbone=None, image_grad=True, num_classes=0):
        super(SSLClassifier, self).__init__()
        
        self.img_encoder = backbone
        self.feature_dim = get_encoder_output_dim(self.img_encoder)
        self.classifier = nn.Linear(self.feature_dim, num_classes)

    def forward(self, x):
        x = self.img_encoder(x)
        pred = self.classifier(x)
        return pred
    

@torch.no_grad()
def get_encoder_output_dim(module: torch.nn.Module) -> int:
    """Calculate the output dimension of ssl encoder by making a single forward pass.
    :param module: Encoder module.
    """
    # Target device
    device = next(module.parameters()).device  # type: ignore
    assert isinstance(device, torch.device)

    x = torch.rand((1, 3, 448, 448)).to(device)

    # Extract the number of output feature dimensions
    representations = module(x)
    return representations.shape[1]


        
        

        