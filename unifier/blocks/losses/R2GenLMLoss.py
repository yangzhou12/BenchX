import torch
import torch.nn as nn


class R2GenLMLoss(nn.Module):
    def __init__(self):
        super(R2GenLMLoss, self).__init__()

    def forward(self, input, target, mask):
        # truncate to the same size
        target = target[:, : input.size(1)]
        mask = mask[:, : input.size(1)]
        output = -input.gather(2, target.long().unsqueeze(2)).squeeze(2) * mask
        output = torch.sum(output) / torch.sum(mask)

        return output
