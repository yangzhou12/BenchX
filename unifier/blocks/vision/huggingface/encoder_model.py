import os
import torch
import torch.nn as nn

from transformers.models.auto import AutoModel, AutoConfig


class HFVisionEncoder(nn.Module):
    """
    If proto is mentioned in encoder dict, loads pretrained models from proto strings.
    Otherwise, loads a BertGenerationEncoder model from encoder dict.
    """

    def __init__(self, encoder, **kwargs):
        super().__init__()
        path = encoder.pop("proto")
        self.enc_config = AutoConfig.from_pretrained(path)
        self.encoder = AutoModel.from_pretrained(path, config=self.enc_config)

        # Load custom pretrained weights
        if encoder.pretrained and os.path.exists(encoder.pretrained):
            checkpoint = torch.load(encoder.pretrained, map_location="cpu")

            for key in ["state_dict", "model"]:  # resolve differences in saved checkpoints
                if key in checkpoint:
                    checkpoint = checkpoint[key]
                    break
            
            if encoder.prefix:
                state_dict = {k.replace(encoder.prefix, ""): v for k, v in checkpoint.items() if encoder.prefix in k}
            else:
                state_dict = checkpoint
                print("Checkpoint prefix not set; Full state dictionary returned")

            msg = self.encoder.load_state_dict(state_dict, strict=False)
            print(f"(Vision) Missing keys: {msg.missing_keys}\nUnexpected keys: {msg.unexpected_keys}")

    def forward(self, pixel_values):
        '''args:
        pixel_values: tensor with shape [bs, 3, img_size, img_size]
        '''
        output = self.encoder(pixel_values)
        img_embeds = output['last_hidden_state']
        return img_embeds

    def __repr__(self):
        s = str(type(self.encoder).__name__) + "(" + str(self.encoder.config) + ")\n"
        return s