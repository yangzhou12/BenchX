import torch.nn as nn
import math


# Config values
patch_size = (32, 32)
in_channels = 3
out_channels = 1
hidden_size = 1024
num_hidden_layers = 8
num_attention_heads = 16
decode_features = [512, 256, 128, 64]
sample_rate = 4



# class Decoder2D(nn.Module):
#     def __init__(self, in_channels, out_channels, features=[512, 256, 128, 64]):
#         super().__init__()
#         self.decoder_1 = nn.Sequential(
#             nn.Conv2d(in_channels, features[0], 3, padding=1),
#             nn.BatchNorm2d(features[0]),
#             nn.ReLU(inplace=True),
#             nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
#         )
#         self.decoder_2 = nn.Sequential(
#             nn.Conv2d(features[0], features[1], 3, padding=1),
#             nn.BatchNorm2d(features[1]),
#             nn.ReLU(inplace=True),
#             nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
#         )
#         self.decoder_3 = nn.Sequential(
#             nn.Conv2d(features[1], features[2], 3, padding=1),
#             nn.BatchNorm2d(features[2]),
#             nn.ReLU(inplace=True),
#             nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
#         )
#         self.decoder_4 = nn.Sequential(
#             nn.Conv2d(features[2], features[3], 3, padding=1),
#             nn.BatchNorm2d(features[3]),
#             nn.ReLU(inplace=True),
#             nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
#         )

#         self.final_out = nn.Conv2d(features[-1], out_channels, 3, padding=1)

#     def forward(self, x):
#         x = self.decoder_1(x)
#         x = self.decoder_2(x)
#         x = self.decoder_3(x)
#         x = self.decoder_4(x)
#         x = self.final_out(x)
#         return x
    


# class UNet(nn.Module):
#     def __init__(self, encoder):
#         super().__init__()

        
