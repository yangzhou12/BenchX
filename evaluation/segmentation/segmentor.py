import torch.nn as nn
import math



# class Encoder2D(nn.Module):
#     def __init__(self, trans_enc=None, is_segmentation=True):
#         super().__init__()
#         self.encoder = trans_enc
#         sample_rate = config.sample_rate
#         sample_v = int(math.pow(2, sample_rate))
#         assert config.patch_size[0] * config.patch_size[1] * \
#             config.hidden_size % (sample_v**2) == 0, "不能除尽"
#         self.final_dense = nn.Linear(
#             config.hidden_size, config.patch_size[0] * config.patch_size[1] * config.hidden_size // (sample_v**2))
#         self.patch_size = config.patch_size
#         self.hh = self.patch_size[0] // sample_v
#         self.ww = self.patch_size[1] // sample_v

#         self.is_segmentation = is_segmentation

#     def forward(self, x):
#         b, c, h, w = x.shape
#         p1 = self.patch_size[0]
#         p2 = self.patch_size[1]

#         hh = h // p1
#         ww = w // p2

#         encode_x = self.encoder(x)[:, 1:]
#         if not self.is_segmentation:
#             return encode_x

#         x = self.final_dense(encode_x)
#         x = rearrange(x, "b (h w) (p1 p2 c) -> b c (h p1) (w p2)",
#                       p1=self.hh, p2=self.ww, h=hh, w=ww, c=self.config.hidden_size)
#         return encode_x, x
    

class Decoder2D(nn.Module):
    def __init__(self, in_channels, out_channels, features=[512, 256, 128, 64]):
        super().__init__()
        self.decoder_1 = nn.Sequential(
            nn.Conv2d(in_channels, features[0], 3, padding=1),
            nn.BatchNorm2d(features[0]),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        )
        self.decoder_2 = nn.Sequential(
            nn.Conv2d(features[0], features[1], 3, padding=1),
            nn.BatchNorm2d(features[1]),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        )
        self.decoder_3 = nn.Sequential(
            nn.Conv2d(features[1], features[2], 3, padding=1),
            nn.BatchNorm2d(features[2]),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        )
        self.decoder_4 = nn.Sequential(
            nn.Conv2d(features[2], features[3], 3, padding=1),
            nn.BatchNorm2d(features[3]),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        )

        self.final_out = nn.Conv2d(features[-1], out_channels, 3, padding=1)

    def forward(self, x):
        x = self.decoder_1(x)
        x = self.decoder_2(x)
        x = self.decoder_3(x)
        x = self.decoder_4(x)
        x = self.final_out(x)
        return x
    

## ResUNet
## TransUNet