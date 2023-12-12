import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from segmentation_models_pytorch import Unet

from unifier.blocks.custom.setr import SETRModel
from unifier.blocks.losses import MixedLoss
from unifier.models.utils import get_n_params



def evaluation(models, config, dl, **kwargs):
    logits = np.zeros((len(dl.dataset), len(models)))
    masks = np.zeros(len(dl.dataset))
    losses = np.zeros((len(dl), len(models)))

    with torch.no_grad():
        for num_batch, batch in enumerate(tqdm(dl, total=len(dl))):
            batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            masks[num_batch] = batch["pathology_masks"]

            for num_model, model in enumerate(models):
                result = model(batch["images"])
                losses[num_batch][num_model] = result["loss"].cpu().item()
                logits[num_batch][num_model] = result["output"].cpu().item()

    preds = np.mean(logits, axis=1)
    loss = np.mean(losses)

    return {
        "loss": loss,
        "refs": masks,
        "hyps": preds,
    }


class ImageSegmenter(nn.Module):
    def __init__(self, cnn, loss, **kwargs):
        super(ImageSegmenter, self).__init__()

        backbone = cnn.get("backbone", None)

        # Use UNet for Resnet-50 backbone
        if backbone == "resnet50":
            self.model = Unet("resnet50", encoder_weights=None, activation=None)
            self.visual_encoder = self.model.encoder
            pretrained = cnn.pop("pretrained", None)
            if pretrained:
                self.model = self.load_encoder_from_checkpoint(pretrained, cnn.pop("prefix", None))
        
        else: # Use SETR model, replacing encoder
            self.model = SETRModel(
                patch_size=(16, 16),
                in_channels=3,
                out_channels=1,
                hidden_size=768,
                num_hidden_layers=12,
                num_attention_heads=12,
                decode_features=[512, 256, 128, 64],
            )
            cnn_func = cnn.pop("proto")
            visual_encoder = eval(cnn_func)(**cnn)
            self.model.encoder_2d.bert_model = visual_encoder

        # Loss function
        loss_func = loss.pop("proto")
        self.loss_func = eval(loss_func)(**loss).cuda()

        # Evaluation
        self.eval_func = evaluation

    def load_encoder_from_checkpoint(self, pretrained=None, prefix=None):
        checkpoint = torch.load(pretrained, map_location="cpu")

        for key in ["state_dict", "model"]:  # resolve differences in saved checkpoints
            if key in checkpoint:
                checkpoint = checkpoint[key]
                break
    
        if prefix:
            state_dict = {k.replace(prefix, ""): v for k, v in checkpoint.items() if prefix in k}
        else:
            state_dict = checkpoint
            print("Checkpoint prefix not set; Full state dictionary returned")

        for layer_k in ["head.weight", "head.bias", "fc.weight", "fc.bias"]:
            if layer_k in state_dict:
                del state_dict[layer_k]
    
        msg = self.visual_encoder.load_state_dict(state_dict, strict=True)

    def forward(self, images, pathology_masks=None, from_training=True, iteration=None, epoch=None, **kwargs):
        logit = self.model(images)
        out = logit.squeeze(dim=1)

        loss = torch.tensor(0.0)
        if from_training:
            loss = self.loss_func(out, pathology_masks.cuda().float(), **kwargs)

        return {"loss": loss, "output": out, "pred": torch.sigmoid(out)}

    def __repr__(self):
        s = super().__repr__() + "\n"
        s += "{}\n".format(get_n_params(self))
        return s
