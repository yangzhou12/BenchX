import os
import torch
import torch.nn as nn

from transformers.models.auto import AutoModel, AutoConfig
from transformers.models.bert_generation import (
    BertGenerationEncoder,
    BertGenerationConfig,
)

from transformers import RobertaConfig, RobertaModel
from transformers.models.bert.modeling_bert import BertPooler


def load_pretrained(network, pretrain_path, prefix):
    checkpoint = torch.load(pretrain_path, map_location="cpu")

    for key in ["state_dict", "model"]:  # resolve differences in saved checkpoints
        if key in checkpoint:
            checkpoint = checkpoint[key]
            break
    
    if prefix:
        state_dict = {k.replace(prefix, ""): v for k, v in checkpoint.items() if prefix in k}
    else:
        state_dict = checkpoint
        print("Checkpoint prefix not set; Full state dictionary returned")
    
    msg = network.load_state_dict(state_dict, strict=False)
    print(f"Missing keys: {msg.missing_keys}\nUnexpected keys: {msg.unexpected_keys}")
    
    return network


class EncoderModel(nn.Module):
    """
    If proto is mentioned in encoder dict, loads pretrained models from proto strings.
    Otherwise, loads a BertGenerationEncoder model from encoder dict.
    """

    def __init__(self, encoder, **kwargs):
        super().__init__()
        if encoder.proto is not None:
            path = encoder.pop("proto")
            self.enc_config = AutoConfig.from_pretrained(path)
            self.encoder = AutoModel.from_pretrained(path, config=self.enc_config)
        else:
            enc_config = BertGenerationConfig(
                **encoder, is_decoder=False, add_cross_attention=False
            )
            self.encoder = BertGenerationEncoder(enc_config)

        if encoder.add_pooling_layer:
            self.pooler = BertPooler(encoder)

        if encoder.language_projection:
            self.language_projection = eval(encoder.language_projection.layer)

            if encoder.pretrained:
                self.language_projection = load_pretrained(self.visual_projection, encoder.pretrained, self.language_projection.prefix)

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
            print(f"(Language) Missing keys: {msg.missing_keys}\nUnexpected keys: {msg.unexpected_keys}")

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        **kwargs
    ):
        out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        if hasattr(self, "pooler"):
            pooled_output = self.pooler(hidden_states=out.last_hidden_state)
            setattr(out, "pooler_output", pooled_output)

        return out

    def __repr__(self):
        s = str(type(self.encoder).__name__) + "(" + str(self.encoder.config) + ")\n"
        return s
