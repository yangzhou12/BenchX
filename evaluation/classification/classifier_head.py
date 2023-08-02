import torch
import torch.nn as nn


class ImageClassifier(nn.Module):
    '''
        Take in pre-trained image backbone (CNN or ViT) for finetuning classification.
    '''
    
    def __init__(self, backbone=None, image_grad=True, num_classes=0):
        super(ImageClassifier, self).__init__()
        
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


class PromptClassifier(nn.Module):
    '''
        Take in pre-trained text and image encoders with prompts and tokenizer for 
        zero-shot classification.
    '''

    def __init__(self, img_encoder_forward, text_encoder_forward, get_global_similarities, 
                 get_local_similarities, similarity_type="both"):
        super(PromptClassifier, self).__init__()
        self.img_encoder_forward = img_encoder_forward
        self.text_encoder_forward = text_encoder_forward
        self.get_global_similarities = get_global_similarities
        self.get_local_similarities = get_local_similarities

        if similarity_type not in ["global", "local", "both"]:
            raise RuntimeError(
                "Similarity type should be one of ['global', 'local', 'both']"
            )
        self.similarity_type = similarity_type

    def get_similarities(self, imgs, texts): # imgs and texts must be tensors

        # get global and local image features
        with torch.no_grad():
            img_emb_l, img_emb_g = self.img_encoder_forward(imgs)
            text_emb_l, text_emb_g, _ = self.text_encoder_forward(texts["caption_ids"], texts["attention_mask"], texts["token_type_ids"])
        
        global_similarities = self.get_global_similarities(img_emb_g, text_emb_g)
        local_similarities = self.get_local_similarities(img_emb_l, text_emb_l, texts["cap_lens"])
        similarities = (local_similarities + global_similarities) / 2

        if self.similarity_type == "global":
            return global_similarities.detach().cpu().numpy()
        elif self.similarity_type == "local":
            return local_similarities.detach().cpu().numpy()
        else:
            return similarities.detach().cpu().numpy()

    def forward(self, img_values=None, prompt_inputs=None):
        class_similarities = []
        class_names = []
        
        for cls_name, cls_text in prompt_inputs.items():
            similarities = self.get_similarities(img_values, cls_text)
            cls_similarity = torch.max(torch.from_numpy(similarities), 1)[0] # Take max as the logits similarity
            class_similarities.append(cls_similarity)
            class_names.append(cls_name)
        
        class_similarities = torch.stack(class_similarities, 1)

        # standardize across class
        if class_similarities.shape[0] > 1:
            class_similarities = (class_similarities - class_similarities.mean(axis=0)) / (class_similarities.std(axis=0))

        outputs = {
            'logits': class_similarities,
            'class_names': class_names
        }
        return outputs
    


        