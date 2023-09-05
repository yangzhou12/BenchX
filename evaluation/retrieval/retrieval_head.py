import torch
import torch.nn as nn
import numpy as np
from sklearn import metrics


class ZeroShotRetrieval(nn.Module):
    """
    Take in image and text encoders for zero-shot image-text retrieval.
    """

    def __init__(
        self,
        img_encoder_forward,
        text_encoder_forward,
        get_global_similarities=None,
        get_local_similarities=None,
        similarity_type="both",
    ):
        super(ZeroShotRetrieval, self).__init__()
        self.img_encoder_forward = img_encoder_forward
        self.text_encoder_forward = text_encoder_forward
        self.get_local_similarities = get_local_similarities

        ## Set similarity type
        if similarity_type not in ["global", "local", "both"]:
            raise RuntimeError(
                "Similarity type should be one of ['global', 'local', 'both']"
            )

        if get_local_similarities == None and similarity_type in ["both", "local"]:
            raise RuntimeError("Local similarity function not specified")

        self.similarity_type = similarity_type

        ## Set global similarity function
        if (
            get_global_similarities
        ):  # if custom global similarity function exists, use custom function
            self.get_global_similarities = get_global_similarities
        else:  # else, use GLoRIA global similarity function
            self.get_global_similarities = self.calc_global_similarities

    def calc_global_similarities(self, img_emb_g, text_emb_g):
        img_emb_g = img_emb_g.detach().cpu().numpy()
        text_emb_g = text_emb_g.detach().cpu().numpy()
        global_similarities = metrics.pairwise.cosine_similarity(img_emb_g, text_emb_g)
        global_similarities = torch.Tensor(global_similarities)
        return global_similarities

    def get_similarities(self, imgs, texts):
        with torch.no_grad():  # get image features and compute similarities (global and local)
            if self.get_local_similarities:
                img_emb_l, img_emb_g = self.img_encoder_forward(imgs)
                text_emb_l, text_emb_g, sents = self.text_encoder_forward(
                    texts["input_ids"], texts["attention_mask"], texts["token_type_ids"]
                )
                cap_lens = [
                    len([w for w in sent if not w.startswith("[")]) for sent in sents
                ]
                text_emb_l = text_emb_l[:, :, 1:]  # remove [CLS] token
                local_similarities = self.get_local_similarities(
                    img_emb_l, text_emb_l, cap_lens
                )
            else:
                img_emb_g = self.img_encoder_forward(imgs)
                text_emb_g = self.text_encoder_forward(
                    texts["input_ids"], texts["attention_mask"], texts["token_type_ids"]
                )

            # print(img_emb_g.shape, text_emb_g.shape)
            global_similarities = self.get_global_similarities(img_emb_g, text_emb_g)

        if self.similarity_type == "global":
            return global_similarities.detach().cpu().numpy()
        elif self.similarity_type == "local":
            return local_similarities.detach().cpu().numpy()
        else:
            # norm = lambda x: (x - x.mean(axis=0)) / (x.std(axis=0))
            # similarities = np.stack(
            #     [norm(local_similarities), norm(global_similarities)]
            # )
            similarities = np.stack([local_similarities, global_similarities])
            similarities = similarities.mean(axis=0)
            return similarities

    def forward(self, img_values=None, text_inputs=None, retrieve_images=False):
        similarity_matrix = torch.tensor(
            self.get_similarities(img_values, text_inputs)
        )  # n_images, n_reports

        if retrieve_images:  # text-to-image retrieval
            similarity_matrix = torch.transpose(
                similarity_matrix, 0, 1
            )  # n_reports, n_images

        predictions = torch.argsort(similarity_matrix, descending=True, dim=1)
        return predictions
