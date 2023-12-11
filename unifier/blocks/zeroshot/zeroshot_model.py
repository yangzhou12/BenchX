import torch
import torch.nn as nn
from sklearn import metrics
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_curve


class ZeroshotModel(nn.Module):
    def __init__(self,
                 zeroshot_forward=None,
                 forward_embeddings=None,
                 global_similarities_func=None,
                 local_similarities_func=None,
                 similarity_type="both",
                 mode="classification"):
        super(ZeroshotModel, self).__init__()

        if mode not in ["retrieval", "classification"]:
            raise RuntimeError(f"Zero-shot task {mode} not supported")
        self.mode = mode

        if zeroshot_forward:
            # Initialize custom zeroshot forward method that returns [num_images, num_classes]
            self.forward = zeroshot_forward
        
        else:
            # Initialize model's image and text forward methods
            self.forward_embeddings = forward_embeddings

            # Similarity type
            self.similarity_type = similarity_type
            if self.similarity_type not in ["global", "local", "both"]:
                raise RuntimeError("Similarity type should be one of ['global', 'local', 'both']")
            if local_similarities_func == None and self.similarity_type in ["both", "local"]:
                raise RuntimeError("Local similarity function not specified")

            # Override similarities functions if they exist
            if global_similarities_func:
                self.get_global_similarities = global_similarities_func
            if local_similarities_func:
                self.get_local_similarities = local_similarities_func

    def get_global_similarities(self, img_emb, text_emb): # Taken from GLoRIA
        img_emb = img_emb.detach().cpu().numpy()
        text_emb = text_emb.detach().cpu().numpy()
        global_similarities = metrics.pairwise.cosine_similarity(img_emb, text_emb)
        global_similarities = torch.Tensor(global_similarities)
        return global_similarities # [num_images, num_prompts]

    def get_similarities(self, imgs, texts): 
        with torch.no_grad():
            outputs = self.forward_embeddings(imgs, texts) # imgs - [1000, ...]; texts - [5, ...]
            img_emb_g, text_emb_g = outputs["img_emb_g"], outputs["text_emb_g"]
            global_similarities = self.get_global_similarities(img_emb_g, text_emb_g)

            if hasattr(self, "get_local_similarities"):
                img_emb_l, text_emb_l = outputs["img_emb_l"], outputs["text_emb_l"]
                local_similarities = self.get_local_similarities(img_emb_l, text_emb_l, texts["cap_lens"])
            
        if self.similarity_type == "global":
            return global_similarities.detach().cpu().numpy()
        elif self.similarity_type == "local":
            return local_similarities.detach().cpu().numpy()
        else:
            similarities = (local_similarities + global_similarities) / 2
            return similarities.detach().cpu().numpy()
        
    def forward_prompts(self, img_values=None, prompt_inputs=None):
        class_similarities = []
        class_names = []
        
        for cls_name, cls_text in prompt_inputs.items():
            similarities = self.get_similarities(img_values, cls_text)
            cls_similarity = similarities.max(axis=1) # Take max as the logits similarity
            class_similarities.append(cls_similarity)
            class_names.append(cls_name)
        
        class_similarities = np.stack(class_similarities, 1)

        # standardize across class
        if class_similarities.shape[0] > 1:
            class_similarities = (class_similarities - class_similarities.mean(axis=0)) / (class_similarities.std(axis=0))
            # class_similarities = (class_similarities - class_similarities.min(axis=0)) / (class_similarities.max(axis=0) - class_similarities.min(axis=0))

        return {
            'logits': class_similarities, # num_images, num_classes
            'class_names': class_names
        }
    
    def forward_reports(self, img_values=None, report_texts=None):
        similarity_matrix = torch.tensor(self.get_similarities(img_values, report_texts)) # n_images, n_reports
        predictions = torch.argsort(similarity_matrix, descending=True, dim=1)
        return predictions

    def forward(self, img_values=None, text_inputs=None):
        if self.mode == "classification":
            outputs = self.forward_prompts(img_values, text_inputs)
        elif self.mode == "retrieval":
            outputs = self.forward_reports(img_values, text_inputs)
        return outputs
    
    def evaluate(self, outputs, targets):
        if self.mode == "classification":
            logits = outputs['logits']
            target_classes = outputs['class_names']
            max_f1s = []
            accs = []

            for i in range(len(target_classes)):
                gt_np = targets[:, i]
                pred_np = logits[:, i]
                precision, recall, thresholds = precision_recall_curve(gt_np, pred_np)
                numerator = 2 * recall * precision
                denom = recall + precision
                f1_scores = np.divide(numerator, denom, out=np.zeros_like(denom), where=(denom!=0))
                max_f1 = np.max(f1_scores)
                max_f1_thresh = thresholds[np.argmax(f1_scores)]
                max_f1s.append(max_f1)
                accs.append(accuracy_score(gt_np, pred_np > max_f1_thresh))

            return {
                "F1-score", np.array(max_f1s).mean(),
                "ACC", np.array(accs).mean()
            }

        elif self.mode == "retrieval":
            hits_at_k, precisions_at_k = (), ()

            for k in [1, 5, 10]:
                num_samples = len(outputs)
                num_hits = 0
                num_correct = 0

                for i in range(num_samples):  # for each text/image query
                    top_k_predictions = outputs[i][:k]  # extract top k candidate images/reports
                    hit = False
                    for idx in top_k_predictions:  # for each candidate
                        if torch.equal(targets[idx], targets[i]):  # if class of query matches class of candidate
                            num_correct += 1

                            if hit == False:
                                num_hits += 1
                                hit = True  # class of query is found in classes of top K candidates

                hit_frac = num_hits / num_samples
                hits_at_k += (hit_frac,)

                precision = num_correct / (num_samples * k)
                precisions_at_k += (precision,)

            H1, H5, H10 = hits_at_k
            P1, P5, P10 = precisions_at_k
            return {
                "Hit@1": H1, "Hit@5": H5, "Hit@10": H10,
                "Precision@1": P1, "Precision@5": P5, "Precision@10": P10
            }
            
