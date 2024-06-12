# BenchX: A Unified Benchmark Framework for Medical Vision-Language Pretraining on Chest X-Rays

This repository includes the code for NeurIPS24 Datasets and Benchmarks Submission (Paper ID 279): 

"BenchX: A Unified Benchmark Framework for Medical Vision-Language Pretraining on Chest X-Rays"

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Dataset Preparation

Please refer to [`datasets/README.md`](datasets/README.md) for training configs and results.

## Downstream Evaluation

### 2. Classification

### 3. Segmentation

We evaluate our pre-trained models by specifying the `--pretrain_path` argument before running each downstream task. Arguments can be modified through [`configs/`](configs/). Additional command-line arguments can also be specified to override the configuration setting.

To view all available models for evaluation, you may run the following script:
```python
from evaluation import available_models
available_models()
```

Supported MedVLP Methods:
<details open>
<summary>(click to collapse)</summary>
* [ConVIRT](https://github.com/edreisMD/ConVIRT-pytorch/tree/master)
* [GLoRIA](https://github.com/marshuang80/gloria/tree/main)
* [MedCLIP](https://github.com/RyanWangZf/MedCLIP)
* [MedKLIP](https://github.com/MediaBrain-SJTU/MedKLIP)
* [M-FLAG](https://github.com/cheliu-computation/M-FLAG-MICCAI2023)
* [MGCA](https://github.com/HKU-MedAI/MGCA/tree/main/mgca)
* [MRM](https://github.com/RL4M/MRM-pytorch/tree/main)
* [PTUnifier](https://github.com/zhjohnchan/PTUnifier) (segmentation only)
* [REFERS](https://github.com/funnyzhou/REFERS)
</details>

Supported Tasks:
* Uni-modal Tasks
    * Multi-label Classification on CheXpert (Fine-tuned)
    * Binary Classification on RSNA Pneumonia (Fine-tuned)
    * Semantic Segmentation on SIIM-ACR Pneumothorax (Fine-tuned)
* Cross-modal Tasks
    * Cross-modal Retrieval on CheXpert-5x200/MIMIC-5x200 (Zero-shot)
    * Cross-modal Classification on CheXpert-5x200 (Zero-shot)
    * Cross-modal Classification on RSNA Pneumonia (Zero-shot)
* Multi-modal Tasks
    * Visual Question Answering on Rad-Restruct
    * Visual Question Answering on VQA-RAD

#### Zero-shot Classification
```
python -m evaluation.classification.zeroshot_classifier --config configs/zeroshot_retrieval_config.yaml
```
#### Zero-shot Retrieval
```
python -m evaluation.retrieval.zeroshot_retrieval --config configs/zeroshot_retrieval_config.yaml
```
#### Finetuned Classification
```
python -m evaluation.classification.finetuned_classifier --config configs/finetuned_classification_config.yaml
```
#### Finetuned Segmentation
```
python -m evaluation.segmentation.finetuned_segmentation --config configs/finetuned_segmentation_config.yaml
```

## Acknowledgments

We thanks the following projects for reference of creating BenchX:

- [ViLMedic](https://github.com/jbdel/vilmedic)
- [TorchXRayVision](https://github.com/mlmed/torchxrayvision)
- [MRM](https://github.com/RL4M/MRM-pytorch/tree/main)
