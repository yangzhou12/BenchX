# BenchX: A Unified Benchmark Framework for Medical Vision-Language Models on Chest X-Rays

## Downstream Evaluation

### 0. Download Datasets

- COVIDx: We used Version 9 of the [COVIDx CXR-4](https://www.kaggle.com/datasets/andyczhao/covidx-cxr2) dataset from Kaggle.

- NIH Chest X-rays: We used the [NIH Chest X-rays](https://huggingface.co/datasets/alkzar90/NIH-Chest-X-ray-dataset) dataset from Huggingface. 

- Object-CXR: We used the [object-CXR](https://www.kaggle.com/datasets/raddar/foreign-objects-in-chest-xrays) dataset from Kaggle.

- RSNA: We used the stage 2 data of the [RSNA Pneumonia](https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge) dataset from Kaggle.

- SIIM: We used the stage 1 data of the [SIIM-ACR Pneumothorax Segmentation](https://www.kaggle.com/datasets/vbookshelf/pneumothorax-chest-xray-images-and-masks) dataset from Kaggle.

- TBX11K: We used the [TBX11K Simplified](https://www.kaggle.com/datasets/vbookshelf/tbx11k-simplified) dataset from Kaggle.

- VinDr-CXR: We used the [VinDr-CXR](https://physionet.org/content/vindr-cxr/1.0.0/) dataset from PhysioNet.

- IU Xray: We used the preprocessed [IU Xray](https://drive.google.com/file/d/1c0BXEuDy8Cmm2jfN0YYGkQxFZd2ZIoLg) dataset from [R2Gen](https://github.com/cuhksz-nlp/R2Gen).

Please change dataset paths in the config files from [`config`](config) accordingly.

### 1. Dataset Preparation

Please run the scripts in [`preprocess`](preprocess) to pre-process each dataset and organize the processed datasets as the following structure:

```
root:[datasets]
+--COVIDx-CXR4
| +--test
| +--train
| +--val
| +--covidx_labels.csv
| +--test.txt
| +--train_1.txt
| +--train_10.txt
| +--train.txt
| +--val.txt
+--IU_Xray
| +--images
| +--iu_xray_annotations.csv
+--NIH_Chest_Xray
| +--images
| +--nih_labels.csv
| +--test.txt
| +--train_1.txt
| +--train_10.txt
| +--train.txt
| +--val.txt
+--Object-CXR
| +--images
| +--masks
| +--object_cxr_labels.csv
| +--test.txt
| +--train_1.txt
| +--train_10.txt
| +--train.txt
| +--val.txt
+--RSNA
| +--images
| +--masks
| +--rsna_labels.csv
| +--test.txt
| +--train_1.txt
| +--train_10.txt
| +--train.txt
| +--val.txt
+--SIIM
| +--images
| +--masks
| +--siim_labels.csv
| +--test.txt
| +--train_1.txt
| +--train_10.txt
| +--train.txt
| +--val.txt
+--TBX11K
| +--images
| +--masks
| +--tbx11k_labels.csv
| +--test.txt
| +--train_1.txt
| +--train_10.txt
| +--train.txt
| +--val.txt
+--VinDr_CXR
| +--images
| +--masks
| +--test.txt
| +--train_1.txt
| +--train_10.txt
| +--train.txt
| +--val.txt
| +--vindr_labels.csv
```

### 2. Classification

### 3. Segmentation

We evaluate our pre-trained models by specifying the `--pretrain_path` argument before running each downstream task. Arguments can be modified through [`configs/`](configs/). Additional command-line arguments can also be specified to override the configuration setting.

To view all available models for evaluation, you may run the following script:
```python
from evaluation import available_models
available_models()
```

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


