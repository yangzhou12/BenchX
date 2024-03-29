# BenchX: A Unified Evaluation Framework for Medical Vision-Language Models on Chest X-Rays

This repository includes the [preview code](https://anonymous.4open.science/r/BenchX-46C5/) for ICML24 Submission (Paper ID 9315): 

"BenchX: Benchmarking Medical Vision-Language Pretraining on Chest X-Rays"

## Installation
```bash
conda create -n BenchX python=3.11
conda activate BenchX
```

Clone the repository and install the package:
```bash
git clone https://github.com/XXXXX/BenchX
cd BenchX
pip install -r requirements.txt
```

## Evaluation

### 0. Download Datasets

- MIMIC-CXR: We downloaded the [MIMIC-CXR-JPG](https://physionet.org/content/mimic-cxr-jpg/2.0.0/) dataset for pre-training using paired medical reports and images.

- CheXpert: We downloaded the [CheXpert-v1.0-small](https://stanfordmlgroup.github.io/competitions/chexpert/#:~:text=What%20is%20CheXpert%3F,labeled%20reference%20standard%20evaluation%20sets) dataset from Kaggle.

- RSNA Pneumonia: We used the stage 2 data of the [RSNA Pneumonia](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge) dataset from Kaggle.

- SIIM: We downloaded the stage 1 data of the [SIIM-ACR Pneumothorax Segmentation](https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation) dataset from Kaggle.

- NIH Chest X-rays: We downloaded Version 3 of the [NIH Chest X-rays](https://www.kaggle.com/datasets/nih-chest-xrays/data) dataset from Kaggle. All images from `img_0XX` folders are moved to a combined sub-folder `all_images/`.

Change dataset paths in [`config/templates/finetune/`](config/templates/finetune/) accordingly.


### 1. Dataset Preparation

Please organize the datasets as the following structure:

```
root:[data]
+--mimic_512
| +--files
| +--mimic-cxr-2.0.0-chexpert.csv
| +--mimic-cxr-2.0.0-metadata.csv
| +--mimic-cxr-2.0.0-negbio.csv
| +--mimic-cxr-2.0.0-split.csv
+--nih_chest_xray
| +--all_images
| +--test_list.txt
| +--train_val_list.txt
+--rsna_pneumonia
| +--stage_2_test_images
| +--stage_2_train_images
| +--stage_2_detailed_class_info.csv
| +--stage_2_sample_submission.csv
| +--stage_2_train_labels.csv
+--siim-acr-pneumothorax
| +--dicom-images-test
| +--dicom-images-train
| +--train-rle.csv
```

### 2. Pre-processing

Run the following commands to pre-process the dataset(s) specified below:

```
python -m preprocessing.chexpert
python -m preprocessing.mimic_cxr #mimic_cxr_from_csv if preprocessing CSV file containing reports
python -m preprocessing.rsna_pneumonia
python -m preprocessing.siim_pneumothorax
```

No preprocessing is required for the NIH Chest X-ray dataset.

### 3. Zero-shot Evaluation & Fine-tuning

We evaluate our pre-trained models by specifying the `--pretrain_path` argument before running each downstream task. Arguments can be modified through [`config/`](config/). Additional command-line arguments can also be specified to override the configuration setting.

To view all available models for evaluation, you may run the following script:
```python
from evaluation import available_models
available_models()
```

Supported Tasks:
* Uni-modal Tasks
    * Multi-label Classification on NIH (Fine-tuned)
    * Binary Classification on RSNA Pneumonia (Fine-tuned)
    * Image Segmentation on SIIM-ACR Pneumothorax (Fine-tuned)
* Cross-modal Tasks
    * Cross-modal Retrieval on MIMIC-5x200 (Zero-shot)

#### Finetuned Classification
```
python -m evaluation.classification.finetuned_classifier --config configs/finetuned_classification_config.yaml
```
#### Finetuned Segmentation
```
python -m evaluation.segmentation.finetuned_segmentation --config configs/finetuned_segmentation_config.yaml
```
#### Zero-shot Retrieval
```
python -m evaluation.retrieval.zeroshot_retrieval --config configs/zeroshot_retrieval_config.yaml
```



