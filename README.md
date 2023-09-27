# Unified Framework for Medical Vision-Language Models

## Downstream Evaluation

### 0. Download Datasets

- MIMIC-CXR: We downloaded the [MIMIC-CXR-JPG](https://physionet.org/content/mimic-cxr-jpg/2.0.0/) dataset for pre-training using paired medical reports and images.

- CheXpert: We downloaded the [CheXpert-v1.0-small](https://stanfordmlgroup.github.io/competitions/chexpert/#:~:text=What%20is%20CheXpert%3F,labeled%20reference%20standard%20evaluation%20sets) dataset from Kaggle.

- RSNA Pneumonia: We used the stage 2 data of the [RSNA Pneumonia](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge) dataset from Kaggle.

- SIIM: We downloaded the stage 1 data of the [SIIM-ACR Pneumothorax Segmentation](https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation) dataset from Kaggle.

- NIH Chest X-rays: We downloaded Version 3 of the [NIH Chest X-rays](https://www.kaggle.com/datasets/nih-chest-xrays/data) dataset from Kaggle. All images from `img_0XX` folders are moved to a combined sub-folder `all_images/`.
  
- Rad-Restruct: We downloaded the [Rad-ReStruct](https://github.com/ChantalMP/Rad-ReStruct/tree/master) medical VQA benchmark dataset through the official download link in the repo.

- VQA-RAD: We downloaded the [VQA-RAD](https://osf.io/89kps/) dataset through its official channel.

Change dataset paths in [`utils/constants.py`](utils/constants.py) accordingly.


### 1. Dataset Preparation

Please organize the datasets as the following structure:

```
root:[data]
+--CheXpert-v1.0-small
| +--train
| +--valid
| +--train.csv
| +--valid.csv
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

Note that we conduct our VQA experiments using the [Rad-ReStruct](https://github.com/ChantalMP/Rad-ReStruct/tree/master) benchmark repo and follow their data preparation steps instead for the Rad-Restruct and VQA-RAD datasets.


### 2. Pre-processing

Run the following commands to pre-process the dataset(s) specified below:

```
python -m preprocessing.chexpert
python -m preprocessing.mimic_cxr #mimic_cxr_from_csv if preprocessing CSV file containing reports
python -m preprocessing.rsna_pneumonia
python -m preprocessing.siim_pneumothorax
```

Note that no preprocessing is required for the SIIM dataset.

### 3. Zero-shot Evaluation & Fine-tuning

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
#### Finetuned Classification
```
python -m evaluation.classification.finetuned_classifier --config configs/finetuned_classification_config.yaml
```
#### Zero-shot Retrieval
```
python -m evaluation.classification.zeroshot_retrieval --config configs/zeroshot_retrieval_config.yaml
```
#### Finetuned Segmentation
```
python -m evaluation.classification.finetuned_segmentation --config configs/finetuned_segmentation_config.yaml
```
#### VQA
We conduct all experiments for medical VQA using the Rad-Restruct benchmark repo and provide the necessary files for reproducing the experiments [here](evaluation/vqa/).



