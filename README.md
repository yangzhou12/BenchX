# Unified Framework for Medical Vision-Language Models

## Downstream Evaluation

### 0. Download Datasets

- MIMIC-CXR: We downloaded the MIMIC-CXR-JPG dataset as the radiographs. Paired medical reports can be downloaded in MIMIC-CXR.

- CheXpert: We downloaded the CheXpert small dataset in Kaggle.

- RSNA Pneumonia: We used the stage 2 of RSNA dataset in Kaggle.

- SIIM: We downloaded the stage 1 of SIIM dataset in Kaggle.

Change dataset paths in `constants.py` accordingly.


### 1. Dataset Preparation

Please organize the fine-tuning datasets as the following structure:

```
root:[data]
+--finetune_data
| +--CheXpert-v1.0-small
| | +--train
| | +--valid
| | +--train.csv
| | +--valid.csv
| +--mimic_512
| | +--files
| | +--mimic-cxr-2.0.0-chexpert.csv
| | +--mimic-cxr-2.0.0-metadata.csv
| | +--mimic-cxr-2.0.0-negbio.csv
| | +--mimic-cxr-2.0.0-split.csv
| +--nih_chest_xray
| | +--all_images
| | +--test_list.txt
| | +--train_val_list.txt
| +--rsna_pneumonia
| | +--stage_2_test_images
| | +--stage_2_train_images
| | +--stage_2_detailed_class_info.csv
| | +--stage_2_sample_submission.csv
| | +--stage_2_train_labels.csv
| +--siim-acr-pneumothorax
| | +--dicom-images-test
| | +--dicom-images-train
| | +--train-rle.csv
```


### 2. Pre-processing

Run the following command to pre-process the data:

```
cd preprocessing/
python chexpert.py
python mimic_cxr.py
python rsna_pneumonia.py
python siim_pneumothorax.py
```


### 3. Fine-Tuning

Now you can start to fine-tune the pre-trained model:

```
cd evaluation/{downstream_task}
chmod a+x ./run_{task_name}.sh
./run_{task_name}.sh
```

Supported Tasks:

* Uni-modal Tasks
    * Multi-label Classification on CheXpert (Zero-shot & Fine-tuned)
    * Classification on RSNA Pneumonia (Zero-shot & Fine-tuned)
* Cross-modal Tasks
    * Cross-modal Retrieval on CheXpert-5x200/MIMIC-5x200 (Zero-shot)
    * Cross-modal Retrieval on CheXpert-5x200/MIMIC-5x200 (Fine-tuned)
* Multi-modal Tasks
    * Visual Question Answering on VQA-RAD
    * Visual Question Answering on SLACK
    * Visual Question Answering on MedVQA-2019
