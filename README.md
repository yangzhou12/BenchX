# BenchX: A Unified Benchmark Framework for Medical Vision-Language Pretraining on Chest X-Rays

This repository includes the code for NeurIPS24 Datasets and Benchmarks Submission (Paper ID 279): 

"BenchX: A Unified Benchmark Framework for Medical Vision-Language Pretraining on Chest X-Rays"

## Installation
Create a conda environment and activate it:
```bash
conda create -n BenchX python=3.11
conda activate BenchX
```

Clone the repository of BenchX and install requirements:
```bash
git clone https://github.com/XXXXX/BenchX
cd BenchX
pip install -r requirements.txt
```

Install MMSegmentation following the [offitial instruction](https://mmsegmentation.readthedocs.io/en/latest/get_started.html):
```bash
# Install the dependency for MMSegmentation
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"

# Install MMSegmentation
git clone -b main https://github.com/open-mmlab/mmsegmentation.git
cd mmsegmentation
pip install -v -e .
```

## Dataset Preparation

We provide scripts to process existing datasets. In particular, we do not distribute datasets in this repository, and we do not own copyrights on any of the datasets.

The use of any of the datasets included in BenchX requires accepting its corresponding license on the original website. We refer to each dataset's README for more details.

<details close>
<summary><b>Supported Datasets and Tasks</b> (click to expand)</summary>

* [COVIDx CXR-4](https://www.kaggle.com/datasets/andyczhao/covidx-cxr2) (Binary Classification)
* [NIH Chest X-rays](https://huggingface.co/datasets/alkzar90/NIH-Chest-X-ray-dataset) (Multi-Label Classification)
* [Object-CXR](https://www.kaggle.com/datasets/raddar/foreign-objects-in-chest-xrays) (Binary Classification, Segmentation)
* [RSNA Pneumonia](https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge) (Binary Classification, Segmentation)
* [SIIM-ACR Pneumothorax Segmentation](https://www.kaggle.com/datasets/vbookshelf/pneumothorax-chest-xray-images-and-masks) (Binary Classification, Segmentation)
* [TBX11K](https://www.kaggle.com/datasets/vbookshelf/tbx11k-simplified) (Segmentation)
* [VinDr-CXR](https://physionet.org/content/vindr-cxr/1.0.0/) (Multi-Label Classification, Segmentation)
* [IU X-ray](https://drive.google.com/file/d/1c0BXEuDy8Cmm2jfN0YYGkQxFZd2ZIoLg) (Report Generation)

</details>

Please refer to [`datasets/README.md`](datasets/README.md) for more detail.

## Pre-Trained Checkpoints

We pre-train 8 MedVLP models on the same training set from [MIMIC-CXR](https://www.physionet.org/content/mimic-cxr-jpg/2.1.0/), and release the pre-trained checkpoints. Since the labels for training MedCLIP are not public avaliable, we use its offitial checkpoints for evaluation.
- CIFAR-10 [[Google Drive]](https://drive.google.com/file/d/1byGeYxM_PlLjT72wZsMQvP6popJeWBgt/view?usp=drive_link): ResNet-18 classifiers trained with cross-entropy loss from 3 training runs.
- CIFAR-100 [[Google Drive]](https://drive.google.com/file/d/1s-1oNrRtmA0pGefxXJOUVRYpaoAML0C-/view?usp=drive_link): ResNet-18 classifiers trained with cross-entropy loss from 3 training runs.
- ImageNet-200 [[Google Drive]](https://drive.google.com/file/d/1ddVmwc8zmzSjdLUO84EuV4Gz1c7vhIAs/view?usp=drive_link): ResNet-18 classifiers trained with cross-entropy loss from 3 training runs.
- ImageNet-1K [[Google Drive]](https://drive.google.com/file/d/15PdDMNRfnJ7f2oxW6lI-Ge4QJJH3Z0Fy/view?usp=drive_link): ResNet-50 classifiers including 1) the one from torchvision, 2) the ones that are trained by us with specific methods such as MOS, CIDER, and 3) the official checkpoints of data augmentation methods such as AugMix, PixMix.

Again, these checkpoints can be downloaded with the downloading script [here](https://github.com/Jingkang50/OpenOOD/tree/main/scripts/download).

<details close>
<summary><b>Supported MedVLP methods </b> (click to expand)</summary>

* [ConVIRT](https://github.com/edreisMD/ConVIRT-pytorch/tree/master): "Contrastive Learning of Medical Visual Representations from Paired Images and Text"
* [GLoRIA](https://github.com/marshuang80/gloria/tree/main): "GLoRIA: A Multimodal Global-Local Representation Learning Framework for Label-efficient Medical Image Recognition"
* [MedCLIP](https://github.com/RyanWangZf/MedCLIP): "MedCLIP: Contrastive Learning from Unpaired Medical Images and Texts"
* [MedKLIP](https://github.com/MediaBrain-SJTU/MedKLIP): "MedKLIP: Medical Knowledge Enhanced Language-Image Pre-Training in Radiology"
* [M-FLAG](https://github.com/cheliu-computation/M-FLAG-MICCAI2023): "M-FLAG: Medical Vision-Language Pre-training with Frozen Language Models and Latent Space Geometry Optimization"
* [MGCA](https://github.com/HKU-MedAI/MGCA/tree/main): "Multi-Granularity Cross-modal Alignment for Generalized Medical Visual Representation Learning"
* [PTUnifier](https://github.com/zhjohnchan/PTUnifier): "Towards Unifying Medical Vision-and-Language Pre-training via Soft Prompts"
* [MRM](https://github.com/RL4M/MRM-pytorch/tree/main): "Advancing Radiograph Representation Learning with Masked Record Modeling"
* [REFERS](https://github.com/funnyzhou/REFERS): "Generalized Radiograph Representation Learning via Cross-Supervision Between Images and Free-Text Radiology Reports"

</details>

## Fine-tuning & Evaluation

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
* Multi-modal Tasks
    * Report Generation on IU X-Ray (Fine-tuned)
    * Image-report Retrieval on MIMIC-5x200 (Zero-shot)

### 1. Classification

### 2. Segmentation

Build the unified segementation pipeline by adapting MMSegmentation:

We provide the necessary files for adapting MMSegmentation in the directory [unifed_segmentation](unifed_segmentation). After modifying MMSegmentaiton framework with the provided files, start fine-tuning and evaluation with [ft.sh](Siim_Segmentation/ft.sh) and [test.sh](Siim_Segmentation/test.sh), respectively.

We evaluate our pre-trained models by specifying the `--pretrain_path` argument before running each downstream task. Arguments can be modified through [`configs/`](configs/). Additional command-line arguments can also be specified to override the configuration setting.

To view all available models for evaluation, you may run the following script:
```python
from evaluation import available_models
available_models()
```

#### Finetuned Multi-Class Classification
```
python bin/train.py config/multiclass_classification/$dataset/$method.yml
```
#### Finetuned Multi-Label Classification
```
python bin/train.py config/multilabel_classification/$dataset/$method.yml
```
#### Finetuned Segmentation
```
python bin/train.py config/segmentation/$dataset/$method.yml
```
#### Zero-shot Retrieval
```
python bin/test_zeroshot.py config/retrieval/$dataset/$method.yml
```

## Acknowledgments

We thanks the following projects for reference of creating BenchX:

- [ViLMedic](https://github.com/jbdel/vilmedic)
- [TorchXRayVision](https://github.com/mlmed/torchxrayvision)
- [MRM](https://github.com/RL4M/MRM-pytorch/tree/main)
