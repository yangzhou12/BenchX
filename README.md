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

>  We provide scripts to process existing datasets. Please refer to [`datasets/README.md`](datasets/README.md) for more detail. In particular, we do not distribute datasets in this repository, and we do not own copyrights on any of the datasets.

> The use of any of the datasets included in BenchX requires accepting its corresponding license on the original website. We refer to each dataset's README for more details.

## Pre-Trained Checkpoints

<details close>
<summary><b>Supported MedVLP methods </b> (click to expand)</summary>

* [ConVIRT](https://github.com/edreisMD/ConVIRT-pytorch/tree/master): "Contrastive Learning of Medical Visual Representations from Paired Images and Text" [[Ours]](checkpoints/pretrained)
* [GLoRIA](https://github.com/marshuang80/gloria/tree/main): "GLoRIA: A Multimodal Global-Local Representation Learning Framework for Label-efficient Medical Image Recognition" [[Official]](checkpoints/official) [[Ours]](checkpoints/pretrained)
* [MedCLIP](https://github.com/RyanWangZf/MedCLIP): "MedCLIP: Contrastive Learning from Unpaired Medical Images and Texts" [[Official]](checkpoints/official)
* [MedKLIP](https://github.com/MediaBrain-SJTU/MedKLIP): "MedKLIP: Medical Knowledge Enhanced Language-Image Pre-Training in Radiology" [[Official]](checkpoints/official) [[Ours]](checkpoints/pretrained)
* [M-FLAG](https://github.com/cheliu-computation/M-FLAG-MICCAI2023): "M-FLAG: Medical Vision-Language Pre-training with Frozen Language Models and Latent Space Geometry Optimization" [[Ours]](checkpoints/pretrained)
* [MGCA](https://github.com/HKU-MedAI/MGCA/tree/main): "Multi-Granularity Cross-modal Alignment for Generalized Medical Visual Representation Learning" [[Official]](checkpoints/official) [[Ours]](checkpoints/pretrained)
* [PTUnifier](https://github.com/zhjohnchan/PTUnifier): "Towards Unifying Medical Vision-and-Language Pre-training via Soft Prompts" [[Ours]](checkpoints/pretrained)
* [MRM](https://github.com/RL4M/MRM-pytorch/tree/main): "Advancing Radiograph Representation Learning with Masked Record Modeling" [[Official]](checkpoints/official) [[Ours]](checkpoints/pretrained)
* [REFERS](https://github.com/funnyzhou/REFERS): "Generalized Radiograph Representation Learning via Cross-Supervision Between Images and Free-Text Radiology Reports" [[Official]](checkpoints/official) [[Ours]](checkpoints/pretrained)

</details>

> We pre-train 8 MedVLP models on the same training set from [MIMIC-CXR](https://www.physionet.org/content/mimic-cxr-jpg/2.1.0/), and will release our pre-trained checkpoints. Since the labels for training [MedCLIP](https://github.com/RyanWangZf/MedCLIP) are not public avaliable, we use its offitial checkpoints for evaluation.

> For segmentation, please run the scripts in ['preprocess/model_converters'](preprocess/model_converters) to convert keys in MedVLP models to the MMSegmentation format before training.

Our codebase accesses the datasets from `./datasets/` and pre-trained models from `./checkpoints/` by default.
```
├── bin
├── config
├── checkpoints
├── datasets
│   ├── COVIDx
│   ├── ...
│   └── IU_Xray
├── unifier
├── scripts
├── mmsegmentation
├── unifier
```

## Fine-Tuning & Evaluation

We evaluate our pre-trained MedVLP models by specifying the `--pretrain_path` argument before running each downstream task. Arguments can be modified through [`config/`](config/). Additional command-line arguments can also be specified to override the configuration setting.

### 1. Classification

To fine-tune the MedVLP model for classification, run this command:

Here is an example to train FixMatch on CIFAR-100 with 200 labels. Training other supported algorithms (on other datasets with different label settings) can be specified by a config file:

```
python bin/train.py config/classification/$dataset/$method.yml
```

### 2. Segmentation
To fine-tune the MedVLP model for segmentation, run this command:

```
python mmsegmentation/tools/train.py config/benchmark/$dataset/$method.yml
```

> **Adapting MMSegmentation**: We provide the necessary files for adapting MMSegmentation in the directory [unifed_segmentation](unifed_segmentation). After modifying MMSegmentaiton framework with the provided files, start fine-tuning and evaluation with [ft.sh](Siim_Segmentation/ft.sh) and [test.sh](Siim_Segmentation/test.sh), respectively.

### 3. Report Generation
To fine-tune the MedVLP model for report generation, run this command:
```
python bin/train.py config/report_generation/$dataset/$method.yml
```

### 4. Evaluation

```
# For classification and report generation
python bin/train.py config/$task/$dataset/$method.yml

# For segmentation
python mmsegmentation/tools/train.py config/report_generation/$dataset/$method.yml path/to/checkpoint --work-dir experiments/$dataset/$method/
```

## Running a Benchmark Experiment / Reproduce benchmark



## Acknowledgments

We thanks the following projects for reference of creating BenchX:

- [ViLMedic](https://github.com/jbdel/vilmedic)
- [TorchXRayVision](https://github.com/mlmed/torchxrayvision)
- [MRM](https://github.com/RL4M/MRM-pytorch/tree/main)
