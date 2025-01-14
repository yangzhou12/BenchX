# BenchX: A Unified Benchmark Framework for Medical Vision-Language Pretraining on Chest X-Rays

This repository includes the code for NeurIPS24 Datasets and Benchmarks Paper: 

"[BenchX: A Unified Benchmark Framework for Medical Vision-Language Pretraining on Chest X-Rays](https://arxiv.org/abs/2410.21969)"

## Installation
Create a conda environment and activate it:
```bash
conda create -n BenchX python=3.11
conda activate BenchX
```

Clone the repository of BenchX and install requirements:
```bash
git clone https://github.com/yangzhou12/BenchX
cd BenchX
pip install -r requirements.txt
```

Install MMSegmentation following the [offitial instruction](https://mmsegmentation.readthedocs.io/en/latest/get_started.html):
```bash
# Install the dependency for MMSegmentation
pip install -U openmim
mim install mmengine
mim install "mmcv==2.1.0"

# Install MMSegmentation
git clone -b main https://github.com/open-mmlab/mmsegmentation.git
cd mmsegmentation
pip install -v -e .
```

## Dataset Preparation

<details close>
<summary><b>Supported Datasets and Tasks</b> (click to expand)</summary>

* [COVIDx CXR-4](https://www.kaggle.com/datasets/andyczhao/covidx-cxr2) (Binary Classification)
* [NIH Chest X-Rays](https://huggingface.co/datasets/alkzar90/NIH-Chest-X-ray-dataset) (Multi-Label Classification)
* [Object-CXR](https://www.kaggle.com/datasets/raddar/foreign-objects-in-chest-xrays) (Binary Classification, Segmentation)
* [RSNA Pneumonia](https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge) (Binary Classification, Segmentation)
* [SIIM-ACR Pneumothorax Segmentation](https://www.kaggle.com/datasets/vbookshelf/pneumothorax-chest-xray-images-and-masks) (Binary Classification, Segmentation)
* [TBX11K](https://www.kaggle.com/datasets/vbookshelf/tbx11k-simplified) (Segmentation)
* [VinDr-CXR](https://physionet.org/content/vindr-cxr/1.0.0/) (Multi-Label Classification, Segmentation)
* [IU X-Ray](https://drive.google.com/file/d/1c0BXEuDy8Cmm2jfN0YYGkQxFZd2ZIoLg) (Report Generation)

</details>

>  We provide scripts to process existing datasets. Please refer to [`datasets/README.md`](datasets/README.md) for more detail. In particular, we do not distribute datasets in this repository, and we do not own copyrights on any of the datasets.

## Pre-Trained Checkpoints

<details close>
<summary><b>Supported MedVLP methods </b> (click to expand)</summary>

* [ConVIRT](https://github.com/edreisMD/ConVIRT-pytorch/tree/master): "Contrastive Learning of Medical Visual Representations from Paired Images and Text" [[Ours]](https://huggingface.co/youngzhou12/ConVIRT)
* [GLoRIA](https://github.com/marshuang80/gloria/tree/main): "GLoRIA: A Multimodal Global-Local Representation Learning Framework for Label-efficient Medical Image Recognition" [[Official]](https://github.com/marshuang80/gloria/tree/main?tab=readme-ov-file#usage) [[Ours]](https://huggingface.co/youngzhou12/GLoRIA)
* [MedCLIP](https://github.com/RyanWangZf/MedCLIP): "MedCLIP: Contrastive Learning from Unpaired Medical Images and Texts" [[Official]](https://github.com/RyanWangZf/MedCLIP?tab=readme-ov-file#download-medclip)
* [MedKLIP](https://github.com/MediaBrain-SJTU/MedKLIP): "MedKLIP: Medical Knowledge Enhanced Language-Image Pre-Training in Radiology" [[Official]](https://github.com/MediaBrain-SJTU/MedKLIP/tree/main/checkpoints) [[Ours]](https://huggingface.co/youngzhou12/MedKLIP)
* [M-FLAG](https://github.com/cheliu-computation/M-FLAG-MICCAI2023): "M-FLAG: Medical Vision-Language Pre-training with Frozen Language Models and Latent Space Geometry Optimization" [[Ours]](https://huggingface.co/youngzhou12/M-FLAG)
* [MGCA](https://github.com/HKU-MedAI/MGCA/tree/main): "Multi-Granularity Cross-modal Alignment for Generalized Medical Visual Representation Learning" [[Official]](https://github.com/HKU-MedAI/MGCA/tree/main?tab=readme-ov-file#pre-training) [[Ours-ResNet50]](https://huggingface.co/youngzhou12/MGCA-resnet50) [[Ours-ViT]](https://huggingface.co/youngzhou12/MGCA-vit)
* [PTUnifier](https://github.com/zhjohnchan/PTUnifier): "Towards Unifying Medical Vision-and-Language Pre-training via Soft Prompts" [[Ours]](https://huggingface.co/youngzhou12/PTUnifier)
* [MRM](https://github.com/RL4M/MRM-pytorch/tree/main): "Advancing Radiograph Representation Learning with Masked Record Modeling" [[Official]](https://github.com/RL4M/MRM-pytorch/tree/main?tab=readme-ov-file#42-start-fine-tuning-take-1-percent-data-as-the-example) [[Ours]](https://huggingface.co/youngzhou12/MRM)
* [REFERS](https://github.com/funnyzhou/REFERS): "Generalized Radiograph Representation Learning via Cross-Supervision Between Images and Free-Text Radiology Reports" [[Official]](https://github.com/funnyzhou/REFERS?tab=readme-ov-file#training-models) [[Ours]](https://huggingface.co/youngzhou12/REFERS)

</details>

> We pre-trained 8 MedVLP models on the same training set from [MIMIC-CXR](https://www.physionet.org/content/mimic-cxr-jpg/2.1.0/), and release our pre-trained checkpoints. Since the labels for training [MedCLIP](https://github.com/RyanWangZf/MedCLIP) are not publicly avaliable, we use its offitial checkpoints for evaluation.

> For segmentation, please run the scripts in ['preprocess/model_converters'](preprocess/model_converters) to convert keys in MedVLP models to the MMSegmentation format before training.

## Fine-Tuning & Evaluation

You can run any benchmark task supported by BenchX using the following commands. The MedVLP model, dataset, and training parameters can be specified by a config file in [`config/`](config/). Additional command-line arguments can also be specified to override the configuration setting.

### 1. Classification

To fine-tune a MedVLP model for classification, run this command:

```
python bin/train.py configs/classification/<dataset_name>/<model_name>.yml
```

### 2. Segmentation
To fine-tune a MedVLP model for segmentation, run this command:

```
python mmsegmentation/tools/train.py mmsegmentation/configs/benchmark/<dataset_name>/<model_name>.py
```

> **Adapting MMSegmentation**: We provide the necessary files for adapting MMSegmentation in [preprocess/mmsegmentation/](preprocess/mmsegmentation/). Please modify the installed MMSegmentaiton framework in [mmsegmentation/](mmsegmentation/) by adding the provided files before training and evaluation.

### 3. Report Generation
To fine-tune a MedVLP model for report generation, run this command:
```
python bin/train.py configs/report_generation/<dataset_name>/<model_name>.yml
```

### 4. Evaluation
To evaluate fine-tuned MedVLP models, run:

```
# For classification and report generation
python bin/test.py configs/<task_name>/<dataset_name>/<model_name>.yml validator.splits=[test] ckpt_dir=<path_to_checkpoint>

# For segmentation
python mmsegmentation/tools/my_test.py mmsegmentation/configs/<dataset_name>/<model_name>.yml <path_to_checkpoint>
```

## Reproduce Benchmark Results

To reproduce the benchmark results in the paper, run the following scripts:

```
# For binary classification on COVIDx
sh scripts/classification/run_BenchX_COVIDx.sh

# For binary classification on RSNA
sh scripts/classification/run_BenchX_RSNA.sh

# For binary classification on SIIM
sh scripts/classification/run_BenchX_SIIM.sh

# For multi-lable classification on NIH Chest X-Ray
sh scripts/classification/run_BenchX_NIH.sh

# For multi-lable classification on VinDr-CXR
sh scripts/classification/run_BenchX_VinDr.sh

# For segmentation on Object-CXR
sh scripts/segmentation/run_BenchX_Object_CXR.sh

# For segmentation on RSNA
sh scripts/segmentation/run_BenchX_RSNA.sh

# For segmentation on SIIM
sh scripts/segmentation/run_BenchX_SIIM.sh

# For segmentation on TBX11K
sh scripts/segmentation/run_BenchX_TBX11K.sh

# For report generation on IU X-Ray
sh scripts/report_generation/run_BenchX_IU_XRay.sh
```

## Acknowledgments

We thanks the following projects for reference of creating BenchX:

- [ViLMedic](https://github.com/jbdel/vilmedic)
- [TorchXRayVision](https://github.com/mlmed/torchxrayvision)
- [MRM](https://github.com/RL4M/MRM-pytorch/tree/main)

## Citation

If you find BenchX useful for your research and applications, please cite using this BibTeX:

```bibtex
@inproceedings{zhou2024benchx,
  title={BenchX: A Unified Benchmark Framework for Medical Vision-Language Pretraining on Chest X-Rays},
  author={Yang Zhou, Tan Li Hui Faith, Yanyu Xu, Sicong Leng, Xinxing Xu, Yong Liu, Rick Siow Mong Goh},
  booktitle={Proceedings of NeurIPS},
  year={2024}
}
```
