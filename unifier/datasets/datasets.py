import collections
import os
import os.path
import pprint
import random

import re
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Dict, List
from PIL import Image
import skimage.transform
from skimage.io import imread
import torch
# import cv2
from ast import literal_eval

import sys

import unifier
from collections import Counter
from transformers import AutoTokenizer
from unifier.datasets.utils import split_into_sents
from unifier.blocks.custom.r2gen import Tokenizer
import tarfile

default_pathologies = [
    "Atelectasis",
    "Consolidation",
    "Infiltration",
    "Pneumothorax",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Effusion",
    "Pneumonia",
    "Pleural_Thickening",
    "Cardiomegaly",
    "Nodule",
    "Mass",
    "Hernia",
    "Lung Lesion",
    "Fracture",
    "Lung Opacity",
    "Enlarged Cardiomediastinum",
]

# Use a file that ships with the library
USE_INCLUDED_FILE = "USE_INCLUDED_FILE"

thispath = os.path.dirname(os.path.realpath(__file__))
datapath = os.path.join(thispath, "data")

# This is for caching small things for speed
_cache_dict = {}


def apply_transforms(sample, transform, seed=None) -> Dict:
    """Applies transforms to the image and masks.
    The seeds are set so that the transforms that are applied
    to the image are the same that are applied to each mask.
    This way data augmentation will work for segmentation or
    other tasks which use masks information.
    """

    if seed is None:
        MAX_RAND_VAL = 2147483647
        seed = np.random.randint(MAX_RAND_VAL)

    if transform is not None:
        random.seed(seed)
        torch.random.manual_seed(seed)

        if type(sample["img"]) == list: # multi-image
            for i, img in enumerate(sample["img"]):
                sample["img"][i] = transform(img)
        else: # single image
            sample["img"] = transform(sample["img"])

        if "pathology_masks" in sample:
            for i in sample["pathology_masks"].keys():
                random.seed(seed)
                torch.random.manual_seed(seed)
                sample["pathology_masks"][i] = transform(sample["pathology_masks"][i])

        if "semantic_masks" in sample:
            for i in sample["semantic_masks"].keys():
                random.seed(seed)
                torch.random.manual_seed(seed)
                sample["semantic_masks"][i] = transform(sample["semantic_masks"][i])

    return sample


class Dataset:
    """The datasets in this library aim to fit a simple interface where the
    imgpath and csvpath are specified. Some datasets require more than one
    metadata file and for some the metadata files are packaged in the library
    so only the imgpath needs to be specified.
    """

    def __init__(self):
        pass

    pathologies: List[str]
    """A list of strings identifying the pathologies contained in this 
    dataset. This list corresponds to the columns of the `.labels` matrix. 
    Although it is called pathologies, the contents do not have to be 
    pathologies and may simply be attributes of the patient. """

    labels: torch.Tensor
    """A NumPy array which contains a 1, 0, or NaN for each pathology. Each 
    column is a pathology and each row corresponds to an item in the dataset. 
    A 1 represents that the pathology is present, 0 represents the pathology 
    is absent, and NaN represents no information. """

    csv: pd.DataFrame
    """A Pandas DataFrame of the metadata .csv file that is included with the 
    data. For some datasets multiple metadata files have been merged 
    together. It is largely a "catch-all" for associated data and the 
    referenced publication should explain each field. Each row aligns with 
    the elements of the dataset so indexing using .iloc will work. Alignment 
    between the DataFrame and the dataset items will be maintained when using 
    tools from this library. """

    split: str
    """Each dataset class should receive a split parameter containing one of
    three values: train, valid, test. The .csv file will be filtered according
    to the specified split."""

    def totals(self) -> Dict[str, Dict[str, int]]:
        """Compute counts of pathologies.

        Returns: A dict containing pathology name -> (label -> value)
        """
        counts = [
            dict(collections.Counter(items[~np.isnan(items)]).most_common())
            for items in self.labels.T
        ]
        return dict(zip(self.pathologies, counts))

    def __repr__(self) -> str:
        """Returns the name and a description of the dataset such as:

        .. code-block:: python

            CheX_Dataset num_samples=191010 views=['PA', 'AP']

        If in a jupyter notebook it will also print the counts of the
        pathology counts returned by .totals()

        .. code-block:: python

            {'Atelectasis': {0.0: 17621, 1.0: 29718},
             'Cardiomegaly': {0.0: 22645, 1.0: 23384},
             'Consolidation': {0.0: 30463, 1.0: 12982},
             ...}

        """
        if unifier.datasets.utils.in_notebook():
            pprint.pprint(self.totals())
        return self.string()

    def check_paths_exist(self):
        if not os.path.isdir(self.data_path):
            raise Exception("imgpath must be a directory")
        if not os.path.isfile(self.csvpath):
            raise Exception("csvpath must be a file")

    def limit_to_selected_views(self, views):
        """This function is called by subclasses to filter the
        images by view based on the values in .csv['view'].
        """
        if type(views) is not list:
            views = [views]
        if "*" in views:
            # if you have the wildcard, the rest are irrelevant
            views = ["*"]
        self.views = views

        # missing data is unknown
        self.csv.view.fillna("UNKNOWN", inplace=True)

        if "*" not in views:
            self.csv = self.csv[self.csv["view"].isin(self.views)]  # Select the view

class MIMIC_Dataset(Dataset):
    """MIMIC-CXR Dataset

    Citation:

    Johnson AE, Pollard TJ, Berkowitz S, Greenbaum NR, Lungren MP, Deng CY,
    Mark RG, Horng S. MIMIC-CXR: A large publicly available database of
    labeled chest radiographs. arXiv preprint arXiv:1901.07042. 2019 Jan 21.

    https://arxiv.org/abs/1901.07042

    Dataset website here:
    https://physionet.org/content/mimic-cxr-jpg/2.0.0/
    """

    def __init__(
        self,
        imgpath,
        metacsvpath,  # metadata file
        csvpath=USE_INCLUDED_FILE,  # reports w/ chexpert file
        views=["PA"],
        transform=None,
        seed=42,
        unique_patients=False,
        split="train",
        data_pct=1,
        section=None,  # impression, findings, or both
        tokenizer=None,
        max_words=112,
        **kwargs,
    ):
        super(MIMIC_Dataset, self).__init__()
        np.random.seed(seed)  # Reset the seed so all runs are the same.

        self.split = split

        self.pathologies = [
            "Enlarged Cardiomediastinum",
            "Cardiomegaly",
            "Lung Opacity",
            "Lung Lesion",
            "Edema",
            "Consolidation",
            "Pneumonia",
            "Atelectasis",
            "Pneumothorax",
            "Pleural Effusion",
            "Pleural Other",
            "Fracture",
            "Support Devices",
        ]

        self.pathologies = sorted(self.pathologies)

        self.imgpath = imgpath
        self.transform = transform

        # Load dataset
        if csvpath == USE_INCLUDED_FILE:
            self.csvpath = os.path.join(datapath, "mimic_labelled_reports.csv.zip")
        else:
            self.csvpath = csvpath

        # Filter out irrelevant sections of report
        if section:
            assert section in ["impression", "findings", "both"]
            save_dir = os.path.join(
                datapath, f"mimic_labelled_reports_{section}.csv.zip"
            )
            if os.path.exists(save_dir):  # Overwrite csvpath if sectioned csv exists
                self.csvpath = save_dir
            else:
                df = pd.read_csv(self.csvpath)
                print("Filtering specified report sections...")
                for i in tqdm(range(len(df))):
                    parsed_report = self.get_report_dict(df.loc[i, "Report"])
                    try:
                        if section == "impression":
                            report_content = parsed_report["impression"]
                        elif section == "findings":
                            report_content = parsed_report["findings"]
                        elif section == "both":
                            report_content = " ".join(
                                [parsed_report["impression"], parsed_report["findings"]]
                            )
                    except KeyError:
                        report_content = ""
                    df.loc[i, "Report"] = report_content
                df = df[
                    df["Report"] != ""
                ]  # filter out rows with empty reports after sectioning
                df.to_csv(save_dir, index=False)  # save csv
                self.csvpath = save_dir  # overwrite csvpath

        self.csv = pd.read_csv(self.csvpath)
        self.metacsvpath = metacsvpath
        self.metacsv = pd.read_csv(self.metacsvpath)

        support_splits = ["train", "val", "test", "train_1", "train_10"]
        assert split in support_splits, f"Invalid split: {split}. Supported splits are {support_splits}"

        self.csv = self.csv[self.csv["split"] == split]  # filter for split

        self.csv = self.csv.set_index(["dicom_id", "study_id", "subject_id"])
        self.metacsv = self.metacsv.set_index(["dicom_id", "study_id", "subject_id"])

        self.csv = self.csv.join(self.metacsv).reset_index()

        # Keep only the desired view
        self.csv["view"] = self.csv["ViewPosition"]
        self.limit_to_selected_views(views)

        if unique_patients:
            self.csv = self.csv.groupby("subject_id").first().reset_index()

        # Get our classes (set all 0's for pathology columns w/ no findings)
        healthy = self.csv["No Finding"] == 1
        labels = []
        for pathology in self.pathologies:
            if pathology in self.csv.columns:
                self.csv.loc[healthy, pathology] = 0
                mask = self.csv[pathology]
            labels.append(torch.FloatTensor(mask.values))

        self.labels = torch.stack(labels).T

        # Make all the -1 values into nans to keep things simple
        self.labels[self.labels == -1] = float('nan')

        # Rename pathologies
        self.pathologies = list(
            np.char.replace(self.pathologies, "Pleural Effusion", "Effusion")
        )

        # offset_day_int
        self.csv["offset_day_int"] = self.csv["StudyDate"]

        # patientid
        self.csv["patientid"] = self.csv["subject_id"].astype(str)

        # Create tokenizer from pretrained
        if tokenizer is not None:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
            self.tokenizer_args = {
                "return_tensors": "pt",
                "padding": True,
                "add_special_tokens": True,
            }
            if max_words is not None:
                self.tokenizer_args.update(
                    {
                        "padding": "max_length",
                        "truncation": True,
                        "max_length": max_words,
                    }
                )

    def string(self):
        return (
            self.__class__.__name__
            + " num_samples={} views={} transforms={} split={}".format(
                len(self), self.views, self.transform, self.split
            )
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = {}
        sample["idx"] = idx
        sample["lab"] = self.labels[idx]

        subjectid = str(self.csv.iloc[idx]["subject_id"])
        studyid = str(self.csv.iloc[idx]["study_id"])
        dicom_id = str(self.csv.iloc[idx]["dicom_id"])

        img_path = os.path.join(
            self.imgpath,
            "p" + subjectid[:2],
            "p" + subjectid,
            "s" + studyid,
            dicom_id + ".jpg",
        )
        img = imread(img_path)
        img = Image.fromarray(img).convert("RGB")

        sample["img"] = img  # remove normalization step
        sample["report"] = str(self.csv.iloc[idx]["Report"])

        sample = apply_transforms(sample, self.transform)

        return sample

    def get_report_dict(self, report):
        matches = list(
            re.compile(r"(?P<title>[ [A-Z\]()]+):").finditer(report)
        )  # Remove carat and newline flag

        report_dict = {}

        for match, next_match in zip(matches, matches[1:] + [None]):
            start = match.end()
            end = next_match and next_match.start()

            title = match.group("title")
            title = title.strip().lower()

            paragraph = report[start:end]
            paragraph = re.sub(r"\s{2,}", " ", paragraph)
            paragraph = paragraph.strip()

            report_dict[title] = paragraph.replace("\n", "\\n")

        return report_dict

    def get_collate_fn(self):
        def collate_fn(batch):  # collate function for multimodal dataset
            seq = self.tokenizer([s["report"] for s in batch], **self.tokenizer_args)
            imgs = [s["img"] for s in batch]
            labels = [s["lab"] for s in batch]
            collated = {
                "input_ids": seq.input_ids,
                "attention_mask": seq.attention_mask,
                "images": torch.stack(imgs),
                "labels": torch.stack(labels)
            }
            return collated

        return collate_fn

class CheX_Dataset(Dataset):
    """CheXpert Dataset

    Citation:

    CheXpert: A Large Chest Radiograph Dataset with Uncertainty Labels and
    Expert Comparison. Jeremy Irvin *, Pranav Rajpurkar *, Michael Ko,
    Yifan Yu, Silviana Ciurea-Ilcus, Chris Chute, Henrik Marklund, Behzad
    Haghgoo, Robyn Ball, Katie Shpanskaya, Jayne Seekins, David A. Mong,
    Safwan S. Halabi, Jesse K. Sandberg, Ricky Jones, David B. Larson,
    Curtis P. Langlotz, Bhavik N. Patel, Matthew P. Lungren, Andrew Y. Ng.
    https://arxiv.org/abs/1901.07031

    Dataset website here:
    https://stanfordmlgroup.github.io/competitions/chexpert/

    A small validation set is provided with the data as well. It is used
    as a test set.
    """

    def __init__(
        self,
        data_path,
        csvpath=USE_INCLUDED_FILE,
        views=["PA", "AP"],
        transform=None,
        seed=42,
        unique_patients=False,
        split="train",
    ):
        super(CheX_Dataset, self).__init__()
        np.random.seed(seed)  # Reset the seed so all runs are the same.

        self.split = split

        self.pathologies = [
            "Enlarged Cardiomediastinum",
            "Cardiomegaly",
            "Lung Opacity",
            "Lung Lesion",
            "Edema",
            "Consolidation",
            "Pneumonia",
            "Atelectasis",
            "Pneumothorax",
            "Pleural Effusion",
            "Pleural Other",
            "Fracture",
            "Support Devices",
        ]

        # # MGCA Version
        # self.pathologies = [
        #     "Atelectasis",
        #     "Cardiomegaly",
        #     "Consolidation",
        #     "Edema",
        #     "Pleural Effusion",
        # ]

        self.pathologies = sorted(self.pathologies)

        self.data_path = data_path
        self.transform = transform
        if csvpath == USE_INCLUDED_FILE:
            self.csvpath = os.path.join(data_path, "chexpert_labels.csv")
        else:
            self.csvpath = csvpath
        self.csv = pd.read_csv(self.csvpath)
        # self.csv = self.csv[self.csv["split"] == split]  # filter for split
        self.views = views

        # self.csv["view"] = self.csv["Frontal/Lateral"]  # Assign view column
        # self.csv.loc[(self.csv["view"] == "Frontal"), "view"] = self.csv[
        #     "AP/PA"
        # ]  # If Frontal change with the corresponding value in the AP/PA column otherwise remains Lateral
        # self.csv["view"] = self.csv["view"].replace(
        #     {"Lateral": "L"}
        # )  # Rename Lateral with L

        # self.limit_to_selected_views(views)

        # if unique_patients:
        #     self.csv["PatientID"] = self.csv["Path"].str.extract(pat=r"(patient\d+)")
        #     self.csv = self.csv.groupby("PatientID").first().reset_index()

        support_splits = ["train", "val", "test", "train_1", "train_10"]
        assert split in support_splits, f"Invalid split: {split}. Supported splits are {support_splits}"

        split_path = os.path.join(data_path, f"{split}.txt")
        # Read the .txt file using pandas
        with open(split_path, 'r') as file:
            data_split = [line.strip() for line in file]

        data_split = pd.Series(data_split)
        self.csv = self.csv[self.csv["Path"].isin(data_split)]

        self.csv = self.csv.reset_index()   

        # Get our classes.
        healthy = self.csv["No Finding"] == 1
        labels = [] # [13, len(dataset)]
        for pathology in self.pathologies:
            if pathology in self.csv.columns:
                if pathology != "Support Devices":
                    self.csv.loc[healthy, pathology] = 0
                mask = self.csv[pathology]
            labels.append(torch.FloatTensor(mask.values))

        self.labels = torch.stack(labels).T

        # Make all the -1 values into nans to keep things simple
        self.labels[self.labels == -1] = float('nan')

        # Set labels with nan to 0
        self.labels[torch.isnan(self.labels)] = 0

        # Rename pathologies
        self.pathologies = list(
            np.char.replace(self.pathologies, "Pleural Effusion", "Effusion")
        )

        # # patientid
        # if self.split == "train" or self.split == "valid":
        #     patientid = self.csv.Path.str.split("train/", expand=True)[1]
        # elif self.split == "test":
        #     patientid = self.csv.Path.str.split("valid/", expand=True)[1]
        # else:
        #     raise NotImplementedError

        # patientid = patientid.str.split("/study", expand=True)[0]
        # patientid = patientid.str.replace("patient", "")

        # # patientid
        # self.csv["patientid"] = patientid

        # age
        self.csv["age_years"] = self.csv["Age"] * 1.0
        self.csv.loc[self.csv["Age"] == 0, "Age"] = None

        # sex
        self.csv["sex_male"] = self.csv["Sex"] == "Male"
        self.csv["sex_female"] = self.csv["Sex"] == "Female"     

    def string(self):
        return (
            self.__class__.__name__
            + " num_samples={} views={} transforms={} split={}".format(
                len(self), self.views, self.transform, self.split
            )
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = {}
        sample["idx"] = idx
        sample["lab"] = self.labels[idx]

        imgid = self.csv["Path"].iloc[idx]
        # clean up path in csv so the user can specify the path
        imgid = imgid.replace("CheXpert-v1.0-small/", "")
        img_path = os.path.join(self.data_path, imgid)
        # img = imread(img_path)
        # img = Image.fromarray(img).convert("RGB")
        img = Image.open(img_path).convert("RGB")

        sample["img"] = img

        sample = apply_transforms(sample, self.transform)

        return sample

    def get_collate_fn(self):
        def collate_fn(batch):
            imgs = [s["img"] for s in batch]
            labels = [s["lab"] for s in batch]
            collated = {
                "labels": torch.stack(labels),
                "images": torch.stack(imgs),
            }
            return collated

        return collate_fn

class CheX5_Dataset(Dataset):
    """CheXpert Dataset

    Citation:

    CheXpert: A Large Chest Radiograph Dataset with Uncertainty Labels and
    Expert Comparison. Jeremy Irvin *, Pranav Rajpurkar *, Michael Ko,
    Yifan Yu, Silviana Ciurea-Ilcus, Chris Chute, Henrik Marklund, Behzad
    Haghgoo, Robyn Ball, Katie Shpanskaya, Jayne Seekins, David A. Mong,
    Safwan S. Halabi, Jesse K. Sandberg, Ricky Jones, David B. Larson,
    Curtis P. Langlotz, Bhavik N. Patel, Matthew P. Lungren, Andrew Y. Ng.
    https://arxiv.org/abs/1901.07031

    Dataset website here:
    https://stanfordmlgroup.github.io/competitions/chexpert/

    A small validation set is provided with the data as well. It is used
    as a test set.
    """

    def __init__(
        self,
        data_path,
        csvpath=USE_INCLUDED_FILE,
        views=["PA", "AP"],
        transform=None,
        seed=42,
        unique_patients=False,
        split="train",
    ):
        super(CheX5_Dataset, self).__init__()
        np.random.seed(seed)  # Reset the seed so all runs are the same.

        self.split = split

        # MGCA Version
        self.pathologies = [
            "Atelectasis",
            "Cardiomegaly",
            "Consolidation",
            "Edema",
            "Pleural Effusion",
        ]

        self.pathologies = sorted(self.pathologies)

        self.data_path = data_path
        self.transform = transform
        if csvpath == USE_INCLUDED_FILE:
            self.csvpath = os.path.join(data_path, "chexpert_labels.csv")
        else:
            self.csvpath = csvpath
        self.csv = pd.read_csv(self.csvpath)
        # self.csv = self.csv[self.csv["split"] == split]  # filter for split
        self.views = views

        support_splits = ["train", "val", "test", "train_1", "train_10"]
        assert split in support_splits, f"Invalid split: {split}. Supported splits are {support_splits}"

        split_path = os.path.join(data_path, f"{split}.txt")
        # Read the .txt file using pandas
        with open(split_path, 'r') as file:
            data_split = [line.strip() for line in file]

        data_split = pd.Series(data_split)
        self.csv = self.csv[self.csv["Path"].isin(data_split)]

        self.csv = self.csv.reset_index()   

        # Get our classes.
        healthy = self.csv["No Finding"] == 1
        labels = [] # [13, len(dataset)]
        for pathology in self.pathologies:
            if pathology in self.csv.columns:
                if pathology != "Support Devices":
                    self.csv.loc[healthy, pathology] = 0
                mask = self.csv[pathology]
            labels.append(torch.FloatTensor(mask.values))

        self.labels = torch.stack(labels).T

        # Make all the -1 values into nans to keep things simple
        self.labels[self.labels == -1] = float('nan')

        # Set labels with nan to 0
        self.labels[torch.isnan(self.labels)] = 0

        # Rename pathologies
        self.pathologies = list(
            np.char.replace(self.pathologies, "Pleural Effusion", "Effusion")
        )

        # age
        self.csv["age_years"] = self.csv["Age"] * 1.0
        self.csv.loc[self.csv["Age"] == 0, "Age"] = None

        # sex
        self.csv["sex_male"] = self.csv["Sex"] == "Male"
        self.csv["sex_female"] = self.csv["Sex"] == "Female"     

    def string(self):
        return (
            self.__class__.__name__
            + " num_samples={} views={} transforms={} split={}".format(
                len(self), self.views, self.transform, self.split
            )
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = {}
        sample["idx"] = idx
        sample["lab"] = self.labels[idx]

        imgid = self.csv["Path"].iloc[idx]
        # clean up path in csv so the user can specify the path
        imgid = imgid.replace("CheXpert-v1.0-small/", "")
        img_path = os.path.join(self.data_path, imgid)
        img = Image.open(img_path).convert("RGB")

        sample["img"] = img

        sample = apply_transforms(sample, self.transform)

        return sample

    def get_collate_fn(self):
        def collate_fn(batch):
            imgs = [s["img"] for s in batch]
            labels = [s["lab"] for s in batch]
            collated = {
                "labels": torch.stack(labels),
                "images": torch.stack(imgs),
            }
            return collated

        return collate_fn

class PC_Dataset(Dataset):
    """PadChest dataset from the Hospital San Juan de Alicante - University of
    Alicante

    Note that images with null labels (as opposed to normal), and images that
    cannot be properly loaded (listed as 'missing' in the code) are excluded,
    which makes the total number of available images slightly less than the
    total number of image files.

    Citation:

    PadChest: A large chest x-ray image dataset with multi-label annotated
    reports. Aurelia Bustos, Antonio Pertusa, Jose-Maria Salinas, and Maria
    de la Iglesia-Vayá. arXiv preprint, 2019. https://arxiv.org/abs/1901.07441

    Dataset website:
    http://bimcv.cipf.es/bimcv-projects/padchest/

    Download full size images here:
    https://academictorrents.com/details/dec12db21d57e158f78621f06dcbe78248d14850

    Download resized (224x224) images here (recropped):
    https://academictorrents.com/details/96ebb4f92b85929eadfb16761f310a6d04105797
    """

    def __init__(self,
                 data_path,
                 csvpath=USE_INCLUDED_FILE,
                 views=["PA"],
                 transform=None,
                 data_aug=None,
                 flat_dir=True,
                 seed=42,
                 unique_patients=True,
                 split="train",
                ):

        super(PC_Dataset, self).__init__()
        np.random.seed(seed)  # Reset the seed so all runs are the same.

        self.pathologies = ["Atelectasis", "Consolidation", "Infiltration",
                            "Pneumothorax", "Edema", "Emphysema", "Fibrosis",
                            "Effusion", "Pneumonia", "Pleural_Thickening",
                            "Cardiomegaly", "Nodule", "Mass", "Hernia", "Fracture",
                            "Granuloma", "Flattened Diaphragm", "Bronchiectasis",
                            "Aortic Elongation", "Scoliosis",
                            "Hilar Enlargement", "Tuberculosis",
                            "Air Trapping", "Costophrenic Angle Blunting", "Aortic Atheromatosis",
                            "Hemidiaphragm Elevation",
                            "Support Devices", "Tube'"]  # the Tube' is intentional

        self.pathologies = sorted(self.pathologies)
        self.views = views

        mapping = dict()

        mapping["Infiltration"] = ["infiltrates",
                                   "interstitial pattern",
                                   "ground glass pattern",
                                   "reticular interstitial pattern",
                                   "reticulonodular interstitial pattern",
                                   "alveolar pattern",
                                   "consolidation",
                                   "air bronchogram"]
        mapping["Pleural_Thickening"] = ["pleural thickening"]
        mapping["Consolidation"] = ["air bronchogram"]
        mapping["Hilar Enlargement"] = ["adenopathy",
                                        "pulmonary artery enlargement"]
        mapping["Support Devices"] = ["device",
                                      "pacemaker"]
        mapping["Tube'"] = ["stent'"]  # the ' is to select findings which end in that word

        self.data_path = data_path
        self.transform = transform
        self.data_aug = data_aug
        self.flat_dir = flat_dir

        if csvpath == USE_INCLUDED_FILE:
            self.csvpath = os.path.join(datapath, "padchest_labels.csv")
        else:
            self.csvpath = csvpath

        self.check_paths_exist()
        self.csv = pd.read_csv(self.csvpath, low_memory=False)

        # # Standardize view names
        # self.csv.loc[self.csv["Projection"].isin(["AP_horizontal"]), "Projection"] = "AP Supine"

        # self.csv["view"] = self.csv['Projection']
        # self.limit_to_selected_views(views)

        # # Remove null stuff
        # self.csv = self.csv[~self.csv["Labels"].isnull()]

        # # Remove missing files
        # missing = ["216840111366964012819207061112010307142602253_04-014-084.png",
        #            "216840111366964012989926673512011074122523403_00-163-058.png",
        #            "216840111366964012959786098432011033083840143_00-176-115.png",
        #            "216840111366964012558082906712009327122220177_00-102-064.png",
        #            "216840111366964012339356563862009072111404053_00-043-192.png",
        #            "216840111366964013076187734852011291090445391_00-196-188.png",
        #            "216840111366964012373310883942009117084022290_00-064-025.png",
        #            "216840111366964012283393834152009033102258826_00-059-087.png",
        #            "216840111366964012373310883942009170084120009_00-097-074.png",
        #            "216840111366964012819207061112010315104455352_04-024-184.png",
        #            "216840111366964012819207061112010306085429121_04-020-102.png",
        #            "216840111366964012989926673512011083134050913_00-168-009.png",  # broken PNG file (chunk b'\x00\x00\x00\x00')
        #            "216840111366964012373310883942009152114636712_00-102-045.png",  # "OSError: image file is truncated"
        #            "216840111366964012819207061112010281134410801_00-129-131.png",  # "OSError: image file is truncated"
        #            "216840111366964012487858717522009280135853083_00-075-001.png",  # "OSError: image file is truncated"
        #            "216840111366964012989926673512011151082430686_00-157-045.png",  # broken PNG file (chunk b'\x00\x00\x00\x00')
        #            "216840111366964013686042548532013208193054515_02-026-007.png",  # "OSError: image file is truncated"
        #            "216840111366964013590140476722013058110301622_02-056-111.png",  # "OSError: image file is truncated"
        #            "216840111366964013590140476722013043111952381_02-065-198.png",  # "OSError: image file is truncated"
        #            "216840111366964013829543166512013353113303615_02-092-190.png",  # "OSError: image file is truncated"
        #            "216840111366964013962490064942014134093945580_01-178-104.png",  # "OSError: image file is truncated"
        #            ]
        # self.csv = self.csv[~self.csv["ImageID"].isin(missing)]

        # if unique_patients:
        #     self.csv = self.csv.groupby("PatientID").first().reset_index()

        # # Filter out age < 10 (paper published 2019)
        # self.csv = self.csv[(2019 - self.csv.PatientBirth > 10)]

        support_splits = ["train", "val", "test", "train_1", "train_10"]
        assert split in support_splits, f"Invalid split: {split}. Supported splits are {support_splits}"

        split_path = os.path.join(data_path, f"{split}.txt")
        # Read the .txt file using pandas
        with open(split_path, 'r') as file:
            data_split = [line.strip() for line in file]

        data_split = pd.Series(data_split)
        self.csv = self.csv[self.csv["ImageID"].isin(data_split)]

        self.csv = self.csv.reset_index()    

        # Get our classes.
        labels = []
        for pathology in self.pathologies:
            mask = self.csv["Labels"].str.contains(pathology.lower())
            if pathology in mapping:
                for syn in mapping[pathology]:
                    # print("mapping", syn)
                    mask |= self.csv["Labels"].str.contains(syn.lower())
            labels.append(torch.FloatTensor(mask.values))
        
        self.labels = torch.stack(labels).T

        self.pathologies[self.pathologies.index("Tube'")] = "Tube"

        # Add consistent csv values

        # offset_day_int
        dt = pd.to_datetime(self.csv["StudyDate_DICOM"], format="%Y%m%d")
        self.csv["offset_day_int"] = dt.astype(np.int64) // 10**9 // 86400

        # patientid
        self.csv["patientid"] = self.csv["PatientID"].astype(str)

        # age
        self.csv['age_years'] = (2017 - self.csv['PatientBirth'])

        # sex
        self.csv['sex_male'] = self.csv['PatientSex_DICOM'] == 'M'
        self.csv['sex_female'] = self.csv['PatientSex_DICOM'] == 'F'

    def string(self):
        return self.__class__.__name__ + " num_samples={} views={} data_aug={}".format(len(self), self.views, self.data_aug)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = {}
        sample["idx"] = idx
        sample["lab"] = self.labels[idx]

        imgid = self.csv['ImageID'].iloc[idx]
        img_path = os.path.join(self.data_path, "images", imgid)
        
        img = Image.open(img_path).convert("RGB")
        sample["img"] = img

        sample = apply_transforms(sample, self.transform)
        sample = apply_transforms(sample, self.data_aug)

        return sample
    
    def get_collate_fn(self):
        def collate_fn(batch):
            imgs = [s["img"] for s in batch]
            labels = [s["lab"] for s in batch]
            collated = {
                "labels": torch.stack(labels),
                "images": torch.stack(imgs),
            }
            return collated

        return collate_fn
    
class OpenI_Dataset(Dataset):
    """OpenI Dataset

    Dina Demner-Fushman, Marc D. Kohli, Marc B. Rosenman, Sonya E. Shooshan,
    Laritza Rodriguez, Sameer Antani, George R. Thoma, and Clement J.
    McDonald. Preparing a collection of radiology examinations for
    distribution and retrieval. Journal of the American Medical Informatics
    Association, 2016. doi: 10.1093/jamia/ocv080.

    Views have been determined by projection using T-SNE.  To use the T-SNE
    view rather than the view defined by the record,
    set use_tsne_derived_view to true.

    Dataset website:
    https://openi.nlm.nih.gov/faq
    """

    def __init__(self, data_path,
                 csvpath=USE_INCLUDED_FILE,
                 transform=None,
                 data_aug=None,
                 seed=42,
                 unique_patients=True,
                 split="train",
                ):

        super(OpenI_Dataset, self).__init__()
        np.random.seed(seed)  # Reset the seed so all runs are the same.
        self.data_path = data_path
        self.transform = transform
        self.data_aug = data_aug

        self.pathologies = ["Atelectasis", "Fibrosis",
                            "Pneumonia", "Effusion", "Lesion",
                            "Cardiomegaly", "Calcified Granuloma",
                            "Fracture", "Edema", "Granuloma", "Emphysema",
                            "Hernia", "Mass", "Nodule", "Opacity", "Infiltration",
                            "Pleural_Thickening", "Pneumothorax", ]

        self.pathologies = sorted(self.pathologies)

        mapping = dict()
        mapping["Pleural_Thickening"] = ["pleural thickening"]
        mapping["Infiltration"] = ["Infiltrate"]
        mapping["Atelectasis"] = ["Atelectases"]

        # Load data
        
        if csvpath == USE_INCLUDED_FILE:
            self.csvpath = os.path.join(data_path, "openi_labels.csv")
        else:
            self.csvpath = csvpath

        self.csv = pd.read_csv(self.csvpath)

        # tarf = tarfile.open(self.xmlpath, 'r:gz')

        # # self.projcsv_path = os.path.join(datapath, "indiana_projections.csv")
        # # projcsv = pd.read_csv(self.projcsv_path)
        # # self.csv = self.csv.join(tsne_pa, on="imageid")

        # samples = []

        # import xml
        # for filename in tarf.getnames():
        #     if (filename.endswith(".xml")):
        #         tree = xml.etree.ElementTree.parse(tarf.extractfile(filename))
        #         root = tree.getroot()
        #         uid = root.find("uId").attrib["id"]
        #         labels_m = [node.text.lower() for node in root.findall(".//MeSH/major")]
        #         labels_m = "|".join(np.unique(labels_m))
        #         labels_a = [node.text.lower() for node in root.findall(".//MeSH/automatic")]
        #         labels_a = "|".join(np.unique(labels_a))
        #         image_nodes = root.findall(".//parentImage")
        #         for image in image_nodes:
        #             sample = {}
        #             sample["uid"] = uid
        #             sample["imageid"] = image.attrib["id"]
        #             sample["labels_major"] = labels_m
        #             sample["labels_automatic"] = labels_a
        #             samples.append(sample)

        # self.csv = pd.DataFrame(samples)

        # if dicomcsv_path == USE_INCLUDED_FILE:
        #     self.dicomcsv_path = os.path.join(datapath, "nlmcxr_dicom_metadata.csv.gz")
        # else:
        #     self.dicomcsv_path = dicomcsv_path

        # self.dicom_metadata = pd.read_csv(self.dicomcsv_path, index_col="imageid", low_memory=False)

        # # Merge in dicom metadata
        # self.csv = self.csv.join(self.dicom_metadata, on="imageid")

        # if tsnepacsv_path == USE_INCLUDED_FILE:
        #     self.tsnepacsv_path = os.path.join(datapath, "nlmcxr_tsne_pa.csv.gz")
        # else:
        #     self.tsnepacsv_path = tsnepacsv_path

        # # Attach view computed by tsne
        # tsne_pa = pd.read_csv(self.tsnepacsv_path, index_col="imageid")
        # self.csv = self.csv.join(tsne_pa, on="imageid")

        # if use_tsne_derived_view:
        #     self.csv["view"] = self.csv["tsne-view"]
        # else:
        #     self.csv["view"] = self.csv["View Position"]

        # self.limit_to_selected_views(views)

        # if unique_patients:
        #     self.csv = self.csv.groupby("uid").first().reset_index()

        support_splits = ["train", "val", "test", "train_1", "train_10"]
        assert split in support_splits, f"Invalid split: {split}. Supported splits are {support_splits}"

        split_path = os.path.join(data_path, f"{split}.txt")
        # Read the .txt file using pandas
        with open(split_path, 'r') as file:
            data_split = [line.strip() for line in file]

        data_split = pd.Series(data_split)
        self.csv = self.csv[self.csv["imageid"].isin(data_split)]

        self.csv = self.csv.reset_index()   

        # # Get our classes.
        # labels = []
        # for pathology in self.pathologies:
        #     mask = self.csv["labels_automatic"].str.contains(pathology.lower())
        #     if pathology in mapping:
        #         for syn in mapping[pathology]:
        #             # print("mapping", syn)
        #             mask |= self.csv["labels_automatic"].str.contains(syn.lower())
        #     labels.append(torch.FloatTensor(mask.values))

        # self.labels = torch.stack(labels).T
        self.labels = torch.FloatTensor([eval(label) for label in self.csv["labels"].tolist()])

        # Rename pathologies
        self.pathologies = list(np.char.replace(self.pathologies, "Opacity", "Lung Opacity"))
        self.pathologies = list(np.char.replace(self.pathologies, "Lesion", "Lung Lesion"))

        # add consistent csv values

        # offset_day_int
        # self.csv["offset_day_int"] =

        # patientid
        self.csv["patientid"] = self.csv["uid"].astype(str)

    def string(self):
        return self.__class__.__name__ + " num_samples={}".format(len(self))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = {}
        sample["idx"] = idx
        sample["lab"] = self.labels[idx]

        imageid = self.csv.iloc[idx].imageid
        img_path = os.path.join(self.data_path, "images", imageid + ".png")

        # img = imread(img_path)
        # sample["img"] = normalize(img, maxval=255, reshape=True)

        img = Image.open(img_path).convert("RGB")
        sample["img"] = img

        sample = apply_transforms(sample, self.transform)
        sample = apply_transforms(sample, self.data_aug)

        return sample
    
    def get_collate_fn(self):
        def collate_fn(batch):
            imgs = [s["img"] for s in batch]
            labels = [s["lab"] for s in batch]
            collated = {
                "labels": torch.stack(labels),
                "images": torch.stack(imgs),
            }
            return collated

        return collate_fn

class COVID19_Dataset(Dataset):
    """COVID-19 Image Data Collection

    This dataset currently contains hundreds of frontal view X-rays and is
    the largest public resource for COVID-19 image and prognostic data,
    making it a necessary resource to develop and evaluate tools to aid in
    the treatment of COVID-19. It was manually aggregated from publication
    figures as well as various web based repositories into a machine learning
    (ML) friendly format with accompanying dataloader code. We collected
    frontal and lateral view imagery and metadata such as the time since
    first symptoms, intensive care unit (ICU) status, survival status,
    intubation status, or hospital location. We present multiple possible use
    cases for the data such as predicting the need for the ICU, predicting
    patient survival, and understanding a patient's trajectory during
    treatment.

    Citations:

    COVID-19 Image Data Collection: Prospective Predictions Are the Future
    Joseph Paul Cohen and Paul Morrison and Lan Dao and Karsten Roth and Tim
    Q Duong and Marzyeh Ghassemi arXiv:2006.11988, 2020

    COVID-19 image data collection,
    Joseph Paul Cohen and Paul Morrison and Lan Dao
    arXiv:2003.11597, 2020

    Dataset: https://github.com/ieee8023/covid-chestxray-dataset

    Paper: https://arxiv.org/abs/2003.11597
    """

    dataset_url = "https://github.com/ieee8023/covid-chestxray-dataset"

    def __init__(self,
                 data_path: str,
                 csvpath: str,
                 views=["PA", "AP"],
                 transform=None,
                 data_aug=None,
                 seed: int = 0,
                 split="train",
                 extension=".png"
                ):
        """
        Args:
            imgpath: Path to the directory containing images
            csvpath: Path to the image directory
        """

        super(COVID19_Dataset, self).__init__()
        np.random.seed(seed)  # Reset the seed so all runs are the same.
        self.data_path = data_path
        self.transform = transform
        self.data_aug = data_aug
        self.views = views

        if not os.path.exists(csvpath):
            raise FileNotFoundError(f'The csvpath does not point to a valid metadata.csv file. Please download it from {self.dataset_url}')

        # Load data
        self.csvpath = csvpath
        self.csv = pd.read_csv(self.csvpath)

        # Keep only the selected views.
        self.limit_to_selected_views(views)

        # Filter out in progress samples
        self.csv = self.csv[~(self.csv.finding == "todo")]
        self.csv = self.csv[~(self.csv.finding == "Unknown")]

        self.pathologies = self.csv.finding.str.split("/", expand=True).values.ravel()
        self.pathologies = self.pathologies[~pd.isnull(self.pathologies)]
        self.pathologies = sorted(np.unique(self.pathologies))

        support_splits = ["train", "val", "test", "train_1", "train_10"]
        assert split in support_splits, f"Invalid split: {split}. Supported splits are {support_splits}"

        split_path = os.path.join(data_path, f"{split}.txt")
        # Read the .txt file using pandas
        with open(split_path, 'r') as file:
            data_split = [line.strip() for line in file]

        data_split = pd.Series(data_split).apply(lambda x: x + extension)
        self.csv = self.csv[self.csv["filename"].isin(data_split)]

        self.csv = self.csv.reset_index()   

        labels = []
        for pathology in self.pathologies:
            mask = self.csv["finding"].str.contains(pathology)
            labels.append(torch.FloatTensor(mask.values))

        self.labels = torch.stack(labels).T

        self.csv = self.csv.reset_index()

        # offset_day_int
        self.csv["offset_day_int"] = self.csv["offset"]

    def string(self):
        return self.__class__.__name__ + " num_samples={} views={} data_aug={}".format(len(self), self.views, self.data_aug)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = {}
        sample["idx"] = idx
        sample["lab"] = self.labels[idx]

        imgid = self.csv['filename'].iloc[idx]
        img_path = os.path.join(self.data_path, "images", imgid)

        # img = imread(img_path)
        # sample["img"] = normalize(img, maxval=255, reshape=True)

        img = Image.open(img_path).convert("RGB")
        sample["img"] = img

        # if self.semantic_masks:
        #     sample["semantic_masks"] = self.get_semantic_mask_dict(imgid, sample["img"].shape)

        sample = apply_transforms(sample, self.transform)
        sample = apply_transforms(sample, self.data_aug)

        return sample
    
    def get_collate_fn(self):
        def collate_fn(batch):
            imgs = [s["img"] for s in batch]
            labels = [s["lab"] for s in batch]
            collated = {
                "labels": torch.stack(labels),
                "images": torch.stack(imgs),
            }                
            return collated
        
        return collate_fn

class ObjectCXR_Dataset(Dataset):
    """ObjectCXR Dataset

    "We provide a large dataset of chest X-rays with strong annotations of
    foreign objects, and the competition for automatic detection of foreign
    objects. Specifically, 5000 frontal chest X-ray images with foreign
    objects presented and 5000 frontal chest X-ray images without foreign
    objects are provided. All the chest X-ray images were filmed in township
    hospitals in China and collected through our telemedicine platform.
    Foreign objects within the lung field of each chest X-ray are annotated
    with bounding boxes, ellipses or masks depending on the shape of the
    objects."

    Challenge dataset from MIDL2020

    https://jfhealthcare.github.io/object-CXR/

    https://academictorrents.com/details/fdc91f11d7010f7259a05403fc9d00079a09f5d5
    """

    def __init__(self,
                 data_path,
                 csvpath,
                 transform=None,
                 data_aug=None,
                 seed=42,
                 pathology_masks=False,
                 extension=".jpg",
                 split="train",
                ):
        super(ObjectCXR_Dataset, self).__init__()

        np.random.seed(seed)  # Reset the seed so all runs are the same.
        self.data_path = data_path
        self.csvpath = csvpath
        self.transform = transform
        self.data_aug = data_aug
        self.pathology_masks = pathology_masks
        self.views = []
        self.pathologies = ['Foreign Object']

        # Load data
        self.csv = pd.read_csv(self.csvpath)

        labels = []
        labels.append(torch.FloatTensor(~self.csv["annotation"].isnull()))

        self.labels = torch.stack(labels).T

        support_splits = ["train", "val", "test", "train_1", "train_10"]
        assert split in support_splits, f"Invalid split: {split}. Supported splits are {support_splits}"

        split_path = os.path.join(imgpath, f"{split}.txt")
        # Read the .txt file using pandas
        with open(split_path, 'r') as file:
            data_split = [line.strip() for line in file]

        data_split = pd.Series(data_split).apply(lambda x: x + extension)
        self.csv = self.csv[self.csv["image_name"].isin(data_split)]

        self.csv = self.csv.reset_index()

        self.csv["has_masks"] = ~self.csv["annotation"].isnull()

        # self.imgzip = zipfile.ZipFile(self.imgzippath)

    def string(self):
        return self.__class__.__name__ + " num_samples={} views={} data_aug={}".format(len(self), self.views, self.data_aug)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        sample = {}
        sample["idx"] = idx
        sample["lab"] = self.labels[idx]
        image_name = self.csv.iloc[idx]["image_name"]

        img_path = os.path.join(self.data_path, "images", image_name)
        img = Image.open(img_path).convert("RGB")
        sample["img"] = img

        if self.pathology_masks:
            sample["pathology_masks"] = self.get_mask_dict(image_name)

        sample = apply_transforms(sample, self.transform)
        sample = apply_transforms(sample, self.data_aug)

        return sample

    def get_mask_dict(self, image_name):
        path_mask = {}
        for i, pathology in enumerate(self.pathologies):
            mask_path = os.path.join(self.data_path, "images", image_name)
            mask = Image.open(mask_path)
            path_mask[i] = mask[None, :, :]
        return path_mask

    # def get_mask_dict(self, idx, this_size):
    #     h, w = this_size

    #     path_mask = {}
    #     row = self.csv.iloc[idx]
    #     annotation = row.annotation
    #     for i, pathology in enumerate(self.pathologies):
    #         mask = np.zeros([w, h], dtype=np.uint8)
    #         if row.has_masks:
    #             for anno in annotation.split(";"):
    #                 anno = np.fromstring(anno, sep=" ", dtype=int)
    #                 anno_type = anno[0]
    #                 anno = anno[1:]
    #                 if anno_type != 2:
    #                     # bbox annotation (xywh)
    #                     cv2.rectangle(mask, (anno[0], anno[1]), (anno[2], anno[3]), color=1, thickness=cv2.FILLED)
    #                     # mask[anno[1]:anno[3], anno[0]:anno[2]] = 1
    #                 else:
    #                     # polygon annotation 
    #                     poly = anno.reshape((-1,2))
    #                     cv2.fillPoly(mask, [poly], color=1)
    #         path_mask[i] = mask[None, :, :]

    #     return path_mask
    
    def get_collate_fn(self):
        def collate_fn(batch):
            imgs = [s["img"] for s in batch]
            labels = [s["lab"] for s in batch]
            collated = {
                "labels": torch.stack(labels),
                "images": torch.stack(imgs),
            }
            if self.pathology_masks:
                for i, s in enumerate(batch):
                    if i == 0:  # initialize dict of empty lists with same keys
                        masks = dict.fromkeys(s["pathology_masks"], [])
                    for k, v in s["pathology_masks"].items():
                        masks[k].append(v)
                masks = {k: torch.stack(v) for k, v in masks.items()}
                collated.update({"pathology_masks": masks})
            return collated

        return collate_fn

class ChestXDet_Dataset(Dataset):
    """ChestX-Det10-Dataset

    "We select 3,543 images from NIH ChestX-14 and invite three board-certified 
    radiologists to annotate them with 10 common categories of diseases or abnormalities. 
    The 10 categories are Atelectasis, Calcification, Consolidation, Effusion, 
    Emphysema, Fibrosis, Fracture, Mass, Nodule, Pneumothorax."

    Dataset: https://github.com/Deepwise-AILab/ChestX-Det10-Dataset
    """

    def __init__(self,
                 imgpath,
                 csvpath,
                 transform=None,
                 data_aug=None,
                 seed=42,
                 pathology_masks=True,
                 split="train",
                 data_pct=1
                ):
        super(ChestXDet_Dataset, self).__init__()

        np.random.seed(seed)  # Reset the seed so all runs are the same.
        self.imgpath = imgpath
        self.csvpath = csvpath
        self.transform = transform
        self.data_aug = data_aug
        self.pathology_masks = pathology_masks
        self.pathologies = ["Atelectasis", "Calcification", "Consolidation", "Effusion", 
                            "Emphysema", "Fibrosis", "Fracture", "Mass", "Nodule", "Pneumothorax"]

        # Load data
        self.csv = pd.read_csv(self.csvpath)

        labels = []
        for pathology in self.pathologies:
            mask = self.csv["syms"].str.lower().str.contains(pathology.lower())
            labels.append(torch.FloatTensor(mask.values))
        self.labels = torch.stack(labels).T

        self.csv["has_masks"] = self.csv["boxes"].map(lambda x: bool(literal_eval(x)))

    def string(self):
        return self.__class__.__name__ + " num_samples={} data_aug={}".format(len(self), self.data_aug)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        sample = {}
        sample["idx"] = idx
        sample["lab"] = self.labels[idx]
        imgid = self.csv.iloc[idx]["file_name"]

        img_path = os.path.join(self.imgpath, imgid)
        img = Image.open(img_path).convert("RGB")
        sample["img"] = img

        if self.pathology_masks:
            sample["pathology_masks"] = self.get_mask_dict(idx, sample["img"].size)

        sample = apply_transforms(sample, self.transform)
        sample = apply_transforms(sample, self.data_aug)

        return sample

    def get_mask_dict(self, idx, this_size):
        h, w = this_size

        row = self.csv.iloc[idx]
        annotation = literal_eval(row.boxes)
        pathologies = literal_eval(row.syms)
        assert len(annotation) == len(pathologies)
        path_mask = {self.pathologies.index(pathology): np.zeros([1, w, h], dtype=np.uint8) for pathology in pathologies}
        
        for i, pathology in enumerate(pathologies):
            anno = annotation[i]
            pathology_idx = self.pathologies.index(pathology)
            # cv2.rectangle(mask, (anno[0], anno[1]), (anno[2], anno[3]), color=1, thickness=cv2.FILLED)
            path_mask[pathology_idx][:, anno[1]:anno[3], anno[0]:anno[2]] = 1
        return path_mask
    
    def get_collate_fn(self):
        def collate_fn(batch):
            imgs = [s["img"] for s in batch]
            labels = [s["lab"] for s in batch]
            collated = {
                "labels": torch.stack(labels),
                "images": torch.stack(imgs),
            }
            # TODO: Double check how to group a batch of masks
            if self.pathology_masks:
                for i, s in enumerate(batch):
                    if i == 0:  # initialize dict of empty lists with same keys
                        masks = dict.fromkeys(s["pathology_masks"], [])
                    for k, v in s["pathology_masks"].items():
                        masks[k].append(v)
                masks = {k: torch.stack(v) for k, v in masks.items()}
                collated.update({"pathology_masks": masks})
            return collated

        return collate_fn

class RSNA_Pneumonia_Dataset(Dataset):
    """RSNA Pneumonia Detection Challenge

    Citation:

    Augmenting the National Institutes of Health Chest Radiograph Dataset
    with Expert Annotations of Possible Pneumonia. Shih, George, Wu,
    Carol C., Halabi, Safwan S., Kohli, Marc D., Prevedello, Luciano M.,
    Cook, Tessa S., Sharma, Arjun, Amorosa, Judith K., Arteaga, Veronica,
    Galperin-Aizenberg, Maya, Gill, Ritu R., Godoy, Myrna C.B., Hobbs,
    Stephen, Jeudy, Jean, Laroia, Archana, Shah, Palmi N., Vummidi, Dharshan,
    Yaddanapudi, Kavitha, and Stein, Anouk. Radiology: Artificial
    Intelligence, 1 2019. doi: 10.1148/ryai.2019180041.

    More info: https://www.rsna.org/en/education/ai-resources-and-training/ai-image-challenge/RSNA-Pneumonia-Detection-Challenge-2018

    Challenge site:
    https://www.kaggle.com/c/rsna-pneumonia-detection-challenge

    JPG files stored here:
    https://academictorrents.com/details/95588a735c9ae4d123f3ca408e56570409bcf2a9
    """

    def __init__(
        self,
        data_path,
        csvpath=USE_INCLUDED_FILE,
        transform=None,
        seed=42,
        pathology_masks=False,
        extension=".jpg",
        split="train",
    ):
        super(RSNA_Pneumonia_Dataset, self).__init__()
        np.random.seed(seed)  # Reset the seed so all runs are the same.

        self.split = split

        self.data_path = data_path
        self.transform = transform
        self.pathology_masks = pathology_masks

        self.pathologies = ["Pneumonia"] # remove lung opacity pathology

        self.pathologies = sorted(self.pathologies)

        self.extension = extension

        # Load data
        if csvpath == USE_INCLUDED_FILE:
            self.csvpath = os.path.join(data_path, "rsna_labels.csv")
        else:
            self.csvpath = csvpath

        self.raw_csv = pd.read_csv(self.csvpath)
        # self.raw_csv = df[df["split"] == split]  # Filter for split

        support_splits = ["train", "val", "test", "train_1", "train_10"]
        assert split in support_splits, f"Invalid split: {split}. Supported splits are {support_splits}"

        split_path = os.path.join(data_path, f"{split}.txt")
        # Read the .txt file using pandas
        with open(split_path, 'r') as file:
            data_split = [line.strip() for line in file]

        data_split = pd.Series(data_split)
        self.raw_csv = self.raw_csv[self.raw_csv["patientId"].isin(data_split)]

        # The labels have multiple instances for each mask
        # So we just need one to get the target label
        self.csv = self.raw_csv.groupby("patientId").first()
        self.csv = self.csv.reset_index()

        # Get our classes.
        # labels = [torch.FloatTensor(self.csv["Target"].values)] # remove column for lung opacity
        # self.labels = torch.stack(labels).T

        labels = [torch.LongTensor(self.csv["Target"].values)]
        self.labels = torch.stack(labels).T
        self.labels = torch.nn.functional.one_hot(self.labels.squeeze(), num_classes=2).float()

        # set if we have masks
        self.csv["has_masks"] = ~np.isnan(self.csv["x"])

        # patientid
        self.csv["patientId"] = self.csv["patientId"].astype(str)

    def string(self):
        return (
            self.__class__.__name__
            + " num_samples={} transforms={} split={}".format(
                len(self), self.transform, self.split
            )
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = {}
        sample["idx"] = idx
        sample["lab"] = self.labels[idx]

        image_name = self.csv["patientId"].iloc[idx]
        img_path = os.path.join(self.data_path, "images", image_name + self.extension)
        img = Image.open(img_path).convert("RGB")

        sample["img"] = img  # remove normalization step

        if self.pathology_masks:
            sample["pathology_masks"] = self.get_mask_dict(image_name)

        sample = apply_transforms(sample, self.transform)

        return sample

    def get_mask_dict(self, image_name):
        path_mask = {}
        # All masks are for both pathologies
        for patho in ["Pneumonia"]:
            mask_path = os.path.join(self.imgpath, "images", image_name + self.extension)
            mask = Image.open(mask_path)
            # Resize so image resizing works
            mask = mask[None, :, :]

            path_mask[self.pathologies.index(patho)] = mask
        return path_mask

    # def get_mask_dict(self, image_name, this_size):
    #     images_with_masks = self.raw_csv[self.raw_csv["patientId"] == image_name]
    #     path_mask = {}

    #     # All masks are for both pathologies
    #     for patho in ["Pneumonia"]:
    #         mask = np.zeros([this_size, this_size])

    #         # Don't add masks for labels we don't have
    #         if patho in self.pathologies:
    #             for i in range(len(images_with_masks)):
    #                 row = images_with_masks.iloc[i]
    #                 xywh = np.asarray([row.x, row.y, row.width, row.height])
    #                 xywh = xywh.astype(int)
    #                 mask[xywh[1] : xywh[1] + xywh[3], xywh[0] : xywh[0] + xywh[2]] = 1

    #         # Resize so image resizing works
    #         mask = mask[None, :, :]

    #         path_mask[self.pathologies.index(patho)] = mask
    #     return path_mask
    
    def get_collate_fn(self):
        def collate_fn(batch):
            imgs = [s["img"] for s in batch]
            labels = [s["lab"] for s in batch]
            collated = {
                "labels": torch.stack(labels),
                "images": torch.stack(imgs),
            }
            if self.pathology_masks:
                for i, s in enumerate(batch):
                    if i == 0:  # initialize dict of empty lists with same keys
                        masks = dict.fromkeys(s["pathology_masks"], [])
                    for k, v in s["pathology_masks"].items():
                        masks[k].append(v)
                masks = {k: torch.stack(v) for k, v in masks.items()}
                collated.update({"pathology_masks": masks})
            return collated

        return collate_fn

class SIIM_Pneumothorax_Dataset(Dataset):
    """SIIM Pneumothorax Dataset

    https://academictorrents.com/details/6ef7c6d039e85152c4d0f31d83fa70edc4aba088
    https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation

    "The data is comprised of images in DICOM format and annotations in the
    form of image IDs and run-length-encoded (RLE) masks. Some of the images
    contain instances of pneumothorax (collapsed lung), which are indicated
    by encoded binary masks in the annotations. Some training images have
    multiple annotations. Images without pneumothorax have a mask value of -1."
    """

    def __init__(
        self,
        data_path,
        csvpath=USE_INCLUDED_FILE,
        transform=None,
        seed=42,
        pathology_masks=False,
        extension=".png",
        split="train",
    ):
        super(SIIM_Pneumothorax_Dataset, self).__init__()
        np.random.seed(seed)  # Reset the seed so all runs are the same.

        self.split = split

        self.data_path = data_path
        self.transform = transform
        self.pathology_masks = pathology_masks

        # Load data
        if csvpath == USE_INCLUDED_FILE:
            self.csvpath = os.path.join(data_path, "siim_labels.csv")
        else:
            self.csvpath = csvpath
        self.csv = pd.read_csv(self.csvpath)

        support_splits = ["train", "val", "test", "train_1", "train_10"]
        assert split in support_splits, f"Invalid split: {split}. Supported splits are {support_splits}"

        split_path = os.path.join(data_path, f"{split}.txt")
        # Read the .txt file using pandas
        with open(split_path, 'r') as file:
            data_split = [line.strip() for line in file]

        data_split = pd.Series(data_split).apply(lambda x: x + extension)
        self.csv = self.csv[self.csv["new_filename"].isin(data_split)]

        self.pathologies = ["Pneumothorax", "No Pneumothorax"]
        self.extension = extension

        # labels = [torch.FloatTensor(self.csv["has_pneumo"].values)]
        # self.labels = torch.stack(labels).T

        labels = [torch.LongTensor(self.csv["has_pneumo"].values)]
        self.labels = torch.stack(labels).T
        self.labels = torch.nn.functional.one_hot(self.labels.squeeze(), num_classes=2).float()
        
        self.csv = self.csv.reset_index()

    def string(self):
        return (
            self.__class__.__name__
            + " num_samples={} transforms={} split={}".format(
                len(self), self.transform, self.split
            )
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = {}
        sample["idx"] = idx
        sample["lab"] = self.labels[idx]

        img_name = self.csv["new_filename"].iloc[idx]
        img_path = os.path.join(self.data_path, "images", img_name)

        img = Image.open(img_path).convert("RGB")
        sample["img"] = img
        
        if self.pathology_masks:
            from torchvision import transforms
            sample["pathology_masks"] = self.get_pathology_mask_dict(img_name)

        sample = apply_transforms(sample, self.transform)

        return sample

    def get_pathology_mask_dict(self, img_name):
        path_mask = {}
        for patho in ["Pneumothorax"]:
            # don't add masks for labels we don't have
            if patho in self.pathologies:
                mask_path = os.path.join(self.imgpath, "masks", img_name)
                mask = Image.open(mask_path)
            # reshape so image resizing works
            mask = mask[None, :, :]

            path_mask[self.pathologies.index(patho)] = mask
        return path_mask

    # def get_pathology_mask_dict(self, image_name, this_size):
    #     # TODO: Double check this, we will preprocess the image to 512x512
    #     base_size = 1024
    #     images_with_masks = self.csv[
    #         np.logical_and(
    #             self.csv["ImageId"] == image_name, self.csv[" EncodedPixels"] != " -1"
    #         )
    #     ]
    #     path_mask = {}

    #     # From kaggle code
    #     def rle2mask(rle, width, height):
    #         mask = np.zeros(width * height)
    #         array = np.asarray([int(x) for x in rle.split()])
    #         starts = array[0::2]
    #         lengths = array[1::2]

    #         current_position = 0
    #         for index, start in enumerate(starts):
    #             current_position += start
    #             mask[current_position : current_position + lengths[index]] = 1
    #             current_position += lengths[index]

    #         return mask.reshape(width, height)

    #     if len(images_with_masks) > 0:
    #         # Using a for loop so it is consistent with the other code
    #         for patho in ["Pneumothorax"]:
    #             mask = np.zeros([this_size, this_size])

    #             # don't add masks for labels we don't have
    #             if patho in self.pathologies:
    #                 for i in range(len(images_with_masks)):
    #                     row = images_with_masks.iloc[i]
    #                     mask = rle2mask(row[" EncodedPixels"], base_size, base_size)
    #                     mask = mask.T
    #                     mask = skimage.transform.resize(
    #                         mask, (this_size, this_size), mode="constant", order=0
    #                     )
    #                     mask = mask.round()  # make 0,1

    #             # reshape so image resizing works
    #             mask = mask[None, :, :]

    #             path_mask[self.pathologies.index(patho)] = mask

    #     return path_mask
    
    def get_collate_fn(self):
        def collate_fn(batch):
            imgs = [s["img"] for s in batch]
            labels = [s["lab"] for s in batch]
            collated = {
                "labels": torch.stack(labels),
                "images": torch.stack(imgs),
            }
            if self.pathology_masks:
                for i, s in enumerate(batch):
                    if i == 0:  # initialize dict of empty lists with same keys
                        masks = dict.fromkeys(s["pathology_masks"], [])
                    for k, v in s["pathology_masks"].items():
                        masks[k].append(v)
                masks = {k: torch.stack(v) for k, v in masks.items()}
                collated.update({"pathology_masks": masks})
            return collated

        return collate_fn

class NIH_Dataset(Dataset):
    """NIH ChestX-ray14 dataset

    ChestX-ray dataset comprises 112,120 frontal-view X-ray images of 30,
    805 unique patients with the text-mined fourteen disease image labels (
    where each image can have multi-labels), mined from the associated
    radiological reports using natural language processing. Fourteen common
    thoracic pathologies include Atelectasis, Consolidation, Infiltration,
    Pneumothorax, Edema, Emphysema, Fibrosis, Effusion, Pneumonia,
    Pleural_thickening, Cardiomegaly, Nodule, Mass and Hernia, which is an
    extension of the 8 common disease patterns listed in our CVPR2017 paper.
    Note that original radiology reports (associated with these chest x-ray
    studies) are not meant to be publicly shared for many reasons. The
    text-mined disease labels are expected to have accuracy >90%. Please find
    more details and benchmark performance of trained models based on 14
    disease labels in our arxiv paper: https://arxiv.org/abs/1705.02315

    Dataset release website:
    https://www.nih.gov/news-events/news-releases/nih-clinical-center-provides-one-largest-publicly-available-chest-x-ray-datasets-scientific-community

    Download full size images here:
    https://academictorrents.com/details/557481faacd824c83fbf57dcf7b6da9383b3235a

    Download resized (224x224) images here:
    https://academictorrents.com/details/e615d3aebce373f1dc8bd9d11064da55bdadede0
    """

    def __init__(
        self,
        data_path,
        csvpath=USE_INCLUDED_FILE,
        bbox_list_path=USE_INCLUDED_FILE,
        views=["PA", "AP"],
        transform=None,
        seed=42,
        pathology_masks=False,
        split="train",
    ):
        super(NIH_Dataset, self).__init__()

        self.seed = seed
        np.random.seed(seed)  # Reset the seed so all runs are the same.
        self.data_path = data_path

        self.split = split
        self.views = views

        if csvpath == USE_INCLUDED_FILE:
            self.csvpath = os.path.join(data_path, "Data_Entry_2017_v2020.csv.gz")
        else:
            self.csvpath = csvpath

        self.transform = transform
        self.pathology_masks = pathology_masks

        self.pathologies = [
            "Atelectasis",
            "Consolidation",
            "Infiltration",
            "Pneumothorax",
            "Edema",
            "Emphysema",
            "Fibrosis",
            "Effusion",
            "Pneumonia",
            "Pleural_Thickening",
            "Cardiomegaly",
            "Nodule",
            "Mass",
            "Hernia",
        ]

        self.pathologies = sorted(self.pathologies)

        # Load data
        self.check_paths_exist()
        self.csv = pd.read_csv(self.csvpath)
        # self.csv = self.csv[self.csv["split"] == split] # Filter for split

        support_splits = ["train", "val", "test", "train_1", "train_10"]
        assert split in support_splits, f"Invalid split: {split}. Supported splits are {support_splits}"

        split_path = os.path.join(data_path, f"{split}.txt")
        # Read the .txt file using pandas
        with open(split_path, 'r') as file:
            data_split = [line.strip() for line in file]

        data_split = pd.Series(data_split)
        self.csv = self.csv[self.csv["Image Index"].isin(data_split)]

        # if data_pct != 1 and self.split == "train":
        #     downloaded_data_label_txt = None
        #     if data_pct == 0.01:
        #         downloaded_data_label_txt = os.path.join(data_path, "train_1.txt")
        #     elif data_pct == 0.1:
        #         downloaded_data_label_txt = os.path.join(data_path, "train_10.txt")
        #     elif data_pct == 1:
        #         downloaded_data_label_txt = os.path.join(data_path, "train_list.txt")
        #     else:
        #         self.csv = self.csv.sample(frac=data_pct, random_state=seed)
            
        #     if downloaded_data_label_txt:
        #         fileDescriptor = open(os.path.join(downloaded_data_label_txt), "r")
        #         trainImages = []
        #         trainLabels = []

        #         line = True
        #         while line:
        #             line = fileDescriptor.readline()
        #             if line:
        #                 lineItems = line.split()
        #                 imageFile = lineItems[0]
        #                 imageLabel = lineItems[1:]
        #                 imageLabel = [int(i) for i in imageLabel]
        #                 trainImages.append(imageFile)
        #                 trainLabels.append(imageLabel)

        #         fileDescriptor.close()
        #         self.csv = self.csv[self.csv["Image Index"].isin(pd.Series(trainImages))]

        # MRM Order
        # CLASS_NAMES = [ 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
        #         'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']

        # # Remove images with view position other than specified
        # self.csv["view"] = self.csv["View Position"]
        # self.limit_to_selected_views(views)

        # self.csv['ID'] = pd.Categorical(self.csv['Image Index'], categories=data_split.tolist(), ordered=True)
        # # Sort the DataFrame based on the 'ID' column
        # self.csv.sort_values(by='ID', inplace=True)

        self.csv = self.csv.reset_index()

        # load NIH pathology masks
        if self.pathology_masks:
            if bbox_list_path == USE_INCLUDED_FILE:
                self.bbox_list_path = os.path.join(datapath, "BBox_List_2017.csv.gz")
            else:
                self.bbox_list_path = bbox_list_path
            self.pathology_maskscsv = pd.read_csv(
                self.bbox_list_path,
                names=[
                    "Image Index",
                    "Finding Label",
                    "x",
                    "y",
                    "w",
                    "h",
                    "_1",
                    "_2",
                    "_3",
                ],
                skiprows=1,
            )

            # change label name to match
            self.pathology_maskscsv.loc[
                self.pathology_maskscsv["Finding Label"] == "Infiltrate", "Finding Label"
            ] = "Infiltration"
            self.csv["has_masks"] = self.csv["Image Index"].isin(
                self.pathology_maskscsv["Image Index"]
            )

        # self.csv['labels'] = self.csv['labels'].apply(literal_eval)
        # self.labels = torch.FloatTensor(self.csv["labels"])

        # Get our classes.
        labels = []
        for pathology in self.pathologies:
            mask = self.csv["Finding Labels"].str.contains(pathology)
            labels.append(torch.FloatTensor(mask.values))

        self.labels = torch.stack(labels).T

    def string(self):
        return (
            self.__class__.__name__
            + " num_samples={} views={} transforms={} split={}".format(
                len(self), self.views, self.transform, self.split
            )
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = {}

        imgid = self.csv["Image Index"].iloc[idx]
        img_path = os.path.join(self.data_path, "images", imgid) 
        img = Image.open(img_path).convert("RGB")

        if self.pathology_masks:
            sample["pathology_masks"] = self.get_mask_dict(
                imgid, sample["img"].shape[2]
            )

        # if self.transform != None: img = self.transform(img)
        
        sample["img"] = img
        sample["lab"] = self.labels[idx]

        sample = apply_transforms(sample, self.transform)

        return sample

    def get_mask_dict(self, image_name, this_size):
        base_size = 1024
        scale = this_size / base_size

        images_with_masks = self.pathology_maskscsv[
            self.pathology_maskscsv["Image Index"] == image_name
        ]
        path_mask = {}

        for i in range(len(images_with_masks)):
            row = images_with_masks.iloc[i]

            # Don't add masks for labels we don't have
            if row["Finding Label"] in self.pathologies:
                mask = np.zeros([this_size, this_size])
                xywh = np.asarray([row.x, row.y, row.w, row.h])
                xywh = xywh * scale
                xywh = xywh.astype(int)
                mask[xywh[1] : xywh[1] + xywh[3], xywh[0] : xywh[0] + xywh[2]] = 1

                # Resize so image resizing works
                mask = mask[None, :, :]

                path_mask[self.pathologies.index(row["Finding Label"])] = mask
        return path_mask

    def get_collate_fn(self):
        def collate_fn(batch):
            imgs = [s["img"] for s in batch]
            labels = [s["lab"] for s in batch]
            collated = {
                "labels": torch.stack(labels),
                "images": torch.stack(imgs),
            }
            if self.pathology_masks:
                for i, s in enumerate(batch):
                    if i == 0:  # initialize dict of empty lists with same keys
                        masks = dict.fromkeys(s["pathology_masks"], [])
                    for k, v in s["pathology_masks"].items():
                        masks[k].append(v)
                masks = {k: torch.stack(v) for k, v in masks.items()}
                collated.update({"pathology_masks": masks})
            return collated

        return collate_fn

class COVIDx_Dataset(Dataset):
    """COVIDx-CXR4 dataset
    Chest x-ray images for the detection of COVID-19

    Dataset website:
    https://www.kaggle.com/datasets/andyczhao/covidx-cxr2
    """

    def __init__(
        self,
        data_path,
        csvpath=USE_INCLUDED_FILE,
        transform=None,
        seed=42,
        split="train",
    ):
        super(COVIDx_Dataset, self).__init__()

        self.data_path = data_path
        np.random.seed(seed)  # Reset the seed so all runs are the same.
        self.split = split

        if csvpath == USE_INCLUDED_FILE:
            self.csvpath = os.path.join(data_path, "covidx_labels.csv")
        else:
            self.csvpath = csvpath
        self.csv = pd.read_csv(self.csvpath)

        self.transform = transform

        support_splits = ["train", "val", "test", "train_1", "train_10"]
        assert split in support_splits, f"Invalid split: {split}. Supported splits are {support_splits}"

        split_path = os.path.join(data_path, f"{split}.txt")
        # Read the .txt file using pandas
        with open(split_path, 'r') as file:
            data_split = [line.strip() for line in file]

        data_split = pd.Series(data_split)
        self.csv = self.csv[self.csv["filename"].isin(data_split)]

        # if data_pct != 1 and self.split == "train":
        #     self.csv = self.csv.sample(frac=data_pct, random_state=seed)

        self.csv = self.csv.reset_index()

        self.pathologies = ["Negative", "Positive"]
        # self.pathologies = ["Positive"]

        # Get our classes.
        labels = []
        for pathology in self.pathologies:
            mask = self.csv["class"].str.contains(pathology.lower())
            labels.append(torch.FloatTensor(mask.values))
        
        self.labels = torch.stack(labels).T

    def string(self):
        return (
            self.__class__.__name__
            + " num_samples={} transforms={} split={}".format(
                len(self), self.transform, self.split
            )
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = {}

        img_name = self.csv["filename"].iloc[idx]
        img_path = os.path.join(self.data_path, img_name)
        img = Image.open(img_path).convert("RGB")
        
        sample["img"] = img
        sample["lab"] = self.labels[idx]

        sample = apply_transforms(sample, self.transform)

        return sample

    def get_collate_fn(self):
        def collate_fn(batch):
            imgs = [s["img"] for s in batch]
            labels = [s["lab"] for s in batch]
            collated = {
                "labels": torch.stack(labels),
                "images": torch.stack(imgs),
            }
            return collated

        return collate_fn

class VinDr_Dataset(Dataset):
    """VinBrain Dataset

    .. code-block:: python

        d_vin = xrv.datasets.VinDr_Dataset(
            imgpath=".../train",
            csvpath=".../train.csv"
        )

    Nguyen, H. Q., Lam, K., Le, L. T., Pham, H. H., Tran, D. Q., Nguyen,
    D. B., Le, D. D., Pham, C. M., Tong, H. T. T., Dinh, D. H., Do, C. D.,
    Doan, L. T., Nguyen, C. N., Nguyen, B. T., Nguyen, Q. V., Hoang, A. D.,
    Phan, H. N., Nguyen, A. T., Ho, P. H., … Vu, V. (2020). VinDr-CXR: An
    open dataset of chest X-rays with radiologist’s annotations.
    http://arxiv.org/abs/2012.15029

    https://www.kaggle.com/c/vinbigdata-chest-xray-abnormalities-detection
    """

    def __init__(self,
                 data_path,
                 csvpath=USE_INCLUDED_FILE,
                 views=None,
                 transform=None,
                 data_aug=None,
                 seed=42,
                 pathology_masks=False,
                 split="train_1",
                 ):
        super(VinDr_Dataset, self).__init__()

        np.random.seed(seed)  # Reset the seed so all runs are the same.
        self.data_path = data_path

        if csvpath == USE_INCLUDED_FILE:
            self.csvpath = os.path.join(data_path, "vindr_labels.csv")
        else:
            self.csvpath = csvpath

        self.transform = transform
        self.data_aug = data_aug
        self.pathology_masks = pathology_masks
        self.views = views

        # self.pathologies = ['Aortic enlargement',
        #                     'Atelectasis',
        #                     'Cardiomegaly',
        #                     'Calcification',
        #                     'Clavicle fracture',
        #                     'Consolidation',
        #                     'Edema',
        #                     'Emphysema',
        #                     'Enlarged PA',
        #                     'ILD',
        #                     'Infiltration',
        #                     'Lung cavity',
        #                     'Lung cyst',
        #                     'Lung Opacity',
        #                     'Mediastinal shift',
        #                     'Nodule/Mass',
        #                     'Pulmonary fibrosis',
        #                     'Pneumothorax',
        #                     'Pleural thickening',
        #                     'Pleural effusion',
        #                     'Rib fracture',
        #                     'Other lesion',
        #                     'Lung tumor',
        #                     'Pneumonia',
        #                     'Tuberculosis',
        #                     'Other disease',
        #                     'COPD',
        #                     'No finding'
        #                     ]

        # VinBig version
        self.pathologies = ['Aortic enlargement',
                            'Atelectasis',
                            'Calcification',
                            'Cardiomegaly',
                            'Consolidation',
                            'ILD',
                            'Infiltration',
                            'Lung Opacity',
                            'Nodule/Mass',
                            'Lesion',
                            'Effusion',
                            'Pleural_Thickening',
                            'Pneumothorax',
                            'Pulmonary Fibrosis']

        self.pathologies = sorted(np.unique(self.pathologies))

        self.mapping = dict()
        self.mapping["Pleural_Thickening"] = ["Pleural thickening"]
        self.mapping["Effusion"] = ["Pleural effusion"]

        # Load data
        self.check_paths_exist()
        # self.rawcsv = pd.read_csv(self.csvpath)
        # self.csv = pd.DataFrame(self.rawcsv.groupby("image_id")["class_name"].apply(lambda x: "|".join(np.unique(x))))
        # self.csv["has_masks"] = self.csv.class_name != "No finding"
        self.csv = pd.read_csv(self.csvpath)

        support_splits = ["train", "val", "test", "train_1", "train_10"]
        assert split in support_splits, f"Invalid split: {split}. Supported splits are {support_splits}"

        split_path = os.path.join(data_path, f"{split}.txt")
        # Read the .txt file using pandas
        with open(split_path, 'r') as file:
            data_split = [line.strip() for line in file]

        data_split = pd.Series(data_split)
        self.csv = self.csv[self.csv["image_id"].isin(data_split)]

        self.csv = self.csv.reset_index()

        labels = []
        for pathology in self.pathologies:
            mask = self.csv["class_name"].str.lower().str.contains(pathology.lower())
            if pathology in self.mapping:
                for syn in self.mapping[pathology]:
                    mask |= self.csv["class_name"].str.lower().str.contains(syn.lower())
            labels.append(torch.FloatTensor(mask.values))
        self.labels = torch.stack(labels).T

    def string(self):
        return self.__class__.__name__ + " num_samples={} views={} data_aug={}".format(len(self), self.views, self.data_aug)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = {}
        sample["idx"] = idx
        sample["lab"] = self.labels[idx]

        imgid = self.csv['image_id'].iloc[idx]
        img_path = os.path.join(self.data_path, "images", imgid + ".png")

        img = Image.open(img_path).convert("RGB")
        sample["img"] = img

        if self.pathology_masks:
            sample["pathology_masks"] = self.get_mask_dict(imgid, sample["img"].size)

        sample = apply_transforms(sample, self.transform)
        sample = apply_transforms(sample, self.data_aug)

        return sample

    def get_mask_dict(self, image_name, this_size): 
        # TODO: NA Now Need Update
        h, w = this_size

        path_mask = {}
        rows = self.rawcsv[self.rawcsv.image_id.str.contains(image_name)]

        for i, pathology in enumerate(self.pathologies):
            if pathology != 'No finding':
                for group_name, df_group in rows.groupby("class_name"):
                    if group_name == pathology:
                        mask = np.zeros([h, w])
                        for idx, row in df_group.iterrows():
                            mask[int(row.y_min):int(row.y_max), int(row.x_min):int(row.x_max)] = 1

                        path_mask[i] = mask[None, :, :]

        return path_mask
    
    def get_collate_fn(self):
        def collate_fn(batch):
            imgs = [s["img"] for s in batch]
            labels = [s["lab"] for s in batch]
            collated = {
                "labels": torch.stack(labels),
                "images": torch.stack(imgs),
            }
            if self.pathology_masks:
                for i, s in enumerate(batch):
                    if i == 0:  # initialize dict of empty lists with same keys
                        masks = dict.fromkeys(s["pathology_masks"], [])
                    for k, v in s["pathology_masks"].items():
                        masks[k].append(v)
                masks = {k: torch.stack(v) for k, v in masks.items()}
                collated.update({"pathology_masks": masks})
            return collated

        return collate_fn

class TBX11K_Dataset(Dataset):
    """TBX11K Tuberculosis Classification and Detection Challenge

    Citation:

    [TPAMI23] Revisiting Computer-Aided Tuberculosis Diagnosis

    More info: https://github.com/yun-liu/Tuberculosis

    Challenge site:
    https://codalab.lisn.upsaclay.fr/competitions/7916
    """

    def __init__(
        self,
        data_path,
        csvpath=USE_INCLUDED_FILE,
        transform=None,
        seed=42,
        pathology_masks=False,
        extension='.png',
        split="train",
    ):
        super(TBX11K_Dataset, self).__init__()
        np.random.seed(seed)  # Reset the seed so all runs are the same.

        self.split = split

        self.data_path = data_path
        self.transform = transform
        self.pathology_masks = pathology_masks

        self.pathologies = ["healthy", "sick_but_no_tb", "tb"]
        self.pathologies = sorted(self.pathologies)

        # Load data
        if csvpath == USE_INCLUDED_FILE:
            self.csvpath = os.path.join(datapath, "tbx11k_labels.csv")
        else:
            self.csvpath = csvpath

        self.csv = pd.read_csv(self.csvpath)

        support_splits = ["train", "val", "test", "train_1", "train_10"]
        assert split in support_splits, f"Invalid split: {split}. Supported splits are {support_splits}"

        split_path = os.path.join(data_path, f"{split}.txt")
        # Read the .txt file using pandas
        with open(split_path, 'r') as file:
            data_split = [line.strip() for line in file]

        data_split = pd.Series(data_split).apply(lambda x: x + extension)
        self.csv = self.csv[self.csv["fname"].isin(data_split)]

        self.csv = self.csv.reset_index()

        # Get our classes.
        labels = []
        for pathology in self.pathologies:
            mask = self.csv["image_type"].str.lower().str.contains(pathology.lower())
            labels.append(torch.FloatTensor(mask.values))

        self.labels = torch.stack(labels).T

        # set if we have masks
        self.csv["has_masks"] = self.csv.bbox != "none"

    def string(self):
        return (
            self.__class__.__name__
            + " num_samples={} transforms={} split={}".format(
                len(self), self.transform, self.split
            )
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = {}
        sample["idx"] = idx
        sample["lab"] = self.labels[idx]

        image_name = self.csv["fname"].iloc[idx]
        img_path = os.path.join(self.data_path, "images", image_name)

        img = Image.open(img_path).convert("RGB")

        sample["img"] = img  # remove normalization step
        sample["has_mask"] = self.csv["has_masks"][idx]

        if self.pathology_masks:
            sample["pathology_masks"] = self.get_mask_dict(image_name, self.csv["target"].iloc[idx])

        sample = apply_transforms(sample, self.transform)

        return sample

    def get_mask_dict(self, image_name, target):
        path_mask = {}

        mask_path = os.path.join(self.imgpath, "masks", image_name)
        mask = Image.open(mask_path).convert("RGB")

        # Resize so image resizing works
        mask = mask[None, :, :]

        path_mask[target] = mask
        return path_mask

    # def get_mask_dict(self, image_name, has_mask):
    #     images_with_masks = self.csv[self.csv["fname"] == image_name]
    #     this_size = images_with_masks[['image_height', 'image_width']].values.tolist()[0]

    #     path_mask = {}
    #     # All masks are for both pathologies
    #     if has_mask:
    #         mask = np.zeros(this_size)
    #         for i in range(len(images_with_masks)):
    #             row = images_with_masks.iloc[i]
    #             xywh = eval(row.bbox)
    #             xywh = np.array(list(xywh.values()))
    #             # xywh = xywh * scale
    #             xywh = xywh.astype(int)
    #             mask[xywh[1] : xywh[1] + xywh[3], xywh[0] : xywh[0] + xywh[2]] = 1

    #         # Resize so image resizing works
    #         mask = mask[None, :, :]

    #         path_mask[images_with_masks.target.values[0]] = mask
    #     return path_mask
    
    def get_collate_fn(self):
        def collate_fn(batch):
            imgs = [s["img"] for s in batch]
            labels = [s["lab"] for s in batch]
            collated = {
                "labels": torch.stack(labels),
                "images": torch.stack(imgs),
            }
            if self.pathology_masks:
                for i, s in enumerate(batch):
                    if i == 0:  # initialize dict of empty lists with same keys
                        masks = dict.fromkeys(s["pathology_masks"], [])
                    for k, v in s["pathology_masks"].items():
                        masks[k].append(v)
                masks = {k: torch.stack(v) for k, v in masks.items()}
                collated.update({"pathology_masks": masks})
            return collated

        return collate_fn

class VQA_RAD_Dataset(Dataset):
    """VQA-RAD dataset

    VQA-RAD is the first manually constructed dataset where clinicians asked
    naturally occurring questions about radiology images and provided reference
    answers. Manual categorization of images and questions provides insight into
    clinically relevant tasks and the natural language to phrase them. VQA-RAD
    consists of 3,515 question-answer pairs on 315 radiology images. We downloaded
    the pre-processed version of the dataset and combined the train and validation
    splits into one csv file for easier processing.

    Paper: https://arxiv.org/pdf/2305.10415v5.pdf

    Download the raw dataset here: https://osf.io/89kps/

    Download the pre-processed dataset through this repo: 
    https://github.com/aioz-ai/MICCAI19-MedVQA
    """

    def __init__(
        self,
        imgpath,
        csvpath=USE_INCLUDED_FILE,
        transform=None,
        seed=42,
        unique_patients=True,
        split="train",
        tokenizer=None,
        max_words=None,
    ):
        super(VQA_RAD_Dataset, self).__init__()
        np.random.seed(seed)  # Reset the seed so all runs are the same.

        self.split = split

        self.imgpath = imgpath

        if csvpath == USE_INCLUDED_FILE:
            self.csvpath = os.path.join(datapath, "vqarad_train_valset.csv.gz")
        else:
            self.csvpath = csvpath

        self.transform = transform

        # Load data
        self.check_paths_exist()
        self.csv = pd.read_csv(self.csvpath)

        # Construct global vocabulary
        (
            self.ans2label,
            self.label2ans,
        ) = self.construct_vocabulary()  # Should be called before filtering for split

        if split == "test":
            split = "valid"  # use same split for test and validation

        self.csv = self.csv[self.csv["split"] == split]  # Filter for split

        self.csv = self.csv.reset_index()

        # Load tokenizer, if exists
        if tokenizer is not None:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
            self.tokenizer_args = {
                "return_tensors": "pt",
                "padding": True,
                "add_special_tokens": True,
            }
            if max_words is not None:
                self.tokenizer_args.update(
                    {
                        "padding": "max_length",
                        "truncation": True,
                        "max_length": max_words,
                    }
                )

        ####### VQA answer & labels ########

        # Normalize word (Set pathologies to be answer in Med-VQA dataset)
        self.pathologies = self.csv["answer"].apply(
            lambda x: unifier.datasets.utils.normalize_word(str(x)).lower()
        )

        # Construct labels
        labels = self.pathologies.apply(lambda x: self.ans2label[x]).tolist()
        self.labels = np.asarray(labels).T

    def string(self):
        return (
            self.__class__.__name__
            + " num_samples={} transforms={} split={}".format(
                len(self), self.transform, self.split
            )
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = {}
        sample["idx"] = idx
        sample["lab"] = self.labels[idx]

        imgid = self.csv["image_name"].iloc[idx]
        question = self.csv["question"].iloc[idx]
        sample["text"] = self.get_text(question)

        img_path = os.path.join(self.imgpath, imgid)
        img = imread(img_path)
        img = Image.fromarray(img).convert("RGB")

        sample["img"] = img

        sample = apply_transforms(sample, self.transform)

        return sample

    def construct_vocabulary(self):  # Called before filtering for split
        questions = self.csv.to_dict(orient="records")

        # Construct vocabulary
        all_major_answers = list()
        for q in tqdm(questions):
            all_major_answers.append(str(q["answer"]).lower())
        all_major_answers = [
            unifier.datasets.utils.normalize_word(word)
            for word in tqdm(all_major_answers)
        ]
        counter = {k: v for k, v in Counter(all_major_answers).items() if v >= 0}
        ans2label = {k: i for i, k in enumerate(counter.keys())}
        label2ans = list(counter.keys())
        print("Label size ({}): {}".format("VQA-RAD", len(ans2label)))

        return ans2label, label2ans

    def get_text(self, text):
        sents = split_into_sents(text)
        text = " ".join(sents)

        encoding = self.tokenizer(
            text,
            # padding="max_length",
            # truncation=True,
            # max_length=self.max_words,
            return_special_tokens_mask=True,
            return_offsets_mapping=True,
            **self.tokenizer_args,
        )

        return encoding

    def get_collate_fn(self):
        def collate_fn(batch):
            imgs = [s["img"] for s in batch]
            labels = [torch.tensor(s["lab"]).long() for s in batch]  # Numerical labels, not tensors

            collated = {
                "labels": torch.stack(labels),
                "images": torch.stack(imgs)
            }

            if self.tokenizer:
                texts = {
                    "text_ids": torch.stack(
                        [torch.as_tensor(s["text"]["input_ids"]) for s in batch]
                    ).squeeze(),
                    "text_masks": torch.stack(
                        [torch.as_tensor(s["text"]["attention_mask"]) for s in batch]
                    ).squeeze(),
                }
                collated["texts"] = texts

            return collated

        return collate_fn

class MedVQA_2019_Dataset(Dataset):
    """ Med-VQA-2019 Dataset
    
        VQA-Med 2019 focused on radiology images and four main categories of questions: Modality, 
        Plane, Organ system and Abnormality. These categories are designed with different degrees 
        of difficulty leveraging both classification and text generation approaches. In this second 
        edition of the VQA challenge, we targeted medical questions asking about one element only 
        (e.g. what is the organ principally shown in this mri? in what plane is this mammograph taken? 
        is this a t1 weighted, t2 weighted, or flair image? what is most alarming about this 
        ultrasound?), and that can be answered from the image content without requiring additional 
        medical knowledge or domain-specific inference.

        The VQA-Med-2019 dataset includes a training set of 3,200 medical images with 12,792 
        Question-Answer (QA) pairs, a validation set of 500 medical images with 2,000 QA pairs, and 
        a test set of 500 medical images with 500 questions.

        Original repo: https://github.com/abachaa/VQA-Med-2019 
    """

    def __init__(
        self,
        imgpath,
        csvpath=USE_INCLUDED_FILE,
        labelspath=USE_INCLUDED_FILE,
        transform=None,
        seed=42,
        unique_patients=True,
        split="train",
        tokenizer=None,
        max_words=None,
    ):
        super(MedVQA_2019_Dataset, self).__init__()
        np.random.seed(seed)  # Reset the seed so all runs are the same.

        self.split = split

        self.imgpath = imgpath
        self.labelspath = labelspath

        if csvpath == USE_INCLUDED_FILE:
            self.csvpath = os.path.join(datapath, "med_vqa_2019_dataset.csv.gz")
        else:
            self.csvpath = csvpath

        self.transform = transform

        # Load data
        self.check_paths_exist()
        self.csv = pd.read_csv(self.csvpath)

        # Construct global vocabulary
        (
            self.ans2label,
            self.label2ans,
        ) = self.construct_vocabulary()  # Should be called before filtering for split

        self.csv = self.csv[self.csv["split"] == split]  # Filter for split
        self.csv = self.csv.reset_index()

        # Construct tokenizer
        if tokenizer is not None:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
            self.tokenizer_args = {
                "return_tensors": "pt",
                "padding": True,
                "add_special_tokens": True,
            }
            if max_words is not None:
                self.tokenizer_args.update(
                    {
                        "padding": "max_length",
                        "truncation": True,
                        "max_length": max_words,
                    }
                )

        ####### VQA answer & labels ########

        # Normalize word (Set pathologies to be answer in Med-VQA dataset)
        self.pathologies = self.ans2label.keys()

        # Construct labels
        labels = self.csv["answer"].apply(
                    lambda x: self.ans2label[unifier.datasets.utils.normalize_word(str(x)).lower()]
                 ).tolist()
        self.labels = np.asarray(labels).T

    def string(self):
        return (
            self.__class__.__name__
            + " num_samples={} transforms={} split={}".format(
                len(self), self.transform, self.split
            )
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = {}
        sample["idx"] = idx
        sample["lab"] = self.labels[idx]

        question = self.csv["question"].iloc[idx]
        sample["text"] = self.get_text(question)

        imgid = self.csv["image_path"].iloc[idx]
        img_path = os.path.join(self.imgpath, imgid + ".jpg")
        img = imread(img_path)
        img = Image.fromarray(img).convert("RGB")

        sample["img"] = img

        sample = apply_transforms(sample, self.transform)

        return sample

    def construct_vocabulary(self):  # Called before filtering for split
        questions = self.csv.to_dict(orient="records")

        # Construct vocabulary
        all_major_answers = list()
        for q in tqdm(questions):
            all_major_answers.append(str(q["answer"]).lower())
        all_major_answers = [
            unifier.datasets.utils.normalize_word(word)
            for word in tqdm(all_major_answers)
        ]
        counter = {k: v for k, v in Counter(all_major_answers).items() if v >= 0}
        ans2label = {k: i for i, k in enumerate(counter.keys())}
        label2ans = list(counter.keys())
        print("Label size ({}): {}".format("MedVQA-2019", len(ans2label)))

        return ans2label, label2ans

    def get_text(self, text):
        sents = split_into_sents(text)
        text = " ".join(sents)

        encoding = self.tokenizer(
            text,
            # padding="max_length",
            # truncation=True,
            # max_length=self.max_words,
            return_special_tokens_mask=True,
            return_offsets_mapping=True,
            **self.tokenizer_args,
        )

        return encoding

    def get_collate_fn(self):
        def collate_fn(batch):
            imgs = [s["img"] for s in batch]
            labels = [torch.tensor(s["lab"]).long() for s in batch]  # Numerical labels

            collated = {
                "labels": torch.stack(labels),  # tensor labels for each sample in batch
                "images": torch.stack(imgs),
            }

            if self.tokenizer:
                texts = {
                    "text_ids": torch.stack(
                        [torch.as_tensor(s["text"]["input_ids"]) for s in batch]
                    ).squeeze(),
                    "text_masks": torch.stack(
                        [torch.as_tensor(s["text"]["attention_mask"]) for s in batch]
                    ).squeeze(),
                }
                collated["texts"] = texts

            return collated

        return collate_fn

class SLAKE_Dataset(Dataset):
    """ SLAKE Dataset

        Downloaded from: https://github.com/xiaoman-zhang/PMC-VQA
    """

    def __init__(
        self,
        imgpath,
        csvpath=USE_INCLUDED_FILE,
        transform=None,
        seed=42,
        unique_patients=True,
        split="train",
        tokenizer=None,
        max_words=None,
    ):
        super(SLAKE_Dataset, self).__init__()
        np.random.seed(seed)  # Reset the seed so all runs are the same.

        self.split = split

        self.imgpath = imgpath

        if csvpath == USE_INCLUDED_FILE:
            self.csvpath = os.path.join(datapath, "slake_dataset.csv.gz")
        else:
            self.csvpath = csvpath

        self.transform = transform

        # Load data
        self.check_paths_exist()
        self.csv = pd.read_csv(self.csvpath)

        # Filter only English QA pairs
        self.csv = self.csv[self.csv["q_lang"] == "en"]

        # Construct global vocabulary
        (
            self.ans2label,
            self.label2ans,
        ) = self.construct_vocabulary()  # Should be called before filtering for split

        self.csv = self.csv[self.csv["split"] == split]  # Filter for split
        self.csv = self.csv.reset_index()

        # Load tokenizer, if exists
        if tokenizer is not None:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
            self.tokenizer_args = {
                "return_tensors": "pt",
                "padding": True,
                "add_special_tokens": True,
            }
            if max_words is not None:
                self.tokenizer_args.update(
                    {
                        "padding": "max_length",
                        "truncation": True,
                        "max_length": max_words,
                    }
                )

        ####### VQA answer & labels ########

        # Normalize word (Set pathologies to be answer in Med-VQA dataset)
        self.pathologies = self.csv["answer"].apply(
            lambda x: unifier.datasets.utils.normalize_word(str(x)).lower()
        )

        # Construct labels
        labels = self.pathologies.apply(lambda x: self.ans2label[x]).tolist()
        self.labels = np.asarray(labels).T

    def string(self):
        return (
            self.__class__.__name__
            + " num_samples={} transforms={} split={}".format(
                len(self), self.transform, self.split
            )
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = {}
        sample["idx"] = idx
        sample["lab"] = self.labels[idx]

        imgid = self.csv["img_name"].iloc[idx]
        question = self.csv["question"].iloc[idx]
        sample["text"] = self.get_text(question)

        img_path = os.path.join(self.imgpath, imgid)
        img = imread(img_path)
        img = Image.fromarray(img).convert("RGB")

        sample["img"] = img

        sample = apply_transforms(sample, self.transform)

        return sample

    def construct_vocabulary(self):  # Called before filtering for split
        questions = self.csv.to_dict(orient="records")

        # Construct vocabulary
        all_major_answers = list()
        for q in tqdm(questions):
            all_major_answers.append(str(q["answer"]).lower())
        all_major_answers = [
            unifier.datasets.utils.normalize_word(word)
            for word in tqdm(all_major_answers)
        ]
        counter = {k: v for k, v in Counter(all_major_answers).items() if v >= 0}
        ans2label = {k: i for i, k in enumerate(counter.keys())}
        label2ans = list(counter.keys())
        print("Label size ({}): {}".format("SLAKE", len(ans2label)))

        return ans2label, label2ans

    def get_text(self, text):
        sents = split_into_sents(text)
        text = " ".join(sents)

        encoding = self.tokenizer(
            text,
            # padding="max_length",
            # truncation=True,
            # max_length=self.max_words,
            return_special_tokens_mask=True,
            return_offsets_mapping=True,
            **self.tokenizer_args,
        )

        return encoding

    def get_collate_fn(self):
        def collate_fn(batch):
            imgs = [s["img"] for s in batch]
            labels = [torch.tensor(s["lab"]).long() for s in batch]  # Numerical labels, not tensors

            collated = {
                "labels": torch.stack(labels),
                "images": torch.stack(imgs)
            }

            if self.tokenizer:
                texts = {
                    "text_ids": torch.stack(
                        [torch.as_tensor(s["text"]["input_ids"]) for s in batch]
                    ).squeeze(),
                    "text_masks": torch.stack(
                        [torch.as_tensor(s["text"]["attention_mask"]) for s in batch]
                    ).squeeze(),
                }
                collated["texts"] = texts

            return collated

        return collate_fn

class IU_Dataset(Dataset):
    """Indiana University Chest X-Ray dataset

    Download link: https://www.kaggle.com/datasets/raddar/chest-xrays-indiana-university

    Original source: https://openi.nlm.nih.gov/
    """

    def __init__(
        self,
        imgpath,
        csvpath=USE_INCLUDED_FILE,
        transform=None,
        seed=42,
        split="train",
        max_words=None,
    ):
        super(IU_Dataset, self).__init__()
        np.random.seed(seed)  # Reset the seed so all runs are the same.

        self.split = split

        self.imgpath = imgpath

        if csvpath == USE_INCLUDED_FILE:
            self.csvpath = os.path.join(datapath, "iu_xray_annotations.csv.gz")
        else:
            self.csvpath = csvpath

        self.transform = transform

        # Load data
        self.check_paths_exist()
        self.csv = pd.read_csv(self.csvpath)

        self.csv = self.csv[self.csv["split"] == split]  # Filter for split

        self.csv = self.csv.reset_index()

        # Load tokenizer
        self.tokenizer = Tokenizer(self.csvpath)

        # Tokenize
        input_ids = []
        masks = []
        for i in range(len(self.csv)):
            encoding = self.tokenizer(self.csv.iloc[i]["report"])[:max_words]
            input_ids.append(encoding)
            masks.append([1] * len(encoding))

        self.labels = input_ids
        self.masks = masks

    def string(self):
        return (
            self.__class__.__name__
            + " num_samples={} transforms={} split={}".format(
                len(self), self.transform, self.split
            )
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = dict()
        row = self.csv.iloc[idx]

        image_1 = Image.open(os.path.join(self.imgpath, eval(row["image_path"])[0])).convert("RGB")
        image_2 = Image.open(os.path.join(self.imgpath, eval(row["image_path"])[1])).convert("RGB")

        sample["img"] = [image_1, image_2] # multi-image
        sample["report_ids"] = self.labels[idx]
        sample["report_masks"] = self.masks[idx]
        sample["seq_length"] = len(sample["report_ids"])

        sample = apply_transforms(sample, self.transform)

        # stack both images
        sample["img"] = torch.stack(sample["img"], 0)

        return sample["img"], sample["report_ids"], sample["report_masks"], sample["seq_length"]

    def get_collate_fn(self):
        def collate_fn(batch):
            images, reports_ids, reports_masks, seq_lengths = zip(*batch)

            images = torch.stack(images, 0)
            max_seq_length = max(seq_lengths)

            targets = np.zeros((len(reports_ids), max_seq_length), dtype=int)
            targets_masks = np.zeros((len(reports_ids), max_seq_length), dtype=int)

            for i, report_ids in enumerate(reports_ids):
                targets[i, :len(report_ids)] = report_ids

            for i, report_masks in enumerate(reports_masks):
                targets_masks[i, :len(report_masks)] = report_masks

            collated = {
                "images": images,
                "reports_ids": torch.LongTensor(targets),
                "reports_masks": torch.FloatTensor(targets_masks),
            }
            
            return collated

        return collate_fn


