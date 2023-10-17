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
import unifier
from collections import Counter
from transformers import AutoTokenizer

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

# this is for caching small things for speed
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

    labels: np.ndarray
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
        if not os.path.isdir(self.imgpath):
            raise Exception("imgpath must be a directory")
        if not os.path.isfile(self.csvpath):
            raise Exception("csvpath must be a file")

    def limit_to_selected_views(self, views):
        """This function is called by subclasses to filter the
        images by view based on the values in .csv['view']
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
        seed=0,
        unique_patients=False,
        split="train",
        section=None,  # impression, findings, or both
        tokenizer=None,
        max_words=112,
        **kwargs,
    ):
        super(MIMIC_Dataset, self).__init__()
        np.random.seed(seed)  # Reset the seed so all runs are the same.

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

            labels.append(mask.values)
        self.labels = np.asarray(labels).T
        self.labels = self.labels.astype(np.float32)

        # Make all the -1 values into nans to keep things simple
        self.labels[self.labels == -1] = np.nan

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
            + " num_samples={} views={} transforms={}".format(
                len(self), self.views, self.transform
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
            collated = {
                "input_ids": seq.input_ids,
                "attention_mask": seq.attention_mask,
                "images": torch.stack(imgs),
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

    A small validation set is provided with the data as well, but is so tiny,
    it is not included here.
    """

    def __init__(
        self,
        imgpath,
        csvpath=USE_INCLUDED_FILE,
        views=["PA"],
        transform=None,
        flat_dir=True,
        seed=0,
        unique_patients=True,
        split="train",
    ):
        super(CheX_Dataset, self).__init__()
        np.random.seed(seed)  # Reset the seed so all runs are the same.

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
        if csvpath == USE_INCLUDED_FILE:
            self.csvpath = os.path.join(datapath, "chexpert_train.csv.gz")
        else:
            self.csvpath = csvpath
        self.csv = pd.read_csv(self.csvpath)
        self.csv = self.csv[self.csv["split"] == split]  # filter for split
        self.views = views

        self.csv["view"] = self.csv["Frontal/Lateral"]  # Assign view column
        self.csv.loc[(self.csv["view"] == "Frontal"), "view"] = self.csv[
            "AP/PA"
        ]  # If Frontal change with the corresponding value in the AP/PA column otherwise remains Lateral
        self.csv["view"] = self.csv["view"].replace(
            {"Lateral": "L"}
        )  # Rename Lateral with L

        self.limit_to_selected_views(views)

        if unique_patients:
            self.csv["PatientID"] = self.csv["Path"].str.extract(pat=r"(patient\d+)")
            self.csv = self.csv.groupby("PatientID").first().reset_index()

        # Get our classes.
        healthy = self.csv["No Finding"] == 1
        labels = []
        for pathology in self.pathologies:
            if pathology in self.csv.columns:
                if pathology != "Support Devices":
                    self.csv.loc[healthy, pathology] = 0
                mask = self.csv[pathology]

            labels.append(mask.values)
        self.labels = np.asarray(labels).T
        self.labels = self.labels.astype(np.float32)

        # Make all the -1 values into nans to keep things simple
        self.labels[self.labels == -1] = np.nan

        # Rename pathologies
        self.pathologies = list(
            np.char.replace(self.pathologies, "Pleural Effusion", "Effusion")
        )

        # patientid
        if "train" in self.csvpath:
            patientid = self.csv.Path.str.split("train/", expand=True)[1]
        elif "valid" in self.csvpath:
            patientid = self.csv.Path.str.split("valid/", expand=True)[1]
        else:
            raise NotImplementedError

        patientid = patientid.str.split("/study", expand=True)[0]
        patientid = patientid.str.replace("patient", "")

        # patientid
        self.csv["patientid"] = patientid

        # age
        self.csv["age_years"] = self.csv["Age"] * 1.0
        self.csv.loc[self.csv["Age"] == 0, "Age"] = None

        # sex
        self.csv["sex_male"] = self.csv["Sex"] == "Male"
        self.csv["sex_female"] = self.csv["Sex"] == "Female"

    def string(self):
        return (
            self.__class__.__name__
            + " num_samples={} views={} transforms={}".format(
                len(self), self.views, self.transform
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
        imgid = imgid.replace("CheXpert-v1.0-small/", "").replace("CheXpert-v1.0/", "")
        img_path = os.path.join(self.imgpath, imgid)
        img = imread(img_path)
        img = Image.fromarray(img).convert("RGB")

        sample["img"] = img

        sample = apply_transforms(sample, self.transform)

        return sample


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
        imgpath,
        csvpath=USE_INCLUDED_FILE,
        dicomcsvpath=USE_INCLUDED_FILE,
        views=["PA"],
        transform=None,
        nrows=None,
        seed=0,
        pathology_masks=False,
        extension=".jpg",
        split="train",
    ):
        super(RSNA_Pneumonia_Dataset, self).__init__()
        np.random.seed(seed)  # Reset the seed so all runs are the same.
        self.imgpath = imgpath
        self.transform = transform
        self.pathology_masks = pathology_masks

        self.pathologies = ["Pneumonia", "Lung Opacity"]

        self.pathologies = sorted(self.pathologies)

        self.extension = extension
        self.use_pydicom = extension == ".dcm"

        # Load data
        if csvpath == USE_INCLUDED_FILE:
            self.csvpath = os.path.join(datapath, "kaggle_stage_2_train_labels.csv.zip")
        else:
            self.csvpath = csvpath

        df = pd.read_csv(self.csvpath, nrows=nrows)
        self.raw_csv = df[df["split"] == split]  # Filter for split

        # The labels have multiple instances for each mask
        # So we just need one to get the target label
        self.csv = self.raw_csv.groupby("patientId").first()

        if dicomcsvpath == USE_INCLUDED_FILE:
            self.dicomcsvpath = os.path.join(
                datapath, "kaggle_stage_2_train_images_dicom_headers.csv.gz"
            )
        else:
            self.dicomcsvpath = dicomcsvpath

        self.dicomcsv = pd.read_csv(
            self.dicomcsvpath, nrows=nrows, index_col="PatientID"
        )

        self.csv = self.csv.join(self.dicomcsv, on="patientId")

        # Remove images with view position other than specified
        self.csv["view"] = self.csv["ViewPosition"]
        self.limit_to_selected_views(views)

        self.csv = self.csv.reset_index()

        # Get our classes.
        labels = [self.csv["Target"].values, self.csv["Target"].values]

        # set if we have masks
        self.csv["has_masks"] = ~np.isnan(self.csv["x"])

        self.labels = np.asarray(labels).T
        self.labels = self.labels.astype(np.float32)

        # patientid
        self.csv["patientid"] = self.csv["patientId"].astype(str)

    def string(self):
        return (
            self.__class__.__name__
            + " num_samples={} views={} transforms={}".format(
                len(self), self.views, self.transform
            )
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = {}
        sample["idx"] = idx
        sample["lab"] = self.labels[idx]

        imgid = self.csv["patientId"].iloc[idx]
        img_path = os.path.join(self.imgpath, imgid + self.extension)

        if self.use_pydicom:
            try:
                import pydicom
            except ImportError as e:
                raise Exception("Please install pydicom to work with this dataset")

            img = pydicom.filereader.dcmread(img_path).pixel_array
        else:
            img = imread(img_path)
        img = Image.fromarray(img).convert("RGB")

        sample["img"] = img  # remove normalization step

        if self.pathology_masks:
            sample["pathology_masks"] = self.get_mask_dict(
                imgid, sample["img"].shape[2]
            )

        sample = apply_transforms(sample, self.transform)

        return sample

    def get_mask_dict(self, image_name, this_size):
        base_size = 1024
        scale = this_size / base_size

        images_with_masks = self.raw_csv[self.raw_csv["patientId"] == image_name]
        path_mask = {}

        # All masks are for both pathologies
        for patho in ["Pneumonia", "Lung Opacity"]:
            mask = np.zeros([this_size, this_size])

            # Don't add masks for labels we don't have
            if patho in self.pathologies:
                for i in range(len(images_with_masks)):
                    row = images_with_masks.iloc[i]
                    xywh = np.asarray([row.x, row.y, row.width, row.height])
                    xywh = xywh * scale
                    xywh = xywh.astype(int)
                    mask[xywh[1] : xywh[1] + xywh[3], xywh[0] : xywh[0] + xywh[2]] = 1

            # Resize so image resizing works
            mask = mask[None, :, :]

            path_mask[self.pathologies.index(patho)] = mask
        return path_mask


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
        imgpath,
        csvpath=USE_INCLUDED_FILE,
        transform=None,
        seed=0,
        unique_patients=True,
        pathology_masks=False,
        split="train",
    ):
        super(SIIM_Pneumothorax_Dataset, self).__init__()
        np.random.seed(seed)  # Reset the seed so all runs are the same.
        self.imgpath = imgpath
        self.transform = transform
        self.pathology_masks = pathology_masks

        # Load data
        if csvpath == USE_INCLUDED_FILE:
            self.csvpath = os.path.join(datapath, "siim-pneumothorax-train-rle.csv.gz")
        else:
            self.csvpath = csvpath

        self.csv = pd.read_csv(self.csvpath)
        self.csv = self.csv[self.csv["split"] == split]  # Filter for split

        self.pathologies = ["Pneumothorax"]

        labels = [self.csv[" EncodedPixels"] != " -1"]
        self.labels = np.asarray(labels).T
        self.labels = self.labels.astype(np.float32)

        self.csv = self.csv.reset_index()

        self.csv["has_masks"] = self.csv[" EncodedPixels"] != " -1"

        # To figure out the paths
        if not ("siim_file_map" in _cache_dict):
            file_map = {}
            for root, directories, files in os.walk(self.imgpath, followlinks=False):
                for filename in files:
                    filePath = os.path.join(root, filename)
                    file_map[filename] = filePath
            _cache_dict["siim_file_map"] = file_map
        self.file_map = _cache_dict["siim_file_map"]

    def string(self):
        return self.__class__.__name__ + " num_samples={} transforms={}".format(
            len(self), self.transform
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = {}
        sample["idx"] = idx
        sample["lab"] = self.labels[idx]

        imgid = self.csv["ImageId"].iloc[idx]
        img_path = self.file_map[imgid + ".dcm"]

        try:
            import pydicom
        except ImportError as e:
            raise Exception("Please install pydicom to work with this dataset")
        img = pydicom.filereader.dcmread(img_path).pixel_array
        img = Image.fromarray(img).convert("RGB")

        sample["img"] = img

        if self.pathology_masks:
            sample["pathology_masks"] = self.get_pathology_mask_dict(
                imgid, sample["img"].shape[2]
            )

        sample = apply_transforms(sample, self.transform)

        return sample

    def get_pathology_mask_dict(self, image_name, this_size):
        base_size = 1024
        images_with_masks = self.csv[
            np.logical_and(
                self.csv["ImageId"] == image_name, self.csv[" EncodedPixels"] != " -1"
            )
        ]
        path_mask = {}

        # From kaggle code
        def rle2mask(rle, width, height):
            mask = np.zeros(width * height)
            array = np.asarray([int(x) for x in rle.split()])
            starts = array[0::2]
            lengths = array[1::2]

            current_position = 0
            for index, start in enumerate(starts):
                current_position += start
                mask[current_position : current_position + lengths[index]] = 1
                current_position += lengths[index]

            return mask.reshape(width, height)

        if len(images_with_masks) > 0:
            # Using a for loop so it is consistent with the other code
            for patho in ["Pneumothorax"]:
                mask = np.zeros([this_size, this_size])

                # don't add masks for labels we don't have
                if patho in self.pathologies:
                    for i in range(len(images_with_masks)):
                        row = images_with_masks.iloc[i]
                        mask = rle2mask(row[" EncodedPixels"], base_size, base_size)
                        mask = mask.T
                        mask = skimage.transform.resize(
                            mask, (this_size, this_size), mode="constant", order=0
                        )
                        mask = mask.round()  # make 0,1

                # reshape so image resizing works
                mask = mask[None, :, :]

                path_mask[self.pathologies.index(patho)] = mask

        return path_mask


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
    text-mined disease labels are expected to have accuracy >90%.Please find
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
        imgpath,
        csvpath=USE_INCLUDED_FILE,
        bbox_list_path=USE_INCLUDED_FILE,
        views=["PA"],
        transform=None,
        seed=0,
        unique_patients=True,
        pathology_masks=False,
        split="train",
    ):
        super(NIH_Dataset, self).__init__()

        np.random.seed(seed)  # Reset the seed so all runs are the same.
        self.imgpath = imgpath

        if csvpath == USE_INCLUDED_FILE:
            self.csvpath = os.path.join(datapath, "Data_Entry_2017_v2020.csv.gz")
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
        self.csv = self.csv[self.csv["split"] == split]  # Filter for split

        # Remove images with view position other than specified
        self.csv["view"] = self.csv["View Position"]
        self.limit_to_selected_views(views)

        if unique_patients:
            self.csv = self.csv.groupby("Patient ID").first()

        self.csv = self.csv.reset_index()

        ####### pathology masks ########
        # load nih pathology masks
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

        ####### pathology masks ########
        # Get our classes.
        self.labels = []
        for pathology in self.pathologies:
            self.labels.append(
                self.csv["Finding Labels"].str.contains(pathology).values
            )

        self.labels = np.asarray(self.labels).T
        self.labels = self.labels.astype(np.float32)

        # patientid
        self.csv["patientid"] = self.csv["Patient ID"].astype(str)

        # age
        self.csv["age_years"] = self.csv["Patient Age"] * 1.0

        # sex
        self.csv["sex_male"] = self.csv["Patient Gender"] == "M"
        self.csv["sex_female"] = self.csv["Patient Gender"] == "F"

    def string(self):
        return (
            self.__class__.__name__
            + " num_samples={} views={} transforms={}".format(
                len(self), self.views, self.transform
            )
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = {}
        sample["idx"] = idx
        sample["lab"] = self.labels[idx]

        imgid = self.csv["Image Index"].iloc[idx]
        img_path = os.path.join(self.imgpath, imgid)
        img = imread(img_path)
        img = Image.fromarray(img).convert("RGB")

        sample["img"] = img

        if self.pathology_masks:
            sample["pathology_masks"] = self.get_mask_dict(
                imgid, sample["img"].shape[2]
            )

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

    Download the raw dataset here:
    https://osf.io/89kps/

    Download the pre-processed dataset through this repo:
    https://github.com/aioz-ai/MICCAI19-MedVQA
    """

    def __init__(
        self,
        imgpath,
        csvpath=USE_INCLUDED_FILE,
        transform=None,
        seed=0,
        unique_patients=True,
        split="train",
    ):
        super(VQA_RAD_Dataset, self).__init__()

        np.random.seed(seed)  # Reset the seed so all runs are the same.
        self.imgpath = imgpath
        self.views = None

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

        ####### VQA answer & labels ########

        # Normalize word (Set pathologies to be answer in Med-VQA dataset)
        self.pathologies = self.csv["answer"].apply(
            lambda x: unifier.datasets.utils.normalize_word(str(x)).lower()
        )

        # Construct labels
        labels = self.pathologies.apply(lambda x: self.ans2label[x]).tolist()
        self.labels = np.asarray(labels).T
        self.labels = self.labels.astype(np.float32)

    def string(self):
        return (
            self.__class__.__name__
            + " num_samples={} views={} transforms={}".format(
                len(self), self.views, self.transform
            )
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = {}
        sample["idx"] = idx
        sample["lab"] = self.labels[idx]

        imgid = self.csv["image_name"].iloc[idx]
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
