import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from collections import Counter

import pydicom as dicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
from PIL import Image
from tqdm import tqdm
from ensemble_boxes import weighted_boxes_fusion

# Download link: https://physionet.org/content/vindr-cxr/1.0.0/

data_path = "/source/path/to/VinDr_CXR/"
processed_datapath = "/target/path/to/VinDr_CXR/"

output_image_dir = os.path.join(processed_datapath, "images")
if not os.path.exists(output_image_dir):
    os.makedirs(output_image_dir)

output_mask_dir = os.path.join(processed_datapath, "masks")
if not os.path.exists(output_mask_dir):
    os.makedirs(output_mask_dir)

desired_size = 512
labels = ['No finding', 'Aortic enlargement', 'Atelectasis',
       'Calcification', 'Cardiomegaly', 'Clavicle fracture', 'Consolidation',
       'Edema', 'Emphysema', 'Enlarged PA', 'ILD', 'Infiltration',
       'Lung Opacity', 'Lung cavity', 'Lung cyst', 'Mediastinal shift',
       'Nodule/Mass', 'Pleural effusion', 'Pleural thickening', 'Pneumothorax',
       'Pulmonary fibrosis', 'Rib fracture', 'Other lesion', 'COPD',
       'Lung tumor', 'Pneumonia', 'Tuberculosis', 'Other diseases']

class_name_to_id = {}
for i, each_label in enumerate(labels):
    class_id = i - 1  # starts with -1
    class_name = each_label
    class_name_to_id[class_name] = class_id


def combine_labels():
    print("================= COMBINE LABELS =====================")
    train_csv_path = os.path.join(data_path, "annotations", "annotations_train.csv")
    test_csv_path = os.path.join(data_path, "annotations", "annotations_test.csv")
    train_csv = pd.read_csv(train_csv_path)
    test_csv = pd.read_csv(test_csv_path)

    train_csv["split"] = "train"
    test_csv["split"] = "test"

    csv = pd.concat([train_csv, test_csv])
    csv.reset_index(drop=True, inplace=True)
    csv.to_csv(os.path.join(processed_datapath, f"vindr_labels.csv"), index=False)


def preprocess_vindr_data():
    print("================= PREPROCESSING =====================")
    csv_path = os.path.join(processed_datapath, f"vindr_labels.csv") 
    csv = pd.read_csv(csv_path)

    for split in ['train', 'test']:
        print(f"Processing {split} data...")
        split_csv = csv[csv['split'] == split]
        image_ids = pd.unique(split_csv['image_id'])
        base_path = os.path.join(data_path, split)

        for image_id in tqdm(image_ids, total=len(image_ids)):
            img_path = os.path.join(base_path, image_id + ".dicom")

            dc_image = dicom.dcmread(img_path, force=True)

            image_array = dc_image.pixel_array.astype(float)
            image_array = apply_voi_lut(image_array, dc_image)
            # depending on this value, X-ray may look inverted - fix that:
            if dc_image.PhotometricInterpretation == "MONOCHROME1":
                image_array = np.amax(image_array) - image_array

            scaled_image = (np.maximum(image_array, 0) / image_array.max()) * 255.0
            scaled_image = np.uint8(scaled_image)

            final_image = Image.fromarray(scaled_image).convert("RGB")
            old_size = final_image.size
            ratio = float(desired_size)/max(old_size)
            new_size = tuple([int(x*ratio) for x in old_size])
            final_image = final_image.resize(new_size, Image.Resampling.LANCZOS)

            final_image.save(os.path.join(output_image_dir, image_id + ".png"))


def generate_vindr_masks():
    print("================= GENERATING MASKS =====================")
    output_csvpath = os.path.join(processed_datapath, "vindr_labels.csv")
    raw_csv = pd.read_csv(output_csvpath)
    image_ids = pd.unique(raw_csv['image_id'])
    for image_id in tqdm(image_ids, total=len(image_ids)):
        rows = raw_csv[raw_csv["image_id"] == image_id]
        mask = np.zeros([desired_size, desired_size]).astype(np.uint8)
        for index, row in rows.iterrows():
            if row.class_name != 'No finding':
                xywh = np.asarray([row.x_min, row.y_min, row.x_max, row.y_max])
                xywh = xywh.astype(int)
                mask[xywh[1] : xywh[1] + xywh[3], xywh[0] : xywh[0] + xywh[2]] = 1
        final_mask = Image.fromarray(mask)
        final_mask.save(os.path.join(output_mask_dir, image_id + ".png"), 'PNG')


def save_anno(img_list, file_path, remove_suffix=False):
    if remove_suffix:
        img_list = [
            img_path.split('/')[-1] for img_path in img_list
        ]
        img_list = [
            '.'.join(img_path.split('.')[:-1]) for img_path in img_list
        ]
    with open(file_path, 'w') as file_:
        for x in list(img_list):
            file_.write(x + '\n')


def split_dataset(seed):
    print("================= SPLITTING DATASET =====================")
    csvpath = os.path.join(processed_datapath, "vindr_labels.csv")
    rawcsv = pd.read_csv(csvpath)

    csv = rawcsv.groupby("image_id", group_keys=True).first().reset_index()
    csv["class_name"] = rawcsv.groupby("image_id")["class_name"].apply(lambda x: "|".join(np.unique(x))).tolist()
    csv["has_masks"] = csv.class_name != "No finding"

    train_csv = csv[csv["split"] == "train"]
    test_csv = csv[csv["split"] == "test"]

    x, y = train_csv["image_id"].tolist(), train_csv["has_masks"].tolist()
    x_test, y_test = test_csv["image_id"].tolist(), test_csv["has_masks"].tolist()

    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=len(x_test), stratify=y, random_state=seed)

    # Further split the training set into subsets with 1% and 10% of the training samples
    x_train_1, _ = train_test_split(x_train, test_size=0.99, stratify=y_train, random_state=seed)
    x_train_10, _ = train_test_split(x_train, test_size=0.90, stratify=y_train, random_state=seed)

    save_anno(x_train, processed_datapath + '/train.txt')
    save_anno(x_train_1, processed_datapath + '/train_1.txt')
    save_anno(x_train_10, processed_datapath + '/train_10.txt')
    save_anno(x_val, processed_datapath + '/val.txt')
    save_anno(x_test, processed_datapath + '/test.txt')

    csv = pd.concat([train_csv, test_csv], ignore_index=True)
    csv.to_csv(os.path.join(processed_datapath, "vindr_labels.csv"), index=False)


if __name__ == "__main__":
    combine_labels()
    preprocess_vindr_data()
    generate_vindr_masks()
    split_dataset(seed=42)
