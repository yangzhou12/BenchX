import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

import pydicom as dicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
import matplotlib.pylab as plt
from PIL import Image
from tqdm import tqdm

# Download link: https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge/data

data_path = "/source/path/to/RSNA/"
processed_datapath = "/target/path/to/RSNA/"

imgpath = os.path.join(data_path, "stage_2_train_images")
csv_path = os.path.join(data_path, "stage_2_train_labels.csv")
raw_csv = pd.read_csv(csv_path)
csv = raw_csv.groupby("patientId").first()

output_image_dir = os.path.join(processed_datapath, "images")
if not os.path.exists(output_image_dir):
    os.makedirs(output_image_dir)

output_mask_dir = os.path.join(processed_datapath, "masks")
if not os.path.exists(output_mask_dir):
    os.makedirs(output_mask_dir)

desired_size = 512
def preprocess_rsna_data():
    for image_id, row in tqdm(csv.iterrows(), total=len(csv)):
        # image_id = row.patientId
        img_path = os.path.join(imgpath, image_id + ".dcm")

        # Convert dicom to jpg and resize
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

        label_ids = raw_csv[raw_csv["patientId"] == image_id].index
        for i in label_ids:
            if ~np.isnan(raw_csv.iloc[i].x):
                mask_label = raw_csv.iloc[i]
                xywh = np.asarray([mask_label.x, mask_label.y, mask_label.width, mask_label.height])
                xywh = xywh * ratio
                raw_csv.loc[i, ['x','y','width','height']] = xywh.tolist()
        final_image.save(os.path.join(output_image_dir, image_id + ".png"), 'PNG')

    output_csvpath = os.path.join(processed_datapath, "rsna_labels.csv")
    raw_csv.to_csv(output_csvpath)

def generate_rsna_masks():
    output_csvpath = os.path.join(processed_datapath, "rsna_labels.csv")
    raw_csv = pd.read_csv(output_csvpath)
    csv = raw_csv.groupby("patientId").first()
    for patient_id, _ in tqdm(csv.iterrows(), total=len(csv)):
        rows = raw_csv[raw_csv["patientId"] == patient_id]
        mask = np.zeros([desired_size, desired_size]).astype(np.uint8)
        for index, row in rows.iterrows():
            if row.Target == 1:
                xywh = np.asarray([row.x, row.y, row.width, row.height])
                xywh = xywh.astype(int)
                mask[xywh[1] : xywh[1] + xywh[3], xywh[0] : xywh[0] + xywh[2]] = 1
                # final_mask = Image.fromarray(mask * 255)
                # final_mask.save("vis.jpg", 'JPEG')
        final_mask = Image.fromarray(mask)
        final_mask.save(os.path.join(output_mask_dir, patient_id + ".png"), 'PNG')

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

def split_seg_dataset(seed):
    output_csvpath = os.path.join(processed_datapath, "rsna_labels.csv")
    raw_csv = pd.read_csv(output_csvpath)
    csv = raw_csv.groupby("patientId").first()

    x = csv.index.to_list()
    y = csv["Target"].to_list()

    x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.3, stratify=y, random_state=seed)
    x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=seed)

    # Further split the training set into subsets with 1% and 10% of the training samples
    x_train_1, _ = train_test_split(x_train, test_size=0.99, stratify=y_train, random_state=seed)
    x_train_10, _ = train_test_split(x_train, test_size=0.90, stratify=y_train, random_state=seed)

    save_anno(x_train, processed_datapath + '/train.txt')
    save_anno(x_train_1, processed_datapath + '/train_1.txt')
    save_anno(x_train_10, processed_datapath + '/train_10.txt')
    save_anno(x_val, processed_datapath + '/val.txt')
    save_anno(x_test, processed_datapath + '/test.txt')

if __name__ == "__main__":
    preprocess_rsna_data()
    generate_rsna_masks()
    split_seg_dataset(seed=42)