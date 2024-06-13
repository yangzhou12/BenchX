import os
import numpy as np
import pandas as pd
import pydicom
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from pydicom.pixel_data_handlers.util import apply_voi_lut
from PIL import Image
import cv2
import glob
import shutil
import json
import ast

# Download link: https://www.kaggle.com/datasets/vbookshelf/tbx11k-simplified

# Specify data paths and desired preprocessed image size
data_path = "/source/path/to/TBX11K/"
processed_datapath = "/target/path/to/TBX11K/"
desired_size = 512

output_image_dir = os.path.join(processed_datapath, "images")
if not os.path.exists(output_image_dir):
    os.makedirs(output_image_dir)
output_mask_dir = os.path.join(processed_datapath, "masks")
if not os.path.exists(output_mask_dir):
    os.makedirs(output_mask_dir)

def process_data():
    source_directory = os.path.join(data_path, "images")
    files_to_copy = os.listdir(source_directory)
    # Iterate over the files and copy them to the destination directory
    for file_name in files_to_copy:
        source_path = os.path.join(source_directory, file_name)
        destination_path = os.path.join(output_image_dir, file_name)
        
        shutil.copy2(source_path, destination_path)

def generate_masks():
    output_csvpath = os.path.join(data_path, "data.csv")
    raw_csv = pd.read_csv(output_csvpath)
    csv = raw_csv.groupby("fname").first()
    for image_name, row in tqdm(csv.iterrows(), total=len(csv)):
        rows = raw_csv[raw_csv["fname"] == image_name]
        mask = np.zeros([desired_size, desired_size]).astype(np.uint8)
        for index, row in rows.iterrows():
            if row.target == "tb":
                bbox = ast.literal_eval(row.bbox)
                xywh = np.asarray([bbox["xmin"], bbox["ymin"], bbox["width"], bbox["height"]])
                xywh = xywh.astype(int)
                mask[xywh[1] : xywh[1] + xywh[3], xywh[0] : xywh[0] + xywh[2]] = 1
                # final_mask = Image.fromarray(mask * 255)
                # final_mask.save("vis.jpg", 'JPEG')
        final_mask = Image.fromarray(mask)
        final_mask.save(os.path.join(output_mask_dir, image_name), 'PNG')


def save_anno(img_list, file_path, remove_suffix=True):
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
    output_csvpath = os.path.join(data_path, "data.csv")
    raw_csv = pd.read_csv(output_csvpath)
    csv = raw_csv.groupby("fname").first()

    x = csv.index.to_list()
    y = csv["target"].to_list()

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

    csv.to_csv(os.path.join(processed_datapath, "tbx11k_labels.csv"), index=False)

if __name__ == "__main__":
    process_data()
    generate_masks()
    split_seg_dataset(seed=42)
