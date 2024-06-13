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

# Download link: https://www.kaggle.com/datasets/vbookshelf/pneumothorax-chest-xray-images-and-masks

# Specify data paths and desired preprocessed image size
data_path = "/source/path/to/SIIM/"
processed_datapath = "/target/path/to/SIIM/"

output_image_dir = os.path.join(processed_datapath, "images")
if not os.path.exists(output_image_dir):
    os.makedirs(output_image_dir)
output_mask_dir = os.path.join(processed_datapath, "masks")
if not os.path.exists(output_mask_dir):
    os.makedirs(output_mask_dir)

def preprocess_pneumothorax_data():
    desired_size = 512
    all_imgs = sorted(glob.glob(data_path + '/png_images/*.png'))
    for img_path in tqdm(all_imgs):
        img_name = img_path.split("/")[-1]
        mask_path = os.path.join(data_path, "png_masks", img_name)
        # mask_path = os.path.join(output_mask_dir, img_name)
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        # resize and save image in output dir
        old_size = img.size
        ratio = float(desired_size) / max(old_size)
        new_size = tuple([int(x*ratio) for x in old_size])
        final_image = img.resize(new_size, Image.Resampling.LANCZOS)
        final_image.save(os.path.join(output_image_dir, img_name), 'PNG')

        final_mask = mask.resize(new_size, Image.Resampling.LANCZOS)
        final_mask = np.array(final_mask) > 128
        final_mask = Image.fromarray(final_mask)
        final_mask.save(os.path.join(output_mask_dir, img_name), 'PNG')


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
    train_pos_imgs = sorted(glob.glob(processed_datapath + '/images/*train_1*.png'))
    train_neg_imgs = sorted(glob.glob(processed_datapath + '/images/*train_0*.png'))

    x = train_pos_imgs + train_neg_imgs
    y = [1] * len(train_pos_imgs) + [0] * len(train_neg_imgs) 
    x_test = sorted(glob.glob(processed_datapath + '/images/*test*.png'))
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=len(x_test), stratify=y, random_state=seed)
    
    # Further split the training set into subsets with 1% and 10% of the training samples
    x_train_1, _ = train_test_split(x_train, test_size=0.99, stratify=y_train, random_state=seed)
    x_train_10, _ = train_test_split(x_train, test_size=0.90, stratify=y_train, random_state=seed)

    save_anno(x_train, processed_datapath + '/train.txt')
    save_anno(x_train_1, processed_datapath + '/train_1.txt')
    save_anno(x_train_10, processed_datapath + '/train_10.txt')
    save_anno(x_val, processed_datapath + '/val.txt')
    save_anno(x_test, processed_datapath + '/test.txt')

    csv_train = pd.read_csv(os.path.join(data_path, "stage_1_train_images.csv"))
    csv_test = pd.read_csv(os.path.join(data_path, "stage_1_test_images.csv"))
    csv = pd.concat([csv_train, csv_test], axis=0)
    csv.to_csv(os.path.join(processed_datapath, "siim_labels.csv"), index=False)

if __name__ == "__main__":
    preprocess_pneumothorax_data()
    split_seg_dataset(seed=42)
