import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

import cv2
import matplotlib.pylab as plt
from PIL import Image
from tqdm import tqdm

# Download link: https://www.kaggle.com/datasets/raddar/foreign-objects-in-chest-xrays

data_path = "/source/path/to/Object-CXR/"
processed_datapath = "/target/path/to/Object-CXR/"

output_image_dir = os.path.join(processed_datapath, "images")
if not os.path.exists(output_image_dir):
    os.makedirs(output_image_dir)

output_mask_dir = os.path.join(processed_datapath, "masks")
if not os.path.exists(output_mask_dir):
    os.makedirs(output_mask_dir)

# train_csv_path = os.path.join(processed_datapath, "train.csv")
# dev_csv_path = os.path.join(processed_datapath, "dev.csv")
# test_csv_path = os.path.join(processed_datapath, "test.csv")
# train_csv = pd.read_csv(train_csv_path)
# dev_csv = pd.read_csv(dev_csv_path)
# test_csv = pd.read_csv(test_csv_path)

# train_csv["split"] = "train"
# dev_csv["split"] = "val"
# test_csv["split"] = "test"

# csv = pd.concat([train_csv, dev_csv, test_csv])
# csv.reset_index(drop=True, inplace=True)
# csv.drop('Unnamed: 0', axis=1, inplace=True)
# csv.to_csv(os.path.join(processed_datapath, f"object_cxr_labels.csv"), index=False)

desired_size = 512
def preprocess_data():
    data_split = ['train', 'dev', 'test']
    all_csv = pd.DataFrame()
    for split in data_split:
        image_path = os.path.join(data_path, split)
        csv_path = os.path.join(data_path, f"{split}.csv")
        csv = pd.read_csv(csv_path)

        image_name_list = []
        for i, row in tqdm(csv.iterrows(), total=len(csv)):
            csv.reset_index()
            if split == 'test':
                # Test set does not has image_name. Use image_path instead.
                image_name = row.image_path.split("/")[1]
            else:
                image_name = row.image_name
            image_name_list.append(image_name)
            
            img_path = os.path.join(image_path, image_name)
            img = Image.open(img_path).convert("RGB")
            old_size = img.size
            # ratio = float(desired_size)/max(old_size)
            # new_size = tuple([int(x*ratio) for x in old_size])
            ratio = [float(desired_size)/size for size in old_size] 
            new_size = (desired_size, desired_size)
            final_image = img.resize(new_size, Image.Resampling.LANCZOS)

            annotation = row.annotation
            if not pd.isna(annotation):
                # mask = np.zeros(new_size, dtype=np.uint8)
                resized_annotation = []
                for anno in annotation.split(";"):
                    anno = np.fromstring(anno, sep=" ", dtype=int)
                    anno_type = anno[0]
                    coordinate = (anno[1:].reshape((-1,2)) * ratio).astype(int)
                    coordinate = coordinate.reshape(-1)
                    # Convert the coordinates as string following the original data structure
                    coord_as_string = str(anno_type) + ' ' + ' '.join(map(str, coordinate))
                    resized_annotation.append(coord_as_string)
                # Combine multiple boxes separated by ";"
                resized_annotation = ";".join(resized_annotation)
                csv.loc[i].annotation = resized_annotation
            final_image.save(os.path.join(output_image_dir, image_name.replace(".jpg", ".png")), 'PNG')
        if split == 'test':
            # Add image_name into csv
            csv["image_name"] = image_name_list

        csv["split"] = split
        all_csv = pd.concat([all_csv, csv])

    all_csv.reset_index(drop=True, inplace=True)
    all_csv.to_csv(os.path.join(processed_datapath, f"object_cxr_labels.csv"), index=False)

def get_mask(row, image):
    annotation = row.annotation
    mask = np.zeros_like(image, dtype=np.uint8)
    if row.has_masks:
        for anno in annotation.split(";"):
            anno = np.fromstring(anno, sep=" ", dtype=int)
            anno_type = anno[0]
            anno = anno[1:]
            if anno_type != 2:
                # bbox annotation (xywh)
                cv2.rectangle(mask, (anno[0], anno[1]), (anno[2], anno[3]), color=1, thickness=cv2.FILLED)
            else:
                # polygon annotation 
                poly = anno.reshape((-1,2))
                cv2.fillPoly(mask, [poly], color=1)
                # cv2.imwrite("foo.png", mask*255)
    return mask

def generate_masks():
    output_csvpath = os.path.join(processed_datapath, "object_cxr_labels.csv")
    csv = pd.read_csv(output_csvpath)
    csv["has_masks"] = ~csv["annotation"].isnull()
    for index, row in tqdm(csv.iterrows(), total=len(csv)):
        image_name = row.image_name
        image_path = os.path.join(output_image_dir, image_name)
        image = np.array(Image.open(image_path).convert("L"))
        mask = get_mask(row, image)
        final_mask = Image.fromarray(mask)
        final_mask.save(os.path.join(output_mask_dir, image_name.replace(".jpg", ".png")), 'PNG')

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
    output_csvpath = os.path.join(processed_datapath, "object_cxr_labels.csv")
    csv = pd.read_csv(output_csvpath)
    csv["has_masks"] = ~csv["annotation"].isnull()

    x_train = csv[csv["split"] == "train"]["image_name"].to_list()
    x_val = csv[csv["split"] == "dev"]["image_name"].to_list()
    x_test = csv[csv["split"] == "test"]["image_name"].to_list()

    train_labels = csv[csv["split"] == "train"]["has_masks"].to_list()
    _, x_train_1 = train_test_split(x_train, test_size=0.01, stratify=train_labels, random_state=seed)
    _, x_train_10 = train_test_split(x_train, test_size=0.1, stratify=train_labels, random_state=seed)

    save_anno(x_train, processed_datapath + '/train.txt')
    save_anno(x_train_1, processed_datapath + '/train_1.txt')
    save_anno(x_train_10, processed_datapath + '/train_10.txt')
    save_anno(x_val, processed_datapath + '/val.txt')
    save_anno(x_test, processed_datapath + '/test.txt')

if __name__ == "__main__":
    preprocess_data()
    generate_masks()
    split_seg_dataset(seed=42)