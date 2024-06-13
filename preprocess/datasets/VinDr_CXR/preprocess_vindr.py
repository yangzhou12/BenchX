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

processed_datapath = "/target/path/to/Vindr_CXR/"

# Combine all the labels
# train_csv_path = os.path.join(processed_datapath, "annotations", "annotations_train.csv")
# test_csv_path = os.path.join(processed_datapath, "annotations", "annotations_test.csv")
# train_csv = pd.read_csv(train_csv_path)
# test_csv = pd.read_csv(test_csv_path)

# train_csv["split"] = "train"
# test_csv["split"] = "test"

# csv = pd.concat([train_csv, test_csv])
# csv.reset_index(drop=True, inplace=True)
# csv.drop('Unnamed: 0', axis=1, inplace=True)
# csv.to_csv(os.path.join(processed_datapath, f"vindr_labels.csv"), index=False)

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

def fuse_vindr_label():
    output_csvpath = os.path.join(processed_datapath, "annotations", "vindr_labels.csv")
    raw_csv = pd.read_csv(output_csvpath)
    raw_csv['class_id'] = raw_csv['class_name'].map(lambda x:class_name_to_id[x])
    csv = raw_csv.groupby("image_id").first()
    fuse_csv = []

    for patient_id, row in tqdm(csv.iterrows()):
        boxes_list = []
        scores_list = []
        labels_list = []
        weights = []
        
        boxes_single = []
        labels_single = []

        annot = raw_csv[raw_csv["image_id"] == patient_id]
        cls_ids = annot['class_id'].unique().tolist()
        count_dict = Counter(annot['class_id'].tolist())

        if cls_ids == [-1]:
            annot_dict = row.to_dict()
            annot_dict["image_id"] = patient_id
            fuse_csv.append(annot_dict)
            continue
            # pd.DataFrame(fuse_csv)
        elif -1 in cls_ids:
            cls_ids.pop(-1)

        for cid in cls_ids:
            ## Performing Fusing operation only for multiple bboxes with the same label
            if count_dict[cid]==1:
                labels_single.append(cid)
                bbox = annot[annot["class_id"] == cid][['x_min', 'y_min', 'x_max', 'y_max']].to_numpy().squeeze()
                boxes_single.append(bbox.tolist())

            else:
                cls_list = annot[annot["class_id"] == cid]['class_id'].tolist()
                labels_list.append(cls_list)
                bbox = annot[annot["class_id"] == cid][['x_min', 'y_min', 'x_max', 'y_max']].to_numpy()
                
                ## Normalizing Bbox by Image Width and Height
                bbox = bbox/desired_size
                # (scaled_image.shape[1], scaled_image.shape[0], scaled_image.shape[1], scaled_image.shape[0])
                bbox = np.clip(bbox, 0, 1)
                boxes_list.append(bbox.tolist())
                scores_list.append(np.ones(len(cls_list)).tolist())
                weights.append(1)
        
        ## Perform WBF
        iou_thr = 0.5
        skip_box_thr = 0.0001
        boxes, scores, box_labels = weighted_boxes_fusion(boxes_list=boxes_list, scores_list=scores_list,
                                                    labels_list=labels_list, weights=weights,
                                                    iou_thr=iou_thr, skip_box_thr=skip_box_thr)
        
        boxes = boxes*desired_size
        boxes = boxes.round(1).tolist()
        box_labels = box_labels.astype(int).tolist()
        boxes.extend(boxes_single)
        box_labels.extend(labels_single)
        
        for box, label in zip(boxes, box_labels):
            x_min, y_min, x_max, y_max = (box[0], box[1], box[2], box[3])
            area = round((x_max-x_min)*(y_max-y_min),1)
            bbox =[
                    round(x_min, 1),
                    round(y_min, 1),
                    round((x_max-x_min), 1),
                    round((y_max-y_min), 1)
                    ]

def generate_vindr_masks():
    output_csvpath = os.path.join(processed_datapath, "annotations", "vindr_labels.csv")
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

def split_dataset(seed):
    csvpath = os.path.join(processed_datapath, "annotations", "vindr_labels.csv")
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
    fuse_vindr_label()
    generate_vindr_masks()
    split_dataset(seed=42)