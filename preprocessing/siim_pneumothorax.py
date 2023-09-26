import os
import pandas as pd
import numpy as np
import pydicom
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from utils.constants import *


def rle2mask(rle, width, height):
    mask = np.zeros(width * height)
    array = np.asarray([int(x) for x in rle.split()])
    starts = array[0::2]
    lengths = array[1::2]

    current_position = 0
    for index, start in enumerate(starts):  # binary segmentation mask
        current_position += start
        mask[current_position : current_position + lengths[index]] = 255
        current_position += lengths[index]

    return mask.reshape(width, height)


def convert_dcm_to_filetype(file_path):
    """Reads a dcm file and returns the file as jpg/png"""

    # read the dcm file
    ds = pydicom.read_file(file_path)
    im = ds.pixel_array.astype(float)

    # rescale image
    rescaled_image = (np.maximum(im, 0) / im.max()) * 255
    final_image = np.uint8(rescaled_image)
    im = Image.fromarray(final_image)

    return im


def save_img(file_path, split, rle, img_type="png"):
    """Reads a dcm file and saves the image and its mask as jpg/png"""

    filename = file_path.split("/")[-1]

    im = convert_dcm_to_filetype(file_path)

    mask = np.zeros([1024, 1024])
    if rle != " -1":
        mask += rle2mask(rle, PNEUMOTHORAX_IMG_SIZE, PNEUMOTHORAX_IMG_SIZE)
    mask = mask >= 1  # changes to binary mask
    mask = Image.fromarray(mask)

    img_dir = os.path.join(PNEUMOTHORAX_DATA_DIR, "images", split)
    ann_dir = os.path.join(PNEUMOTHORAX_DATA_DIR, "annotations", split)

    if img_type == "png":
        im.save(os.path.join(img_dir, filename.replace(".dcm", ".png")))
        mask.save(os.path.join(ann_dir, filename.replace(".dcm", ".png")))
    else:
        im.save(os.path.join(img_dir, filename.replace(".dcm", ".jpg")))
        mask.save(os.path.join(ann_dir, filename.replace(".dcm", ".jpg")))

    return file_path


def preprocess_pneumothorax_data(test_fac=0.15):
    try:
        df = pd.read_csv(PNEUMOTHORAX_ORIGINAL_TRAIN_CSV)
    except:
        raise Exception(
            "Please make sure the the SIIM Pneumothorax dataset is \
            stored at {PNEUMOTHORAX_DATA_DIR}"
        )

    # get image paths
    os.listdir(PNEUMOTHORAX_ORIGINAL_IMG_DIR)
    img_paths = {}
    for subdir, dirs, files in tqdm(os.walk(PNEUMOTHORAX_ORIGINAL_IMG_DIR)):
        for f in files:
            if "dcm" in f:
                # remove dcm
                file_id = f[:-4]
                img_paths[file_id] = os.path.join(subdir, f)

    # no encoded pixels mean healthy
    df["Label"] = df.apply(
        lambda x: 0.0 if x[" EncodedPixels"] == " -1" else 1.0, axis=1
    )
    df["Path"] = df["ImageId"].apply(lambda x: img_paths[x])

    # split data
    train_df, test_val_df = train_test_split(df, test_size=test_fac * 2, random_state=0)
    test_df, valid_df = train_test_split(test_val_df, test_size=0.5, random_state=0)

    # print split info
    print(f"Number of train samples: {len(train_df)}")
    print(train_df["Label"].value_counts())
    print(f"Number of valid samples: {len(valid_df)}")
    print(valid_df["Label"].value_counts())
    print(f"Number of test samples: {len(test_df)}")
    print(test_df["Label"].value_counts())

    train_df.to_csv(PNEUMOTHORAX_TRAIN_CSV)
    valid_df.to_csv(PNEUMOTHORAX_VALID_CSV)
    test_df.to_csv(PNEUMOTHORAX_TEST_CSV)


def preprocess_mmseg_siim(data_pct=1, seed=42, train_only=False):
    """Convert images to .png format and move them to images/ann dir for mmseg experiments"""

    # Set seed for reproducibility
    np.random.seed(seed)

    path_col = "Path"
    rle_col = " EncodedPixels"

    train_df = pd.read_csv(PNEUMOTHORAX_TRAIN_CSV)

    # undersample for train_df
    train_df["class"] = train_df[" EncodedPixels"].apply(lambda x: x != " -1")
    df_neg = train_df[train_df["class"] == False]
    df_pos = train_df[train_df["class"] == True]
    n_pos = df_pos["ImageId"].nunique()
    neg_series = df_neg["ImageId"].unique()
    neg_series_selected = np.random.choice(neg_series, size=n_pos, replace=False)
    df_neg = df_neg[df_neg["ImageId"].isin(neg_series_selected)]
    train_df = pd.concat([df_pos, df_neg])

    if data_pct != 1:
        ids = train_df["ImageId"].unique()
        n_samples = int(len(ids) * data_pct)
        series_selected = np.random.choice(ids, size=n_samples, replace=False)
        train_df = train_df[train_df["ImageId"].isin(series_selected)]

    train_df.apply(
        lambda x: save_img(x[path_col], "training", x[rle_col]),
        axis=1,
    )

    if train_only:  # early stopping
        print("Only train images added")
        return

    valid_df = pd.read_csv(PNEUMOTHORAX_VALID_CSV)
    test_df = pd.read_csv(PNEUMOTHORAX_TEST_CSV)

    valid_df.apply(
        lambda x: save_img(x[path_col], "validation", x[rle_col]),
        axis=1,
    )
    test_df.apply(
        lambda x: save_img(x[path_col], "test", x[rle_col]),
        axis=1,
    )


if __name__ == "__main__":
    preprocess_pneumothorax_data()
    # preprocess_mmseg_siim(train_only=True)
