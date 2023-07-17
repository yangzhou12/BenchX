from pathlib import Path

# #############################################
#          MIMIC-CXR-JPG constants            #
# #############################################

# MIMIC user constants
MIMIC_CXR_ROOT_DIR = Path("/home/faith/datasets/mimic_512") #change this to your root data directory for MIMIC-CXR
MIMIC_CXR_META_CSV = MIMIC_CXR_ROOT_DIR / "mimic-cxr-2.0.0-metadata.csv"
MIMIC_CXR_SPLIT_CSV = MIMIC_CXR_ROOT_DIR / "mimic-cxr-2.0.0-split.csv"
MIMIC_CXR_CHEX_CSV = MIMIC_CXR_ROOT_DIR / "mimic-cxr-2.0.0-chexpert.csv"
MIMIC_REPORTS_DIR = MIMIC_CXR_ROOT_DIR / "files"

# MIMIC fixed constants
MIMIC_CXR_TEXT_CSV = MIMIC_CXR_ROOT_DIR / "mimic_cxr_sectioned.csv" #created after running preprocessing code
MIMIC_CXR_TRAIN_CSV = MIMIC_CXR_ROOT_DIR / "train.csv" #train split
MIMIC_CXR_VALID_CSV = MIMIC_CXR_ROOT_DIR / "valid.csv" #valid split
MIMIC_CXR_TEST_CSV = MIMIC_CXR_ROOT_DIR / "test.csv" #test split
MIMIC_CXR_MASTER_CSV = MIMIC_CXR_ROOT_DIR / "master_data.csv"
MIMIC_CXR_PATH_COL = "Path"
MIMIC_CXR_SPLIT_COL = "split"


# #############################################
#          RSNA Pneumonia constants           #
# #############################################

# RSNA Pneumonia
PNEUMONIA_ROOT_DIR = Path("/home/faith/datasets/rsna-pneumonia/")
PNEUMONIA_ORIGINAL_TRAIN_CSV = PNEUMONIA_ROOT_DIR / "stage_2_train_labels.csv"
PNEUMONIA_IMG_DIR = PNEUMONIA_ROOT_DIR / "stage_2_train_images"
PNEUMONIA_TRAIN_CSV = PNEUMONIA_ROOT_DIR / "train.csv"
PNEUMONIA_VALID_CSV = PNEUMONIA_ROOT_DIR / "val.csv"
PNEUMONIA_TEST_CSV = PNEUMONIA_ROOT_DIR / "test.csv"
PNEUMONIA_TRAIN_PCT = 0.7
