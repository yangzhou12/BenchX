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
MIMIC_CXR_MASTER_CSV = MIMIC_CXR_ROOT_DIR / "master.csv"
MIMIC_CXR_PATH_COL = "Path"
MIMIC_CXR_SPLIT_COL = "split"