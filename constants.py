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
MIMIC_CXR_5X200 = MIMIC_CXR_ROOT_DIR / "mimic_5x200.csv"
MIMIC_CXR_PATH_COL = "Path"
MIMIC_CXR_SPLIT_COL = "split"
MIMIC_CXR_REPORT_COL = "Report"


# #############################################
#          RSNA Pneumonia constants           #
# #############################################

# RSNA Pneumonia
PNEUMONIA_ROOT_DIR = Path("/home/faith/datasets/rsna_pneumonia/")  #change this to your root data directory for RSNA Pneumonia
PNEUMONIA_ORIGINAL_TRAIN_CSV = PNEUMONIA_ROOT_DIR / "stage_2_train_labels.csv"
PNEUMONIA_IMG_DIR = PNEUMONIA_ROOT_DIR / "stage_2_train_images"
PNEUMONIA_TRAIN_CSV = PNEUMONIA_ROOT_DIR / "train.csv"
PNEUMONIA_VALID_CSV = PNEUMONIA_ROOT_DIR / "val.csv"
PNEUMONIA_TEST_CSV = PNEUMONIA_ROOT_DIR / "test.csv"
PNEUMONIA_TRAIN_PCT = 0.

PNEUMONIA_TASKS = ['Pneumonia']

PNEUMONIA_CLASS_PROMPTS = {
    'Pneumonia': {
        'adjective': ['round', 'early', 'focal', 'multifocal', 'small', ''],
        'subtype': ['bacterial', 'viral', 'mycoplasma', ''],
        "location": [
            "at the mid lung zone",
            "at the upper lung zone",
            "at the right lung zone",
            "at the left lung zone",
            "at the lung bases",
            "at the right lung base",
            "at the left lung base",
            "at the bilateral lung bases",
            "at the left lower lobe",
            "at the right lower lobe",
            "at the left middle lobe",
            "at the right middle lobe",
            ""
        ]
    }
}


# ############################################
#          NIH Chest X-Ray constants         #
# ############################################

NIH_CHEST_ROOT_DIR = Path("/data/faith/datasets/nih_chest_xray") #change this to your root data directory for NIH Chest X-ray

NIH_CHEST_XRAY_TASKS =  ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
                         'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']


# #############################################
#         CheXpert (small) constants          #
# #############################################

# CheXpert class prompts
CHEXPERT_CLASS_PROMPTS = {
    "Atelectasis": {
        "severity": ["", "mild", "minimal"],
        "subtype": [
            "subsegmental atelectasis",
            "linear atelectasis",
            "trace atelectasis",
            "bibasilar atelectasis",
            "retrocardiac atelectasis",
            "bandlike atelectasis",
            "residual atelectasis",
        ],
        "location": [
            "at the mid lung zone",
            "at the upper lung zone",
            "at the right lung zone",
            "at the left lung zone",
            "at the lung bases",
            "at the right lung base",
            "at the left lung base",
            "at the bilateral lung bases",
            "at the left lower lobe",
            "at the right lower lobe",
        ],
    },
    "Cardiomegaly": {
        "severity": [""],
        "subtype": [
            "cardiac silhouette size is upper limits of normal",
            "cardiomegaly which is unchanged",
            "mildly prominent cardiac silhouette",
            "portable view of the chest demonstrates stable cardiomegaly",
            "portable view of the chest demonstrates mild cardiomegaly",
            "persistent severe cardiomegaly",
            "heart size is borderline enlarged",
            "cardiomegaly unchanged",
            "heart size is at the upper limits of normal",
            "redemonstration of cardiomegaly",
            "ap erect chest radiograph demonstrates the heart size is the upper limits of normal",
            "cardiac silhouette size is mildly enlarged",
            "mildly enlarged cardiac silhouette, likely left ventricular enlargement. other chambers are less prominent",
            "heart size remains at mildly enlarged",
            "persistent cardiomegaly with prominent upper lobe vessels",
        ],
        "location": [""],
    },
    "Consolidation": {
        "severity": ["", "increased", "improved", "apperance of"],
        "subtype": [
            "bilateral consolidation",
            "reticular consolidation",
            "retrocardiac consolidation",
            "patchy consolidation",
            "airspace consolidation",
            "partial consolidation",
        ],
        "location": [
            "at the lower lung zone",
            "at the upper lung zone",
            "at the left lower lobe",
            "at the right lower lobe",
            "at the left upper lobe",
            "at the right uppper lobe",
            "at the right lung base",
            "at the left lung base",
        ],
    },
    "Edema": {
        "severity": [
            "",
            "mild",
            "improvement in",
            "presistent",
            "moderate",
            "decreased",
        ],
        "subtype": [
            "pulmonary edema",
            "trace interstitial edema",
            "pulmonary interstitial edema",
        ],
        "location": [""],
    },
    "Pleural Effusion": {
        "severity": ["", "small", "stable", "large", "decreased", "increased"],
        "location": ["left", "right", "tiny"],
        "subtype": [
            "bilateral pleural effusion",
            "subpulmonic pleural effusion",
            "bilateral pleural effusion",
        ],
    },
}

# CheXpert competition tasks (5x200)
CHEXPERT_COMPETITION_TASKS = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Pleural Effusion",
]

# CheXpert constants
CHEXPERT_DATA_DIR = Path("/data/faith/datasets/CheXpert-v1.0-small") #change this to your data directory for CheXpert small dataset
CHEXPERT_ORIGINAL_TRAIN_CSV = CHEXPERT_DATA_DIR / "train.csv"
CHEXPERT_TEST_CSV = (
    CHEXPERT_DATA_DIR / "valid.csv"
)  # using expert-labelled validation set as test set (test set label hidden)
CHEXPERT_5x200 = CHEXPERT_DATA_DIR / "chexpert_5x200.csv"


# #############################################
#         SIIM Pneumothorax constants         #
# #############################################

PNEUMOTHORAX_DATA_DIR = Path("/data/faith/datasets/siim-acr-pneumothorax") #change this to your root data directory for SIIM-ACR Pneumothorax
PNEUMOTHORAX_ORIGINAL_TRAIN_CSV = PNEUMOTHORAX_DATA_DIR / "train-rle.csv"
PNEUMOTHORAX_TRAIN_CSV = PNEUMOTHORAX_DATA_DIR / "train.csv"
PNEUMOTHORAX_VALID_CSV = PNEUMOTHORAX_DATA_DIR / "valid.csv"
PNEUMOTHORAX_TEST_CSV = PNEUMOTHORAX_DATA_DIR / "test.csv"
PNEUMOTHORAX_IMG_DIR = PNEUMOTHORAX_DATA_DIR / "images"
PNEUMOTHORAX_IMG_SIZE = 1024


# ############################################
#          Downstream Task Constants         #
# ############################################

# Classification constants
DATASET_CLASSES = {
    'rsna_pneumonia': PNEUMONIA_TASKS,
    'nih_chest_xray': NIH_CHEST_XRAY_TASKS
}