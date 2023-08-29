import pandas as pd
import numpy as np
from pathlib import Path

import sys
path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))
from constants import *


def preprocess_chexpert_5x200_data(n=200):
    
    df = pd.read_csv(CHEXPERT_ORIGINAL_TRAIN_CSV)
    df = df.fillna(0)
    df = df[df["Frontal/Lateral"] == "Frontal"] # Filter frontal views only

    task_dfs = []
    for i, t in enumerate(CHEXPERT_COMPETITION_TASKS):
        index = np.zeros(14)
        index[i] = 1
        df_task = df[
            (df["Atelectasis"] == index[0])
            & (df["Cardiomegaly"] == index[1])
            & (df["Consolidation"] == index[2])
            & (df["Edema"] == index[3])
            & (df["Pleural Effusion"] == index[4])
            & (df["Enlarged Cardiomediastinum"] == index[5])
            & (df["Lung Lesion"] == index[7])
            & (df["Lung Opacity"] == index[8])
            & (df["Pneumonia"] == index[9])
            & (df["Pneumothorax"] == index[10])
            & (df["Pleural Other"] == index[11])
            & (df["Fracture"] == index[12])
            & (df["Support Devices"] == index[13])
        ]
        df_task = df_task.sample(n)
        task_dfs.append(df_task)

    df_200 = pd.concat(task_dfs)
    df_200['Path'] = df_200['Path'].apply(lambda x: CHEXPERT_DATA_DIR / "/".join(x.split("/")[1:]))
    df_200.to_csv(CHEXPERT_5x200)


if __name__ == "__main__":
    preprocess_chexpert_5x200_data(n=60)