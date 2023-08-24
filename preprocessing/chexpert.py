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
        filter_cols = lambda x: (x["Atelectasis"] == index[0]) \
                        & (x["Cardiomegaly"] == index[1]) \
                        & (x["Consolidation"] == index[2]) \
                        & (x["Edema"] == index[3]) \
                        & (x["Pleural Effusion"] == index[4]) \
                        & (x["Enlarged Cardiomediastinum"] == index[5]) \
                        & (x["Lung Lesion"] == index[7]) \
                        & (x["Lung Opacity"] == index[8]) \
                        & (x["Pneumonia"] == index[9]) \
                        & (x["Pneumothorax"] == index[10]) \
                        & (x["Pleural Other"] == index[11]) \
                        & (x["Fracture"] == index[12]) \
                        & (x["Support Devices"] == index[13])
        df_task = df[filter_cols(df)]
        df_task = df_task.sample(n)

        task_dfs.append(df_task)

    df_200 = pd.concat(task_dfs)
    df_200['Path'] = df_200['Path'].apply(lambda x: CHEXPERT_DATA_DIR / "/".join(x.split("/")[1:]))

    # Merge sampled reports from MIMIC 5x200 to CheXpert-5x200
    mimic_5x200 = pd.read_csv(MIMIC_CXR_5X200, index_col=0).reset_index()
    mimic_5x200 = mimic_5x200[['findings', 'impression']]
    df_200 = pd.concat([df_200.reset_index(), mimic_5x200], axis=1).set_index('index')
    
    df_200.to_csv(CHEXPERT_5x200)



if __name__ == "__main__":
    preprocess_chexpert_5x200_data(n=200)