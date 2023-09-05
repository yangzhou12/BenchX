import os
import pandas as pd
import json

import sys
from pathlib import Path
path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))
from constants import *


def preprocess_slake_data():
    data = {
        "train": [],
        "valid": [],
        "test": []
    }

    for split, file in zip(list(data.keys()), ["train.json", "validate.json", "test.json"]):
        with open(f"{SLAKE_DATA_DIR}/{file}", "r") as fp:
            samples = json.load(fp)
            for sample in samples:
                if sample["q_lang"] != "en":
                    continue
                img_path = os.path.join(SLAKE_IMG_DIR, sample["img_name"])
                qid = sample["qid"]
                question = sample["question"]
                answer = sample["answer"]
                answer_type = sample["answer_type"]
                data[split].append({
                    "img_path": img_path,
                    "qid": qid,
                    "question": question,
                    "answer": answer,
                    "answer_type": answer_type
                })
    
    train_df = pd.DataFrame(data["train"])
    valid_df = pd.DataFrame(data["valid"])
    test_df = pd.DataFrame(data["test"])

    print(f"Number of train samples: {len(train_df)}")
    print(f"Number of valid samples: {len(valid_df)}")
    print(f"Number of test samples: {len(test_df)}")

    train_df.to_csv(SLAKE_TRAIN_CSV)
    valid_df.to_csv(SLAKE_VALID_CSV)
    test_df.to_csv(SLAKE_TEST_CSV)


if __name__ == "__main__":
    preprocess_slake_data()