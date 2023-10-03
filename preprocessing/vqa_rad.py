import re
import pandas as pd

from utils.constants import *


def preprocess_vqarad_data():
    df = pd.read_json(VQA_RAD_ORIGINAL_TRAIN_JSON)

    # split the data into official train-test splits
    train_df = df[df["phrase_type"].isin(["freeform", "para"])]
    test_df = df[df["phrase_type"].isin(["test_freeform", "test_para"])]

    # keep only image-question-answer triplets
    train_df = train_df[["image_name", "question", "answer"]]
    test_df = test_df[["image_name", "question", "answer"]]

    # drop duplicate image-question-answer triplets
    train_df = train_df.drop_duplicates(ignore_index=True)
    test_df = test_df.drop_duplicates(ignore_index=True)

    # drop common image-question-answer triplets across splits
    train_df = train_df[~train_df.apply(tuple, 1).isin(test_df.apply(tuple, 1))]
    train_df = train_df.reset_index(drop=True)

    # perform some basic data cleaning/normalization
    f = lambda x: re.sub(' +', ' ', str(x).lower()).replace(" ?", "?").strip()
    train_df["question"] = train_df["question"].apply(f)
    test_df["question"] = test_df["question"].apply(f)
    train_df["answer"] = train_df["answer"].apply(f)
    test_df["answer"] = test_df["answer"].apply(f)

    # save the metadata
    train_df.to_csv(VQA_RAD_TRAIN_CSV, index=False)
    test_df.to_csv(VQA_RAD_TEST_CSV, index=False)

    # Split test data into open and closed questions
    close_df = test_df[test_df["answer_type"] == "CLOSED"]
    close_df = close_df[["image_name", "question", "answer"]]

    open_df = test_df[test_df["answer_type"] == "OPEN"]
    open_df = open_df[["image_name", "question", "answer"]]
    
    close_df = close_df.drop_duplicates(ignore_index=True)
    open_df = open_df.drop_duplicates(ignore_index=True)

    f = lambda x: re.sub(' +', ' ', str(x).lower()).replace(" ?", "?").strip()
    close_df["question"] = close_df["question"].apply(f)
    close_df["answer"] = close_df["answer"].apply(f)
    open_df["question"] = open_df["question"].apply(f)
    open_df["answer"] = open_df["answer"].apply(f)

    close_df.to_csv(VQA_RAD_TEST_CLOSE_CSV)
    open_df.to_csv(VQA_RAD_TEST_OPEN_CSV)


if __name__ == "__main__":
    preprocess_vqarad_data()