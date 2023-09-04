import argparse
import pandas as pd
import re
from sklearn.model_selection import GroupShuffleSplit

import sys
from pathlib import Path

path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))
from constants import *


def parse_report(report, key):
    matches = list(
        re.compile(r"(?P<title>[ [A-Z\]()]+):").finditer(report)
    )  # Remove carat and newline flag

    parsed_report = {}

    for match, next_match in zip(matches, matches[1:] + [None]):
        start = match.end()
        end = next_match and next_match.start()

        title = match.group("title")
        title = title.strip()

        paragraph = report[start:end]
        paragraph = re.sub(r"\s{2,}", " ", paragraph)
        paragraph = paragraph.strip()

        parsed_report[title] = paragraph.replace("\n", "\\n")

    if key in parsed_report:
        if parsed_report[key] != "":
            return parsed_report[key]

    return " "


def main():
    df = pd.read_csv(MIMIC_CXR_TRAINING_CSV, index_col=0)
    print(f"Original number of samples: {len(df)}")

    # Extract dicom ID
    df["dicom_id"] = df["image_path"].apply(lambda x: x.split("/")[4][:-4])

    # Rename column image_path to Path
    df = df.rename(columns={"image_path": "Path"})

    # Add metadata information
    metadata_df = pd.read_csv(MIMIC_CXR_META_CSV)
    df = df.merge(metadata_df, how="inner", on="dicom_id")

    # Filter out frontal views only
    df = df[df["ViewPosition"].isin(["PA", "AP"])]
    print(f"Number of samples after filtering for frontal views: {len(df)}")

    # Extract findings and impressions
    df[MIMIC_CXR_REPORT_COL] = df["report_content"].apply(
        lambda x: " ".join(
            [parse_report(x, "FINDINGS"), parse_report(x, "IMPRESSION")]
        ).strip()
    )

    # Remove columns with empty report
    df = df[df[MIMIC_CXR_REPORT_COL] != ""]
    print(f"Number of samples after filtering out empty reports: {len(df)}")

    # Combine labels from CheXpert labeller
    master_df = df.loc[
        :, ["Path", "dicom_id", "subject_id", "study_id", MIMIC_CXR_REPORT_COL]
    ]
    chexpert_labels_df = pd.read_csv(MIMIC_CXR_CHEX_CSV)
    master_df = master_df.merge(
        chexpert_labels_df, how="inner", on=["subject_id", "study_id"]
    )
    master_df = master_df.fillna(0)

    # Generate train-test-val splits in the ratio of 70:15:15
    # Ensure there are no overlaps of patients between splits
    train_valtest_splitter = GroupShuffleSplit(test_size=0.3, random_state=42)
    train_valtest_split = train_valtest_splitter.split(
        master_df, groups=master_df["subject_id"]
    )
    train_inds, valtest_inds = next(train_valtest_split)

    train_df = master_df.iloc[train_inds]
    valtest_df = master_df.iloc[valtest_inds]

    val_test_splitter = GroupShuffleSplit(test_size=0.5, random_state=42)
    val_test_split = val_test_splitter.split(
        valtest_df, groups=valtest_df["subject_id"]
    )
    val_inds, test_inds = next(val_test_split)

    val_df = valtest_df.iloc[val_inds]
    test_df = valtest_df.iloc[test_inds]

    # Generate csv files
    train_df.to_csv(MIMIC_CXR_TRAIN_CSV, index=False)
    print(f"Number of train samples: {len(train_df)}")
    val_df.to_csv(MIMIC_CXR_VALID_CSV, index=False)
    print(f"Number of validation samples: {len(val_df)}")
    test_df.to_csv(MIMIC_CXR_TEST_CSV, index=False)
    print(f"Number of test samples: {len(test_df)}")


if __name__ == "__main__":
    # 1. Filter out specified views
    # 2. Extract and combine findings and impression sections of report
    # 3. Use 70:15:15 train-test-valid split

    main()
