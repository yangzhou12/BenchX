import os
import pandas as pd
import argparse
import csv
import re
from os.path import exists
from tqdm import tqdm
from ..constants import *
from .utils import *


def extract_mimic_text(args):
    if args.include_exceptions:
        custom_section_names, custom_indices = custom_mimic_cxr_rules()

    # get all higher up folders (p00, p01, etc)
    p_grp_folders = os.listdir(MIMIC_REPORTS_DIR)
    p_grp_folders = [p for p in p_grp_folders
                     if p.startswith('p') and len(p) == 3]
    p_grp_folders.sort()

    # patient_studies will hold the text for use in NLP labeling
    patient_studies = []

    # study_sections will have an element for each study
    # this element will be a list, each element having text for a specific section
    study_sections = []
    for p_grp in p_grp_folders:
        # get patient folders, usually around ~6k per group folder
        cxr_path = MIMIC_REPORTS_DIR / p_grp
        p_folders = os.listdir(cxr_path)
        p_folders = [p for p in p_folders if p.startswith('p')]
        p_folders.sort()

        # For each patient in this grouping folder
        for p in tqdm(p_folders):
            patient_path = cxr_path / p

            # get the filename for all their free-text reports
            studies = os.listdir(patient_path)
            studies = [s for s in studies
                       if s.endswith('.txt') and s.startswith('s')]

            for s in studies:
                # load in the free-text report
                with open(patient_path / s, 'r') as fp:
                    text = ''.join(fp.readlines())

                # get study string name without the txt extension
                s_stem = s[:-4]

                if args.include_exceptions:
                    # custom rules for some poorly formatted reports
                    if s_stem in custom_indices:
                        idx = custom_indices[s_stem]
                        patient_studies.append([s_stem, text[idx[0]:idx[1]]])
                        continue

                # split text into sections
                sections, section_names, section_idx = section_text(args, text)

                if args.include_exceptions:
                    # check to see if this has mis-named sections
                    # e.g. sometimes the impression is in the comparison section
                    if s_stem in custom_section_names:
                        sn = custom_section_names[s_stem]
                        idx = list_rindex(section_names, sn)
                        patient_studies.append([s_stem, sections[idx].strip()])
                        continue

                # grab the *last* section with the given title
                # prioritizes impression > findings, etc.

                # "last_paragraph" is text up to the end of the report
                # many reports are simple, and have a single section
                # header followed by a few paragraphs
                # these paragraphs are grouped into section "last_paragraph"

                # note also comparison seems unusual but if no other sections
                # exist the radiologist has usually written the report
                # in the comparison section

                idx = -1

                for sn in ('impression', 'findings',
                           'last_paragraph', 'comparison'):
                    if sn in section_names: #if report contains one of 4 above sections
                        idx = list_rindex(section_names, sn) #take the last matching section index
                        break

                if idx == -1:
                    # we didn't find any sections we can use :(
                    patient_studies.append([s_stem, ''])
                    print(f'no impression/findings: {patient_path / s}')
                else:
                    # store the text of the conclusion section
                    patient_studies.append([s_stem, sections[idx].strip()])

                study_sectioned = [s_stem]
                for sn in ('impression', 'findings',
                           'last_paragraph', 'comparison'):
                    if sn in section_names:
                        idx = list_rindex(section_names, sn)
                        study_sectioned.append(sections[idx].strip())
                    else:
                        study_sectioned.append(None)
                study_sections.append(study_sectioned)

    # patient_studies
    # write distinct files to facilitate modular processing
    if len(patient_studies) > 0:
        # write out a single CSV with the sections
        with open(MIMIC_CXR_TEXT_CSV, 'w') as fp:
            csvwriter = csv.writer(fp)
            # write header
            csvwriter.writerow(['study', 'impression', 'findings',
                                'last_paragraph', 'comparison'])
            for row in study_sections:
                csvwriter.writerow(row)

        # write all the reports out to a single file
        with open(MIMIC_CXR_ROOT_DIR / 'mimic_cxr_sections.csv', 'w') as fp:
            csvwriter = csv.writer(fp)
            for row in patient_studies:
                csvwriter.writerow(row)

def section_text(args, text):
    """Splits text into sections.

    Assumes text is in a radiology report format, e.g.:

        COMPARISON:  Chest radiograph dated XYZ.

        IMPRESSION:  ABC...

    Given text like this, it will output text from each section, 
    where the section type is determined by the all cap s header.

    Returns a three element tuple:
        sections - list containing the text of each section
        section_names - a normalized version of the section name
        section_idx - list of start indices of the text in the section
    """
    p_section = re.compile(
        r'\n ([A-Z ()/,-]+):\s', re.DOTALL)

    sections = list()
    section_names = list()
    section_idx = list()

    idx = 0
    s = p_section.search(text, idx)

    if s:
        sections.append(text[0:s.start(1)])
        section_names.append('preamble')
        section_idx.append(0)

        while s:
            current_section = s.group(1).lower()
            # get the start of the text for this section
            idx_start = s.end()
            # skip past the first newline to avoid some bad parses
            idx_skip = text[idx_start:].find('\n')
            if idx_skip == -1:
                idx_skip = 0

            s = p_section.search(text, idx_start + idx_skip)

            if s is None:
                idx_end = len(text)
            else:
                idx_end = s.start()

            sections.append(text[idx_start:idx_end])
            section_names.append(current_section)
            section_idx.append(idx_start)

    else:
        sections.append(text)
        section_names.append('full report')
        section_idx.append(0)

    if args.normalize: #Default false
        section_names = normalize_section_names(section_names)

    # remove empty sections
    # this handles when the report starts with a finding-like statement
    #  .. but this statement is not a section, more like a report title
    #  e.g. p10/p10103318/s57408307
    #    CHEST, PA LATERAL:
    #
    #    INDICATION:   This is the actual section ....
    # it also helps when there are multiple findings sections
    # usually one is empty
    for i in reversed(range(len(section_names))):
        if section_names[i] in ('impression', 'findings'):
            if sections[i].strip() == '':
                sections.pop(i)
                section_names.pop(i)
                section_idx.pop(i)

    if ('impression' not in section_names) & ('findings' not in section_names):
        # create a new section for the final paragraph
        if '\n \n' in sections[-1]:
            sections.append('\n \n'.join(sections[-1].split('\n \n')[1:]))
            sections[-2] = sections[-2].split('\n \n')[0]
            section_names.append('last_paragraph')
            section_idx.append(section_idx[-1] + len(sections[-2]))

    return sections, section_names, section_idx



def main(args):
    if (not exists(MIMIC_CXR_TEXT_CSV)) or args.overwrite_text:
        extract_mimic_text(args)

    metadata_df = pd.read_csv(MIMIC_CXR_META_CSV)
    metadata_df = metadata_df[["dicom_id", "subject_id",
                               "study_id", "ViewPosition"]].astype(str)
    metadata_df["study_id"] = metadata_df["study_id"].apply(lambda x: "s" + x)
    
    #Filter out different radiograph views according to setting
    if args.view == 'frontal': #default
        # Only keep frontal images
        metadata_df = metadata_df[metadata_df["ViewPosition"].isin(["PA", "AP"])]
    elif args.view == 'lateral':
        # Only keep lateral images
        metadata_df = metadata_df[~metadata_df["ViewPosition"].isin(["PA", "AP"])]

    # Organise report content
    text_df = pd.read_csv(MIMIC_CXR_TEXT_CSV)
    text_df.dropna(subset=["impression", "findings"], how="all", inplace=True)
    text_df = text_df[["study", "impression", "findings"]]
    text_df.rename(columns={"study": "study_id"}, inplace=True)

    # Use official train-test splits
    split_df = pd.read_csv(MIMIC_CXR_SPLIT_CSV)
    split_df = split_df.astype(str)
    split_df["study_id"] = split_df["study_id"].apply(lambda x: "s"+x)
    # FIX: removed merging of validate and test into test.
    # split_df["split"] = split_df["split"].apply(
    #     lambda x: "valid" if x == "validate" or x == "test" else x)

    chexpert_df = pd.read_csv(MIMIC_CXR_CHEX_CSV)
    chexpert_df[["subject_id", "study_id"]] = chexpert_df[[
        "subject_id", "study_id"]].astype(str)
    chexpert_df["study_id"] = chexpert_df["study_id"].apply(lambda x: "s"+x)

    master_df = pd.merge(metadata_df, text_df, on="study_id", how="left")
    master_df = pd.merge(master_df, split_df, on=["dicom_id", "subject_id", "study_id"], how="inner")
    master_df.dropna(subset=["impression", "findings"], how="all", inplace=True)
    
    n = len(master_df)
    master_data = master_df.values

    root_dir = str(MIMIC_CXR_CHEX_CSV).split("/")[-1] + "/files"
    path_list = []
    for i in range(n):
        row = master_data[i]
        file_path = "%s/p%s/p%s/%s/%s.jpg" % (root_dir, str(
            row[1])[:2], str(row[1]), str(row[2]), str(row[0]))
        path_list.append(file_path)
        
    master_df.insert(loc=0, column="Path", value=path_list)

    # TODO: Create labeled data df
    labeled_data_df = pd.merge(master_df, chexpert_df, on=[
                               "subject_id", "study_id"], how="inner")
    labeled_data_df.drop(["dicom_id", "subject_id", "study_id",
                          "impression", "findings"], axis=1, inplace=True)

    train_df = labeled_data_df.loc[labeled_data_df["split"] == "train"]
    train_df.to_csv(MIMIC_CXR_TRAIN_CSV, index=False)
    valid_df = labeled_data_df.loc[labeled_data_df["split"] == "valid"]
    valid_df.to_csv(MIMIC_CXR_TEST_CSV, index=False)

    # master_df.drop(["dicom_id", "subject_id", "study_id"],
    #                axis=1, inplace=True)

    # Fill nan in text
    master_df[["impression"]] = master_df[["impression"]].fillna(" ")
    master_df[["findings"]] = master_df[["findings"]].fillna(" ")
    master_df.to_csv(MIMIC_CXR_MASTER_CSV, index=False)



if __name__ == "__main__":
    # 1. Filter out specified views
    # 2. Extract both findings AND impression sections of report
    # 3. Use official train-test splits

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--normalize', type=bool, default=False, help='normalize section names, e.g. impressions -> impression')
    parser.add_argument('--include_exceptions', type=bool, default=False, help='handle misformatted report sections')
    parser.add_argument('--overwrite_text', type=bool, default=False, help='overwrite report text file')
    parser.add_argument('--view', type=str, default='frontal', help="frontal/lateral, or empty string for both")

    args = parser.parse_args()

    main(args)