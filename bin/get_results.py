import os
import json
import pandas as pd
from collections import defaultdict
import argparse
import glob
from tqdm import tqdm
import numpy as np

# test_method = ["GLoRIA", "MedCLIP_R50", "MedCLIP_ViT", "MedKLIP", "MGCA_R50", "MGCA_ViT", "MRM", "REFERS", 
#                "Our_ConVIRT", "Our_GLoRIA", "Our_MedKLIP", "Our_MFLAG", "Our_MGCA_R50", "Our_MGCA_ViT", "Our_MRM", "Our_REFERS"]

# test_method = ["Our_ConVIRT", "GLoRIA", "Our_GLoRIA", "MedCLIP_R50", "MedCLIP_ViT", "MedKLIP", "Our_MedKLIP", "Our_MFLAG", 
#                "MGCA_R50", "Our_MGCA_R50", "MGCA_ViT", "Our_MGCA_ViT", "MRM", "Our_MRM", "REFERS", "Our_REFERS"]

# test_method = ["Our_ConVIRT", "GLoRIA", "Our_GLoRIA", "MedCLIP_R50", "MedCLIP_ViT", "MedKLIP", "Our_MedKLIP", "Our_MFLAG", 
#                "MGCA_R50", "Our_MGCA_R50", "MGCA_ViT", "Our_MGCA_ViT", "MRM", "Our_MRM", "REFERS"]

test_method = ["convirt", "gloria", "medclip_rn50", "medclip_vit", "medklip", "mflag", "mgca_rn50", "mgca_vit", "mrm", "refers"]

def main():
    parser = argparse.ArgumentParser(description="Process experiment results.")
    parser.add_argument("--dataset", default="covidx", help="Dataset name")
    # parser.add_argument("--split", default="_1", help="Dataset name")
    args = parser.parse_args()

    # Specify the experiment path
    exp_path = os.path.join("experiments/classification", args.dataset)
    random_seeds = [42, 0, 1]
    # random_seeds = [0, 1]
    # random_seeds = [42, 0]
    # random_seeds = [42]
    splits = ["_1", "_10", ""]
    # splits = ["_1"]
    # splits = [""]
    # splits = ["_1", "_10"]

    for split in splits:
        # List to store results for each method
        method_results = []
        # Iterate through all files in the directory
        for name in tqdm(test_method):

            methodname = name + split

            # Dictionary to store accumulated scores for each metric
            average_scores = defaultdict(float)
            all_scores = {"multilabel_auroc": [], "multiclass_auroc": [], "multiclass_f1": []}
            for seed in random_seeds:
                filename = f"test_{seed}_metrics.txt"

                # Create the full file path
                file_path = os.path.join(exp_path, methodname, filename)

                # Read the content of the file
                with open(file_path, "r") as file:
                    try:
                        # Load the JSON data from the file
                        data = json.load(file)

                        # Access the relevant information
                        scores = data["scores"]

                        # Accumulate scores for each metric
                        for metric, value in scores.items():
                            if metric in ["multilabel_auroc", "multiclass_auroc", "multiclass_f1"]:
                                average_scores[metric] += value / len(random_seeds)
                                all_scores[metric].append(value)

                    except json.JSONDecodeError as e:
                        print(f"Error decoding JSON in file {filename}: {e}")

            report_scores = {}
            for key, value in all_scores.items():
                if value and key == "multiclass_f1":
                    value = np.array(value)
                    if key == "multiclass_auroc":
                        report_scores.update({"mean_"+key: round(value.mean() * 100, 1), "std_"+key: round(value.std() * 100, 2)})
                    else:
                        report_scores.update({"mean_"+key: round(value.mean(), 1), "std_"+key: round(value.std(), 2)})
            # Store results for each method
            method_results.append({
                "Method": methodname.split("/")[-1],
                **report_scores
            })

        # Create a DataFrame from the results
        df = pd.DataFrame(method_results)

        # Save the DataFrame to a CSV file
        csv_file_path = os.path.join(exp_path, f"{args.dataset}{split}_results.csv")
        df.to_csv(csv_file_path, index=False)

        print(f"Average metrics saved to {csv_file_path}")

if __name__ == "__main__":
    main()
