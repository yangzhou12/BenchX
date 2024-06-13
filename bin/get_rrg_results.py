import os
import json
import pandas as pd
from collections import defaultdict
import argparse
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Process experiment results.")
    parser.add_argument("--dataset", default="iuxray", help="Dataset name")
    args = parser.parse_args()

    # Specify the experiment path
    exp_path = os.path.join("experiments/rrg", args.dataset)
    random_seeds = [42, 0, 1]

    # List to store results for each method
    method_results = []

    # Iterate through all files in the directory
    for methodname in os.listdir(exp_path):
        # Dictionary to store accumulated scores for each metric
        average_scores = defaultdict(float)
        all_scores = defaultdict(float)
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
                        average_scores[metric] += value / len(random_seeds)
                        if metric not in all_scores:
                            all_scores[metric] = [value]
                        else:
                            all_scores[metric].append(value)

                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON in file {filename}: {e}")

        # # Store results for each method
        # method_results.append({
        #     "Method": methodname,
        #     **average_scores
        # })

        report_scores = {}
        for key, value in all_scores.items():
            if value:
                value = np.array(value)
                report_scores.update({"mean_"+key: value.mean(), "std_"+key: value.std()})
        # Store results for each method
        method_results.append({
            "Method": methodname.split("/")[-1],
            **report_scores
        })

    # Create a DataFrame from the results
    df = pd.DataFrame(method_results)

    # Save the DataFrame to a CSV file
    csv_file_path = os.path.join(exp_path, f"{args.dataset}_results.csv")
    df.to_csv(csv_file_path, index=False)

    print(f"Average metrics saved to {csv_file_path}")

if __name__ == "__main__":
    main()
