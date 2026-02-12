import argparse
import glob
import json
import os

import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate evaluation results into an Excel file"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Directory containing .metrics.json files",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="MMEB_aggregated_results.xlsx",
        help="Name of the output Excel file",
    )
    args = parser.parse_args()

    results_dir = os.path.abspath(args.results_dir)
    output_file = os.path.join(results_dir, args.output_file)

    metrics_files = glob.glob(os.path.join(results_dir, "*.metrics.json"))

    if not metrics_files:
        print(f"No .metrics.json files found in {results_dir}")
        return

    # Data structure to hold results: { model_name: [ { 'Task': task, 'recall_5': val, ... }, ... ] }
    model_data = {}

    for file_path in metrics_files:
        filename = os.path.basename(file_path)
        # Expected pattern: mmeb-visdoc-<TASK>.<MODEL>.trec.metrics.json
        # Based on examples: mmeb-visdoc-VisRAG_MP-DocVQA.VLM2Vec-V2.0.metrics.json

        try:
            # Remove the prefix
            core_name = filename.replace("mmeb-visdoc-", "")
            # Remove the suffix
            core_name = core_name.replace(".trec.metrics.json", "")

            parts = core_name.split(".")
            if len(parts) < 2:
                print(f"Skipping {filename}, cannot parse task and model.")
                continue

            task = parts[0]
            model = ".".join(parts[1:])  # Everything else is the model

            with open(file_path, "r") as f:
                metrics = json.load(f)

            row = {"Task": task}
            row.update(metrics)

            if model not in model_data:
                model_data[model] = []
            model_data[model].append(row)

        except Exception as e:
            print(f"Error processing {filename}: {e}")

    if not model_data:
        print("No valid data found to aggregate.")
        return

    # Create Excel writer
    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        for model, rows in model_data.items():
            df = pd.DataFrame(rows)
            # Sort by Task name for better readability
            df = df.sort_values(by="Task")

            # Sanitize sheet name (Excel sheets have a 31 char limit and restricted chars)
            sheet_name = (
                model[:31]
                .replace("[", "")
                .replace("]", "")
                .replace("*", "")
                .replace("?", "")
                .replace(":", "")
                .replace("/", "")
                .replace("\\", "")
            )

            df.to_excel(writer, sheet_name=sheet_name, index=False)
            print(f"Added sheet for model: {model}")

    print(f"Aggregated results saved to {output_file}")


if __name__ == "__main__":
    main()
