import argparse
import glob
import json
import os
import subprocess


def run_trec_eval(qrel_path, run_path):
    cmd = [
        "python",
        "-m",
        "pyserini.eval.trec_eval",
        "-c",
        "-m",
        "recall.5,10",
        "-m",
        "ndcg_cut.5,10",
        qrel_path,
        run_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error running trec_eval for {run_path}: {e.stderr}")
        return None


def parse_trec_eval_output(output):
    metrics = {}
    for line in output.splitlines():
        parts = line.split()
        if len(parts) == 3:
            metric_name = parts[0]
            # Use the middle field if it's there, but trec_eval usually has 'metric qid value' or 'metric all value'
            # Pyserini trec_eval output format: metric_name qid value
            metric_name = parts[0]
            value = float(parts[2])
            metrics[metric_name] = value
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate TREC runfiles using pyserini.eval.trec_eval"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Directory containing .trec files",
    )
    parser.add_argument(
        "--qrels_dir", type=str, required=True, help="Directory containing qrel files"
    )
    args = parser.parse_args()

    results_dir = os.path.abspath(args.results_dir)
    qrels_dir = os.path.abspath(args.qrels_dir)

    trec_files = glob.glob(os.path.join(results_dir, "*.trec"))

    if not trec_files:
        print(f"No .trec files found in {results_dir}")
        return

    for run_path in trec_files:
        run_filename = os.path.basename(run_path)
        # Example run_filename: mmeb-visdoc-ViDoRe_arxivqa.VLM2Vec-V2.0.trec
        # Task part: ViDoRe_arxivqa

        if not run_filename.startswith("mmeb-visdoc-"):
            print(f"Skipping {run_filename}, doesn't follow expected pattern.")
            continue

        task_part = run_filename.replace("mmeb-visdoc-", "").split(".")[0]

        # Look for matching qrel file
        # Expected qrel filename: qrels.mmeb-visdoc-<TASK>.test.txt
        qrel_pattern = os.path.join(
            qrels_dir, f"qrels.mmeb-visdoc-{task_part}.test.txt"
        )
        qrel_files = glob.glob(qrel_pattern)

        if not qrel_files:
            # Try a fuzzy match if exact task name doesn't work
            print(f"Exact match failed for {task_part}, trying fuzzy search...")
            qrel_files = glob.glob(os.path.join(qrels_dir, f"*qrel*{task_part}*"))

        if not qrel_files:
            print(
                f"Could not find qrel file for task: {task_part} (Runfile: {run_filename})"
            )
            continue

        qrel_path = qrel_files[0]
        print(f"Evaluating {run_filename} using {os.path.basename(qrel_path)}...")

        output = run_trec_eval(qrel_path, run_path)
        if output:
            metrics = parse_trec_eval_output(output)
            # Use run name to avoid collisions in the same directory
            run_name = run_filename.replace(".trec", "")
            metrics_file = os.path.join(results_dir, run_name + ".metrics.json")
            with open(metrics_file, "w") as f:
                json.dump(metrics, f, indent=4)
            print(f"Saved metrics to {metrics_file}")


if __name__ == "__main__":
    main()
