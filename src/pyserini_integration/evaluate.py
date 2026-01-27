import argparse
import os
import sys
import re
import subprocess
import glob


def find_pyserini_jar():
    # Try to find the pyserini jar in common locations
    potential_paths = [
        "/u6/s8sharif/pyserini/pyserini/resources/jars/*.jar",
        os.path.join(
            os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            ),
            "pyserini/pyserini/resources/jars/*.jar",
        ),
    ]
    for path in potential_paths:
        jars = glob.glob(path)
        if jars:
            return jars[0]
    return None


def run_trec_eval(qrels, run, metrics):
    jar_path = find_pyserini_jar()
    if not jar_path:
        print("Error: Could not find Pyserini jar file for evaluation.")
        return

    cmd = ["java", "-cp", jar_path, "trec_eval", "-c"]
    for metric in metrics.split():
        cmd.extend(["-m", metric])
    cmd.extend([qrels, run])

    print(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.stderr:
        print(result.stderr)
    print(result.stdout)


def main():
    parser = argparse.ArgumentParser(
        description="Run trec_eval for a given TREC run file."
    )
    parser.add_argument("run", type=str, help="Path to TREC run file.")
    parser.add_argument(
        "--qrels",
        type=str,
        help="Path to qrels file. If not provided, it will be inferred from the run filename.",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default="ndcg_cut.10 recall.10,100,1000",
        help="Comma-separated metrics to evaluate (default: ndcg_cut.10,recall.10,recall.100,recall.1000).",
    )

    args = parser.parse_args()

    if not os.path.exists(args.run):
        print(f"Error: Run file not found at {args.run}")
        sys.exit(1)

    qrels_path = args.qrels
    if not qrels_path:
        # Try to infer qrels path from run filename
        run_basename = os.path.basename(args.run)
        match = re.search(r"mmeb-visdoc-(ViDoRe_[\w\-\._]+)", run_basename)
        if match:
            dataset_name = match.group(1).split(".")[0]
            # Try a few common locations for qrels
            potential_base_dirs = [
                os.path.dirname(os.path.dirname(os.path.abspath(args.run))),
                "/u6/s8sharif/MMEB-V2/visdoc-tasks/pyserini_v1",
            ]

            for base_dir in potential_base_dirs:
                potential_qrels = [
                    os.path.join(
                        base_dir, "qrels", f"qrels.mmeb-visdoc-{dataset_name}.test.txt"
                    ),
                    # VisRAG test is using the train qrels
                    os.path.join(
                        base_dir, "qrels", f"qrels.mmeb-visdoc-{dataset_name}.train.txt"
                    ),
                ]
                for pq in potential_qrels:
                    if os.path.exists(pq):
                        qrels_path = pq
                        break
                if qrels_path:
                    break

        if not qrels_path:
            print(
                "Error: Qrels file could not be inferred. Please provide it using --qrels."
            )
            sys.exit(1)

    if not os.path.exists(qrels_path):
        print(f"Error: Qrels file not found at {qrels_path}")
        sys.exit(1)

    print(f"Run file: {args.run}")
    print(f"Qrels file: {qrels_path}")

    run_trec_eval(qrels_path, args.run, args.metrics)


if __name__ == "__main__":
    main()
