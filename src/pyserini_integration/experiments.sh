#!/bin/bash

# Configuration
BASE_DIR="./MMEB-V2/visdoc-tasks/pyserini" # <-- Change this to your base directory
VISDOC_YAML="visdoc.yaml"
PYTHON_SCRIPT="quick_start_demo.py"
EVAL_SCRIPT="evaluate_results.py"
AGGREGATE_SCRIPT="aggregate_results.py"

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR" || exit 1

# Initialize Conda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mmebv2_exec_install

# 1. Run the experiment (Encoding and Searching)
echo "Running experiment with $PYTHON_SCRIPT..."
python "$PYTHON_SCRIPT" \
    --base_dir "$BASE_DIR" \
    --visdoc_yaml "$VISDOC_YAML"

conda deactivate
conda activate rankllm-2

# 2. Run Evaluation
echo "Running evaluation with $EVAL_SCRIPT..."
python "$EVAL_SCRIPT" \
    --results_dir "$BASE_DIR/results" \
    --qrels_dir "$BASE_DIR/qrels"

# 3. Aggregate Results
echo "Aggregating results with $AGGREGATE_SCRIPT..."
python "$AGGREGATE_SCRIPT" \
    --results_dir "$BASE_DIR/results"

echo "All steps finished."
