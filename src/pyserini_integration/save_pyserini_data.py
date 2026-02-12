import argparse
import glob
import hashlib
import os
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
import yaml
from tqdm import tqdm


def get_suffix(dataset_name):
    # VisRAG-RET-test datasets only have train parquet files
    if "visrag" in dataset_name.lower():
        return "train"
    return "test"


def find_parquets(directory):
    """
    Finds all parquet files in the given directory.
    Returns a list of file paths sorted by name.
    """
    pattern = os.path.join(directory, "*.parquet")
    files = glob.glob(pattern)
    # VisRAG testdatasets only have train parquet files, so we should not filter them out.
    if not "visrag" in directory.lower():
        files = [f for f in files if "train" not in f]
    print(f"Found {len(files)} parquet files in {directory}")
    print(f"Files: {files}")
    return sorted(files)


def convert_parquet_to_pyserini(
    data_basedir, data_root, image_root, dataset_name, output_dir, query_subset=None
):
    def process_columns(df):
        # Convert all the values in the columns with -id in their columnnames to strings
        for col in df.columns:
            if "-id" in col:
                df[col] = df[col].astype(str)
        # Replace - with _ in all column names
        df.columns = df.columns.str.replace("-", "_")
        return df

    def process_images(df, data_basedir, image_root, output_dir):
        print(f"Processing images for df of size {len(df)} and columns {df.columns}")
        if "image" not in df.columns:
            return df

        # Identify ID column for image matching
        id_col = next(
            (
                c
                for c in df.columns
                if c in ["query-id", "corpus-id", "_id", "id", "docid", "qid"]
            ),
            None,
        )
        assert id_col in df.columns, f"ID column {id_col} not found in {df.columns}"

        def validate_and_replace(row):
            img_data = row["image"]
            if img_data is None:
                return None

            # Extract bytes from image column (can be bytes or a dict with 'bytes' key)
            if isinstance(img_data, dict):
                img_bytes = img_data.get("bytes")
            else:
                img_bytes = img_data

            assert img_bytes is not None, f"Image bytes are None for {obj_id}"

            obj_id = str(row[id_col])
            # The user specified qid.png (which we generalize to id.png)
            # img_path = os.path.join(original_image_root, f"{obj_id}.png")
            # assert os.path.exists(img_path), f"Image path {img_path} does not exist for df data: {row.to_dict()}"
            if "visrag" in dataset_name.lower():
                # image names are used as object ids, and some of them are super long...
                base, _ = os.path.splitext(obj_id)
                short_base = (
                    base[:50]
                    + "_"
                    + hashlib.md5(obj_id.encode("utf-8")).hexdigest()[:8]
                )  # Truncate base, add original filename hash
                new_img_path = os.path.join(output_dir, image_root, f"{short_base}.png")
            else:
                new_img_path = os.path.join(output_dir, image_root, f"{obj_id}.png")
            os.makedirs(os.path.dirname(new_img_path), exist_ok=True)
            with open(new_img_path, "wb") as f:
                f.write(img_bytes)
            return new_img_path

        df["image_path"] = df.apply(validate_and_replace, axis=1)
        df.drop(columns=["image"], inplace=True)
        return df

    data_dir = os.path.join(data_basedir, data_root)
    assert os.path.exists(data_dir), f"Data directory {data_dir} does not exist."
    print(f"Converting {dataset_name} from {data_dir}...")
    # 1. Convert Queries (topics)
    # Try "queries" or "query" subfolder
    queries_dir = os.path.join(data_dir, "queries")
    if not os.path.exists(queries_dir):
        queries_dir = os.path.join(data_dir, "query")

    query_files = find_parquets(queries_dir)
    assert len(query_files) > 0, f"No queries parquet found in {queries_dir}"

    topics_output_dir = os.path.join(output_dir, "topics")
    os.makedirs(topics_output_dir, exist_ok=True)
    topics_path = os.path.join(
        topics_output_dir,
        f"topics.mmeb-visdoc-{dataset_name}.{get_suffix(dataset_name)}.jsonl",
    )

    with open(topics_path, "w", encoding="utf-8") as f_out:
        total_queries = 0
        for q_file in query_files:
            df = pd.read_parquet(q_file)
            df = process_images(df, data_basedir, image_root, topics_output_dir)
            df = process_columns(df)
            if query_subset:
                df = df[df["language"] == query_subset]
            df = df.rename(columns={"query_id": "qid"})
            json_str = df.to_json(
                orient="records",
                lines=True,
                force_ascii=False,
            )
            if json_str:
                f_out.write(json_str.strip().replace("\\/", "/") + "\n")
            total_queries += len(df)
        print(f"  - Saved {total_queries} queries to {topics_path}")

    # 2. Convert Corpus (documents)
    corpus_dir = os.path.join(data_dir, "corpus")
    if not os.path.exists(corpus_dir):
        corpus_dir = os.path.join(data_dir, "document")

    corpus_files = find_parquets(corpus_dir)
    assert len(corpus_files) > 0, f"No corpus parquet found in {corpus_dir}"

    corpus_output_dir = os.path.join(output_dir, "corpus")
    os.makedirs(corpus_output_dir, exist_ok=True)
    corpus_path = os.path.join(corpus_output_dir, f"mmeb_visdoc_{dataset_name}.jsonl")

    with open(corpus_path, "w", encoding="utf-8") as f_out:
        total_docs = 0
        for c_file in corpus_files:
            df = pd.read_parquet(c_file)
            df = process_images(df, data_basedir, image_root, corpus_output_dir)
            df = process_columns(df)
            json_str = df.to_json(
                orient="records",
                lines=True,
                force_ascii=False,
            )
            if json_str:
                f_out.write(json_str.strip().replace("\\/", "/") + "\n")
            total_docs += len(df)
        print(f"  - Saved {total_docs} documents to {corpus_path}")

    # 3. Convert Qrels
    qrels_dir = os.path.join(data_dir, "qrels")
    if not os.path.exists(qrels_dir):
        qrels_dir = os.path.join(data_dir, "qrel")

    qrels_files = find_parquets(qrels_dir)
    assert len(qrels_files) > 0, f"No qrels parquet found in {qrels_dir}"

    qrels_out_dir = os.path.join(output_dir, "qrels")
    os.makedirs(qrels_out_dir, exist_ok=True)
    qrels_path = os.path.join(
        qrels_out_dir,
        f"qrels.mmeb-visdoc-{dataset_name}.{get_suffix(dataset_name)}.txt",
    )

    with open(qrels_path, "w", encoding="utf-8") as f_out:
        total_qrels = 0
        for qr_file in qrels_files:
            df = pd.read_parquet(qr_file)
            df = df[["query-id", "corpus-id", "score"]]
            df["q0"] = 0
            df = df[["query-id", "q0", "corpus-id", "score"]]

            # Keep max score per (query-id, corpus-id) without changing order
            max_scores = df.groupby(
                ["query-id", "corpus-id"]
            )["score"].transform("max")
            cleaned_df = df[df["score"] == max_scores]
            cleaned_df = cleaned_df.drop_duplicates(
                subset=["query-id", "corpus-id"], keep="first"
            )

            # Remove fully duplicated rows if any remain
            if len(df) != len(cleaned_df):
                print(
                    f"  - Removed {len(df) - len(cleaned_df)} qid, docid duplicate pairs from {qr_file} keeping the one with the highest score"
                )
                df = cleaned_df
            df.to_csv(qrels_path, sep="\t", index=False, header=False)
            total_qrels += len(df)
        print(f"  - Saved {total_qrels} qrels to {qrels_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert Parquet files to Pyserini JSONL/TSV format"
    )
    parser.add_argument(
        "--yaml_file",
        type=str,
        required=True,
        help="YAML file containing tasks and data_root",
    )
    parser.add_argument(
        "--data_basedir",
        type=str,
        default=".",
        help="Base directory to prepend to data_root",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the converted files",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4)",
    )
    args = parser.parse_args()

    with open(args.yaml_file, "r") as f:
        tasks_config = yaml.safe_load(f)

    if not tasks_config:
        print(f"Error: YAML file {args.yaml_file} is empty or invalid.")
        return

    # Prepare tasks for parallel processing
    task_args = []
    for task_name, config in tasks_config.items():
        os.makedirs(args.output_dir, exist_ok=True)
        task_args.append(
            (
                args.data_basedir,
                config.get("data_root"),
                config.get("image_root"),
                task_name,
                args.output_dir,
                config.get("query_subset"),
            )
        )

    if not task_args:
        print("No valid tasks found to process.")
        return

    # Process tasks in parallel
    print(f"Starting parallel processing with {args.num_workers} workers...")
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        # Use a wrapper list to track futures for tqdm
        futures = [
            executor.submit(convert_parquet_to_pyserini, *task) for task in task_args
        ]
        for _ in tqdm(futures, desc="Converting tasks"):
            _.result()  # Wait for each task and propagate exceptions


if __name__ == "__main__":
    main()
