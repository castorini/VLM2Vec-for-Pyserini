import argparse
import gc
import os

import torch
import yaml
from pyserini.encode import JsonlCollectionIterator
from pyserini.encode.optional import FaissRepresentationWriter
from pyserini.query_iterator import MMEBQueryIterator
from pyserini.search.faiss import FaissSearcher

from vlm2vec_for_pyserini.pyserini_integration.mmeb_corpus_encoder import \
    CorpusEncoder
from vlm2vec_for_pyserini.pyserini_integration.mmeb_query_encoder import \
    QueryEncoder


def main():
    parser = argparse.ArgumentParser(
        description="Quick start demo for VLM2Vec with Pyserini"
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        required=True,
        help="Base directory for data, indexes, and results",
    )
    parser.add_argument(
        "--visdoc_yaml",
        type=str,
        default="visdoc.yaml",
        help="Path to visdoc.yaml file",
    )
    parser.add_argument(
        "--model_names",
        nargs="+",
        default=[
            "Alibaba-NLP/gme-Qwen2-VL-2B-Instruct",
            "VLM2Vec/VLM2Vec-V2.0",
            "code-kunkun/LamRA-Ret",
        ],
        help="List of model names to run",
    )
    args = parser.parse_args()

    base_dir = args.base_dir
    visdoc_yaml_path = args.visdoc_yaml
    model_names = args.model_names

    # 1. Encode the corpus
    MMEB_CORPUS_FIELDS = ["corpus_id", "image_path"]

    with open(visdoc_yaml_path, "r") as f:
        visdoc_yaml = yaml.load(f, Loader=yaml.FullLoader)
        dataset_names = list(visdoc_yaml.keys())

    for model_name in model_names:
        model_type = None
        if "gme" in model_name.lower():
            model_type = "gme"
        elif "lamra" in model_name.lower():
            model_type = "lamra"
        model_suffix = model_name.split("/")[-1]
        for dataset_name in dataset_names:
            output_dir = f"{base_dir}/results"
            os.makedirs(output_dir, exist_ok=True)
            if os.path.exists(
                f"{output_dir}/mmeb-visdoc-{dataset_name}.{model_suffix}.trec"
            ):
                print(f"Results already exist for {dataset_name} and {model_name}")
                continue

            mmeb_corpus_encoder = CorpusEncoder(
                model_name=model_name,
                model_type=model_type,
                pooling="eos",
                l2_norm=True,
                device="cuda:0",
            )

            collection_iterator = JsonlCollectionIterator(
                f"{base_dir}/corpus/mmeb_visdoc_{dataset_name}.jsonl",
                fields=MMEB_CORPUS_FIELDS,
                docid_field="corpus_id",
            )
            # get the dimension from the model config file
            dimension = mmeb_corpus_encoder.model.config.hidden_size
            embedding_writer = FaissRepresentationWriter(
                f"{base_dir}/indexes/{dataset_name}.{model_suffix}", dimension=dimension
            )

            with embedding_writer:
                for batch_info in collection_iterator(32):
                    kwargs = {"fp16": True}
                    for field_name in MMEB_CORPUS_FIELDS:
                        kwargs[f"{field_name}s"] = batch_info[field_name]

                    embeddings = mmeb_corpus_encoder.encode(**kwargs)
                    print(f"Embeddings shape: {embeddings.shape}")
                    batch_info["vector"] = embeddings
                    embedding_writer.write(batch_info, MMEB_CORPUS_FIELDS)
            del mmeb_corpus_encoder
            gc.collect()
            torch.cuda.empty_cache()

            # Searching Step
            mmeb_query_encoder = QueryEncoder(
                model_name=model_name,
                model_type=model_type,
                pooling="eos",
                l2_norm=True,
                device="cuda:0",
            )

            searcher = FaissSearcher(
                f"{base_dir}/indexes/{dataset_name}.{model_suffix}", mmeb_query_encoder
            )
            topics_file = (
                f"{base_dir}/topics/topics.mmeb-visdoc-{dataset_name}.test.jsonl"
            )
            if not os.path.exists(topics_file):
                topics_file = (
                    f"{base_dir}/topics/topics.mmeb-visdoc-{dataset_name}.train.jsonl"
                )
            assert os.path.exists(
                topics_file
            ), f"Topics file not found at {topics_file}"
            query_iterator = MMEBQueryIterator.from_topics(topics_file)

            print(f"query_iterator: {query_iterator}")
            results = {}
            for qid, query_data in query_iterator:
                print("qid: " + str(qid))
                print("query_data: " + str(query_data))

                hits = searcher.search(query_data, k=1000)
                results[qid] = [(hit.docid, hit.score) for hit in hits]
            del mmeb_query_encoder
            del searcher
            gc.collect()
            torch.cuda.empty_cache()

            # save the results in trec format
            with open(
                f"{output_dir}/mmeb-visdoc-{dataset_name}.{model_suffix}.trec", "w"
            ) as f:
                for qid, hits in results.items():
                    for i, hit in enumerate(hits):
                        f.write(
                            f"{qid} Q0 {hit[0]} {i+1} {hit[1]} mmeb-visdoc-{dataset_name}.{model_suffix}\n"
                        )


if __name__ == "__main__":
    main()
