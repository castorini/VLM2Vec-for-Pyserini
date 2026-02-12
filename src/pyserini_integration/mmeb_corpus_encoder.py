from typing import List

import torch
import torch.distributed as dist
from datasets.distributed import split_dataset_by_node
from torch.utils.data import DataLoader

from vlm2vec_for_pyserini.data.collator.eval_collator import \
    MultimodalEvalDataCollator
from vlm2vec_for_pyserini.pyserini_integration.mmeb_base_encoder import \
    MMEBBaseEncoder


class CorpusEncoder(MMEBBaseEncoder):
    def __init__(
        self,
        model_name: str,
        model_type: str,
        image_resolution: str | None = None,
        pooling="eos",
        l2_norm=False,
        device="cuda:0",
    ):
        super().__init__(model_name, model_type, pooling, l2_norm, device)
        self.image_resolution = image_resolution

    def encode(
        self,
        corpus_ids: List[int],
        image_paths: List[str],
        fp16: bool = True,
    ):
        if fp16:
            dtype = torch.bfloat16
        else:
            dtype = torch.float32
        self.model = self.model.to(device=self.device, dtype=dtype)
        batch_len = len(corpus_ids)
        assert (
            len(image_paths) == batch_len
        ), "Number of image paths must match the number of corpus ids"
        batch_dict = {
            "corpus_id": corpus_ids,
            "image_path": image_paths,
        }
        collator = MultimodalEvalDataCollator(
            self.processor, self.model_args, self.data_args, "cand"
        )
        dataset_kwargs = {
            "image_resolution": self.image_resolution,
            "model_backbone": self.model_backbone,
        }
        full_eval_cand_dataset = self.dataset_class.corpus_dataset(
            batch_dict, **dataset_kwargs
        )
        eval_cand_dataset = full_eval_cand_dataset
        # Pad datasets to be divisible by world_size before splitting
        if dist.is_initialized():
            padded_cand_dataset, _ = self.pad_dataset_to_divisible(
                full_eval_cand_dataset, self.world_size
            )
            eval_cand_dataset = split_dataset_by_node(
                padded_cand_dataset, rank=self.local_rank, world_size=self.world_size
            )
        else:
            padded_cand_dataset = full_eval_cand_dataset
        dataloader = DataLoader(
            eval_cand_dataset, batch_size=batch_len, collate_fn=collator
        )
        corpus_embeddings, _ = self.encode_embeddings(
            self.model,
            dataloader,
            self.model_args,
            padded_cand_dataset,
            dtype,
            encode_side="cand",
            description=f"Candidates",
        )
        return corpus_embeddings[: len(full_eval_cand_dataset)]
