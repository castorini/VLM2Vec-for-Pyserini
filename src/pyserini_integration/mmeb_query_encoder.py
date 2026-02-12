from typing import List

import torch
import torch.distributed as dist
from datasets.distributed import split_dataset_by_node
from torch.utils.data import DataLoader

from vlm2vec_for_pyserini.data.collator.eval_collator import \
    MultimodalEvalDataCollator
from vlm2vec_for_pyserini.pyserini_integration.mmeb_base_encoder import \
    MMEBBaseEncoder

class QueryEncoder(MMEBBaseEncoder):
    def __init__(
        self,
        model_name: str,
        model_type: str,
        pooling="eos",
        l2_norm=False,
        device="cuda:0",
    ):
        super().__init__(model_name, model_type, pooling, l2_norm, device)
    def encode(
        self,
        qid: List[int] | int,
        query: List[str] | str,
        fp16: bool = True,
    ):
        if fp16:
            dtype = torch.bfloat16
        else:
            dtype = torch.float32
        self.model = self.model.to(device=self.device, dtype=dtype)
        if not isinstance(qid, list):
            qid = [qid]
        if not isinstance(query, list):
            query = [query]
        batch_len = len(qid)
        assert (
            len(query) == batch_len
        ), f"Number of queries must match the number of qids, len(query): {len(query)}, batch_len: {batch_len}"
        batch_dict = {
            "qid": qid,
            "query": query,
        }
        collator = MultimodalEvalDataCollator(
            self.processor, self.model_args, self.data_args, "qry"
        )
        dataset_kwargs = {
            "model_backbone": self.model_backbone,
        }
        full_eval_qry_dataset = self.dataset_class.topics_dataset(
            batch_dict, **dataset_kwargs
        )
        eval_qry_dataset = full_eval_qry_dataset
        # Pad datasets to be divisible by world_size before splitting
        if dist.is_initialized():
            padded_qry_dataset, _ = self.pad_dataset_to_divisible(
                full_eval_qry_dataset, self.world_size
            )
            eval_qry_dataset = split_dataset_by_node(
                padded_qry_dataset, rank=self.local_rank, world_size=self.world_size
            )
        else:
            padded_qry_dataset = full_eval_qry_dataset
        dataloader = DataLoader(
            eval_qry_dataset, batch_size=batch_len, collate_fn=collator
        )
        query_embeddings, _ = self.encode_embeddings(
            self.model,
            dataloader,
            self.model_args,
            padded_qry_dataset,
            dtype,
            encode_side="qry",
            description=f"Queries",
        )
        return query_embeddings[: len(full_eval_qry_dataset)]
