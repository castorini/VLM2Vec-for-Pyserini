import datetime
import os
import random
import sys
import time
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from datasets import Dataset, concatenate_datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoConfig

from vlm2vec_for_pyserini.arguments import DataArguments, ModelArguments
from vlm2vec_for_pyserini.model.model import MMEBModel
from vlm2vec_for_pyserini.model.processor import (COLPALI, get_backbone_name,
                                                  load_processor)
from vlm2vec_for_pyserini.utils.basic_utils import (batch_to_device,
                                                    print_master, print_rank)
from vlm2vec_for_pyserini.pyserini_integration.visdoc_dataset import VisDocDatasetForPyserini


class MMEBBaseEncoder(ABC):
    def __init__(
        self,
        model_name: str,
        model_type: str,
        pooling="eos",
        l2_norm=False,
        device="cuda:0",
    ):
        self.dataset_class = VisDocDatasetForPyserini
        if "RANK" in os.environ and dist.is_available() and not dist.is_initialized():
            dist.init_process_group(
                backend="nccl", timeout=datetime.timedelta(minutes=60)
            )
        self.local_rank = dist.get_rank() if dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        # DEBUG PRINTS for Distributed Setup
        print_master("Distributed init debug info:")
        print_master(f"RANK: {os.environ.get('RANK')}")
        print_master(f"LOCAL_RANK: {os.environ.get('LOCAL_RANK')}")
        print_master(f"WORLD_SIZE: {os.environ.get('WORLD_SIZE')}")
        print_master(f"MASTER_ADDR: {os.environ.get('MASTER_ADDR')}")
        print_master(f"MASTER_PORT: {os.environ.get('MASTER_PORT')}")
        if dist.is_initialized():
            print_rank(f"dist.get_rank(): {dist.get_rank()}")
            print_rank(f"dist.get_world_size(): {dist.get_world_size()}")

        for arg in sys.argv:
            if arg.startswith("--local-rank="):
                rank = arg.split("=")[1]
                sys.argv.remove(arg)
                sys.argv.append("--local_rank")
                sys.argv.append(rank)
        # --- Model Loading ---
        hf_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        self.model_backbone = get_backbone_name(
            hf_config=hf_config, model_type=model_type
        )
        self.model_args = ModelArguments(
            model_name=model_name,
            model_backbone=self.model_backbone,
            pooling=pooling,
            normalize=l2_norm,
        )
        # TODO: Needs confirmation; the processor is using data_args for image resize, but it seems like in none of the eval samples model image resizing is being used.
        self.data_args = DataArguments()

        # --- DDP-Safe Model Loading ---
        # Step 1: Only the master process (rank 0) downloads the model.
        if self.local_rank == 0:
            processor = load_processor(self.model_args, self.data_args)
            model = MMEBModel.load(
                self.model_args, is_trainable=False, processor=processor
            )
            print_master(
                f"[rank=0] Loading the model from Huggingface: {self.model_args.model_name}..."
            )
        # Step 2: All processes wait here. The non-master processes will pause
        # until the master process (rank 0) finishes downloading and exits this barrier.
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        # Step 3: Now that the model is cached, the non-master processes load it from the local cache.
        if self.local_rank != 0:
            print_rank(f"Loading the model from cache...")
            processor = load_processor(self.model_args, self.data_args)
            time.sleep(random.randint(2 * self.local_rank, 3 * self.local_rank))
            model = MMEBModel.load(
                self.model_args, is_trainable=False, processor=processor
            )
        model.eval()
        self.device = device
        self.model = model.to(self.device)
        self.processor = processor

    # Copied from `pad_dataset_to_divisible` function defined in eval.py since eval.py is outside of the src directory and it will not be part of the vlm2vec_for_pyserini package.
    def pad_dataset_to_divisible(self, dataset, world_size):
        num_samples = len(dataset)
        if num_samples % world_size == 0:
            return dataset, num_samples

        num_to_add = world_size - (num_samples % world_size)
        padded_size = num_samples + num_to_add

        padding_data = dataset.select([i % len(dataset) for i in range(num_to_add)])
        padded_dataset = concatenate_datasets([dataset, padding_data])
        return padded_dataset, padded_size

    # Copied from `encode_embeddings` function defined in eval.py since eval.py is outside of the src directory and it will not be part of the vlm2vec_for_pyserini package.
    def encode_embeddings(
        self,
        model: MMEBModel,
        loader: DataLoader,
        model_args: ModelArguments,
        full_dataset: Dataset,
        dtype: torch.dtype,
        encode_side: str,
        description: str = "Encoding",
    ) -> tuple[np.ndarray, list]:
        """
        Encodes embeddings for a given dataset using the model, handling both standard and
        late-interaction models in a DDP-safe manner.
        """
        local_rank = dist.get_rank() if dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist.is_initialized() else 1

        # Check if the model is a late-interaction type
        is_late_interaction = model_args.model_backbone == COLPALI

        local_embeds = []
        local_gt_infos = []
        local_max_len = 0

        model.eval()
        with torch.no_grad():
            for inputs, dataset_info in tqdm(
                loader,
                desc=f"{description} (rank {local_rank})",
                disable=local_rank > 0,
            ):
                inputs = batch_to_device(inputs, self.device)
                if "cuda" in self.device:
                    device_type = "cuda"
                else:
                    device_type = "cpu"
                with torch.autocast(enabled=True, dtype=dtype, device_type=device_type):
                    # Determine if encoding query or target based on available keys
                    if encode_side == "qry":
                        output = model(qry=inputs)
                        reps = output["qry_reps"].detach()
                        local_gt_infos.extend(
                            dataset_info
                        )  # to retain all information per query
                    else:
                        output = model(tgt=inputs)
                        reps = output["tgt_reps"].detach()
                        local_gt_infos.extend(
                            [info["cand_name"] for info in dataset_info]
                        )  # to retain ground-truth labels

                if is_late_interaction and reps.dim() == 3:
                    local_max_len = max(local_max_len, reps.shape[1])

                local_embeds.append(reps)

        if not local_embeds:
            # Handle cases where a rank gets no data
            return np.array([]), []

        # === DDP Synchronization and Padding for Late-Interaction Models ===
        if is_late_interaction:
            if dist.is_initialized():
                # 1. Find the global maximum sequence length across all ranks
                local_max_len_tensor = torch.tensor(local_max_len, device=self.device)
                dist.all_reduce(local_max_len_tensor, op=dist.ReduceOp.MAX)
                global_max_len = local_max_len_tensor.item()
            else:
                global_max_len = local_max_len

            # 2. Pad all local embeddings to the global max length
            padded_embeds = []
            for reps_batch in local_embeds:
                if reps_batch.dim() == 3:
                    B, L, H = reps_batch.shape
                    padding_size = global_max_len - L
                    padded_batch = F.pad(
                        reps_batch, (0, 0, 0, padding_size), "constant", 0
                    )
                    padded_embeds.append(padded_batch)
                else:  # Should not happen if model is consistently late-interaction
                    padded_embeds.append(reps_batch)

            embeds_tensor = torch.cat(padded_embeds, dim=0).contiguous()
        else:  # Standard dense models
            embeds_tensor = torch.cat(local_embeds, dim=0).contiguous()

        # === Gather embeddings and keys from all ranks ===
        if dist.is_initialized() and full_dataset.num_rows >= world_size:
            print_master(f"Gathering {encode_side} embeddings across all ranks...")

            # Use the more efficient all_gather_into_tensor for tensors
            output_shape = list(embeds_tensor.shape)
            output_shape[0] = full_dataset.num_rows
            embeds_tensor = embeds_tensor.to(self.device)
            gathered_embeds_tensor = torch.empty(
                output_shape, dtype=embeds_tensor.dtype, device=self.device
            )
            dist.all_gather_into_tensor(gathered_embeds_tensor, embeds_tensor)
            final_embeddings = gathered_embeds_tensor.cpu().float().numpy()
            # Gather metadata, for which all_gather_object is appropriate
            gathered_gt_infos = [None for _ in range(world_size)]
            dist.all_gather_object(gathered_gt_infos, local_gt_infos)
            all_gt_infos = [key for rank_keys in gathered_gt_infos for key in rank_keys]
        else:
            all_gt_infos = local_gt_infos
            final_embeddings = embeds_tensor.cpu().float().numpy()

        return final_embeddings, all_gt_infos

    @abstractmethod
    def encode(self, **kwargs: Any):
        pass
