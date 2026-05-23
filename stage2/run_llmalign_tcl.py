import argparse
import json
import math
import os
import shutil
import time
import sys
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from transformers import get_linear_schedule_with_warmup

if __package__ in {None, ""}:
    _THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    _REPO_ROOT = os.path.dirname(_THIS_DIR)
    if _REPO_ROOT not in sys.path:
        sys.path.insert(0, _REPO_ROOT)

from stage1.latent_attention import (
    LatentAttentionPooling,
    load_latent_attention_head,
    save_latent_attention_head,
)
from stage2.stage2_llmalign_utils import (
    as_bool,
    STAGE2_CONFIG_NAME,
    STAGE2_STATE_NAME,
    FrozenStage1BackboneEncoder,
    compose_item_text,
    is_baby_dataset_name,
    iter_sorted_item_ids,
    load_hidden_cache,
    load_item_titles,
    load_latent_head_module,
    mine_cooccurrence_pairs,
    normalize_pair_mining_mode,
    pad_hidden_states,
    pair_mining_mode_is_symmetric,
    resolve_stage1_item_settings,
    resolve_stage2_checkpoint_path,
    resolve_stage2_state_path,
    set_seed,
)


def parse_cache_dtype(dtype_name: str) -> torch.dtype:
    key = str(dtype_name).strip().lower()
    mapping = {
        "fp16": torch.float16,
        "float16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    if key not in mapping:
        raise ValueError(f"Unsupported cache dtype: {dtype_name}")
    return mapping[key]


def resolve_default_stage2_max_steps(dataset_name: str) -> int:
    key = str(dataset_name).strip().lower()
    if key in {"baby_5core", "baby_products"}:
        return 250
    return 500


def normalize_cooccurrence_head_init_mode(init_mode: str) -> str:
    key = str(init_mode).strip().lower()
    aliases = {
        "semantic": "semantic",
        "semantic_head": "semantic",
        "cpa": "semantic",
        "cpa_head": "semantic",
        "latent_attention_head": "semantic",
        "random": "random",
        "rand": "random",
        "scratch": "random",
    }
    normalized = aliases.get(key)
    if normalized is None:
        raise ValueError(
            f"Unsupported cooccurrence_head_init={init_mode}. Supported values: semantic, random"
        )
    return normalized


def build_latent_head_from_config(
    checkpoint_path: str,
    hidden_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[LatentAttentionPooling, Dict[str, int], str]:
    saved_cfg, _, resolved_path = load_latent_attention_head(checkpoint_path, map_location="cpu")
    cfg_latent_dim = int(saved_cfg.get("latent_dim", -1))
    if cfg_latent_dim <= 0:
        cfg_latent_dim = hidden_size
    head = LatentAttentionPooling(
        hidden_size=hidden_size,
        num_latents=int(saved_cfg.get("num_latents", 128)),
        latent_dim=cfg_latent_dim,
        num_cross_heads=int(saved_cfg.get("num_cross_heads", 8)),
        cross_dim_head=int(saved_cfg.get("cross_dim_head", 64)),
        ff_mult=int(saved_cfg.get("ff_mult", 4)),
    )
    head.to(device=device, dtype=dtype)
    return head, dict(saved_cfg), resolved_path


class CooccurrencePairDataset(Dataset):
    def __init__(
        self,
        source_item_ids: Sequence[int],
        target_item_ids: Sequence[int],
        pair_weights: Optional[Sequence[float]] = None,
        negative_item_ids: Optional[Sequence[int]] = None,
    ) -> None:
        if len(source_item_ids) != len(target_item_ids):
            raise ValueError("source_item_ids and target_item_ids must have the same length")
        if pair_weights is not None and len(pair_weights) != len(source_item_ids):
            raise ValueError("pair_weights must have the same length as source_item_ids")
        if negative_item_ids is not None and len(negative_item_ids) != len(source_item_ids):
            raise ValueError("negative_item_ids must have the same length as source_item_ids")
        self.source_item_ids = [int(x) for x in source_item_ids]
        self.target_item_ids = [int(x) for x in target_item_ids]
        if pair_weights is None:
            self.pair_weights = [1.0] * len(self.source_item_ids)
        else:
            self.pair_weights = [float(x) for x in pair_weights]
        if negative_item_ids is None:
            self.negative_item_ids = None
        else:
            self.negative_item_ids = [int(x) for x in negative_item_ids]

    def __len__(self) -> int:
        return len(self.source_item_ids)

    def __getitem__(self, idx: int) -> Dict[str, int]:
        sample = {
            "source_item_id": self.source_item_ids[idx],
            "target_item_id": self.target_item_ids[idx],
            "pair_weight": self.pair_weights[idx],
        }
        if self.negative_item_ids is not None and int(self.negative_item_ids[idx]) > 0:
            sample["negative_item_id"] = int(self.negative_item_ids[idx])
        return sample


class CooccurrencePairCollator:
    def __init__(self, hidden_cache: Dict[str, object], hidden_dtype: torch.dtype = torch.float32) -> None:
        self.items = hidden_cache["items"]
        self.hidden_dtype = hidden_dtype

    def __call__(self, batch: List[Dict[str, int]]) -> Dict[str, torch.Tensor]:
        source_hidden: List[torch.Tensor] = []
        source_mask: List[torch.Tensor] = []
        target_hidden: List[torch.Tensor] = []
        target_mask: List[torch.Tensor] = []
        negative_hidden: List[torch.Tensor] = []
        negative_mask: List[torch.Tensor] = []
        source_item_ids: List[int] = []
        target_item_ids: List[int] = []
        negative_item_ids: List[int] = []
        pair_weights: List[float] = []
        use_hard_negatives = all(int(sample.get("negative_item_id", 0)) > 0 for sample in batch)
        for sample in batch:
            source_item_id = int(sample["source_item_id"])
            target_item_id = int(sample["target_item_id"])
            source_entry = self.items[source_item_id]
            target_entry = self.items[target_item_id]
            source_hidden.append(source_entry["hidden_states"])
            source_mask.append(source_entry["attention_mask"])
            target_hidden.append(target_entry["hidden_states"])
            target_mask.append(target_entry["attention_mask"])
            source_item_ids.append(source_item_id)
            target_item_ids.append(target_item_id)
            pair_weights.append(float(sample.get("pair_weight", 1.0)))
            if use_hard_negatives:
                negative_item_id = int(sample["negative_item_id"])
                negative_entry = self.items[negative_item_id]
                negative_hidden.append(negative_entry["hidden_states"])
                negative_mask.append(negative_entry["attention_mask"])
                negative_item_ids.append(negative_item_id)
        source_hidden_tensor, source_mask_tensor = pad_hidden_states(source_hidden, source_mask, self.hidden_dtype)
        target_hidden_tensor, target_mask_tensor = pad_hidden_states(target_hidden, target_mask, self.hidden_dtype)
        output = {
            "source_hidden_states": source_hidden_tensor,
            "source_attention_mask": source_mask_tensor,
            "target_hidden_states": target_hidden_tensor,
            "target_attention_mask": target_mask_tensor,
            "source_item_ids": torch.tensor(source_item_ids, dtype=torch.long),
            "target_item_ids": torch.tensor(target_item_ids, dtype=torch.long),
            "pair_weights": torch.tensor(pair_weights, dtype=torch.float32),
        }
        if use_hard_negatives:
            negative_hidden_tensor, negative_mask_tensor = pad_hidden_states(negative_hidden, negative_mask, self.hidden_dtype)
            output["negative_hidden_states"] = negative_hidden_tensor
            output["negative_attention_mask"] = negative_mask_tensor
            output["negative_item_ids"] = torch.tensor(negative_item_ids, dtype=torch.long)
        return output


class CooccurrenceContrastiveModel(nn.Module):
    def __init__(
        self,
        semantic_head_path: str,
        hidden_size: int,
        temperature: float,
        learnable_temperature: bool,
        logit_scale_max: float,
        normalize_embeddings: bool,
        forward_chunk_size: int,
        gradient_checkpointing: bool,
        cooccurrence_head_init: str,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.cooccurrence_head_init = normalize_cooccurrence_head_init_mode(cooccurrence_head_init)
        _, _, semantic_head_source = load_latent_head_module(
            semantic_head_path,
            hidden_size=hidden_size,
            device=device,
            dtype=dtype,
        )
        self.semantic_head_source = semantic_head_source
        if self.cooccurrence_head_init == "semantic":
            cooccurrence_head, _, cooccurrence_head_source = load_latent_head_module(
                semantic_head_path,
                hidden_size=hidden_size,
                device=device,
                dtype=dtype,
            )
        else:
            cooccurrence_head, _, cooccurrence_head_source = build_latent_head_from_config(
                semantic_head_path,
                hidden_size=hidden_size,
                device=device,
                dtype=dtype,
            )
        self.cooccurrence_head = cooccurrence_head
        self.cooccurrence_head_source = cooccurrence_head_source
        self.normalize_embeddings = bool(normalize_embeddings)
        self.learnable_temperature = bool(learnable_temperature)
        self.logit_scale_max = float(logit_scale_max)
        self.temperature = float(temperature)
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if self.learnable_temperature:
            self.logit_scale = nn.Parameter(
                torch.tensor(math.log(1.0 / self.temperature), dtype=torch.float32)
            )
        else:
            self.register_buffer(
                "logit_scale_const",
                torch.tensor(1.0 / self.temperature, dtype=torch.float32),
                persistent=False,
            )
        self.forward_chunk_size = max(0, int(forward_chunk_size))
        self.gradient_checkpointing = bool(gradient_checkpointing)

    def export_head_config(self) -> Dict[str, int]:
        return {
            "embedding_head_type": "latent_attention",
            **self.cooccurrence_head.export_config(),
        }

    def get_logit_scale(self) -> torch.Tensor:
        if self.learnable_temperature:
            return self.logit_scale.exp().clamp(max=self.logit_scale_max)
        return self.logit_scale_const.to(next(self.cooccurrence_head.parameters()).device)

    def _encode_chunk(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        head_param = next(self.cooccurrence_head.parameters(), None)
        if head_param is not None:
            hidden_states = hidden_states.to(device=head_param.device, dtype=head_param.dtype)
            attention_mask = attention_mask.to(device=head_param.device)
        return self.cooccurrence_head(hidden_states, attention_mask=attention_mask).float()

    def encode_raw(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        batch_size = int(hidden_states.size(0))
        if self.forward_chunk_size > 0 and batch_size > self.forward_chunk_size:
            outputs: List[torch.Tensor] = []
            for start in range(0, batch_size, self.forward_chunk_size):
                end = min(start + self.forward_chunk_size, batch_size)
                chunk_hidden_states = hidden_states[start:end]
                chunk_attention_mask = attention_mask[start:end]
                if self.training and self.gradient_checkpointing:
                    chunk_embeddings = checkpoint(
                        self._encode_chunk,
                        chunk_hidden_states,
                        chunk_attention_mask,
                        use_reentrant=False,
                    )
                else:
                    chunk_embeddings = self._encode_chunk(chunk_hidden_states, chunk_attention_mask)
                outputs.append(chunk_embeddings)
            return torch.cat(outputs, dim=0)
        if self.training and self.gradient_checkpointing:
            return checkpoint(
                self._encode_chunk,
                hidden_states,
                attention_mask,
                use_reentrant=False,
            )
        return self._encode_chunk(hidden_states, attention_mask)

    def maybe_normalize(self, embeddings: torch.Tensor) -> torch.Tensor:
        if self.normalize_embeddings:
            return F.normalize(embeddings, dim=-1)
        return embeddings

    def compute_logits(
        self,
        source_hidden_states: torch.Tensor,
        source_attention_mask: torch.Tensor,
        target_hidden_states: torch.Tensor,
        target_attention_mask: torch.Tensor,
        negative_hidden_states: Optional[torch.Tensor] = None,
        negative_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        source_emb = self.maybe_normalize(self.encode_raw(source_hidden_states, source_attention_mask))
        target_emb = self.maybe_normalize(self.encode_raw(target_hidden_states, target_attention_mask))
        candidate_emb = target_emb
        if negative_hidden_states is not None and negative_attention_mask is not None:
            negative_emb = self.maybe_normalize(self.encode_raw(negative_hidden_states, negative_attention_mask))
            candidate_emb = torch.cat([target_emb, negative_emb], dim=0)
        return torch.matmul(source_emb, candidate_emb.transpose(0, 1)) * self.get_logit_scale()

    def compute_loss_components(
        self,
        source_hidden_states: torch.Tensor,
        source_attention_mask: torch.Tensor,
        target_hidden_states: torch.Tensor,
        target_attention_mask: torch.Tensor,
        labels: torch.Tensor,
        negative_hidden_states: Optional[torch.Tensor] = None,
        negative_attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        logits = self.compute_logits(
            source_hidden_states=source_hidden_states,
            source_attention_mask=source_attention_mask,
            target_hidden_states=target_hidden_states,
            target_attention_mask=target_attention_mask,
            negative_hidden_states=negative_hidden_states,
            negative_attention_mask=negative_attention_mask,
        )
        loss = F.cross_entropy(logits, labels)
        return {
            "loss": loss,
            "logits": logits,
        }


@torch.no_grad()
def evaluate(
    model: CooccurrenceContrastiveModel,
    dataloader: DataLoader,
    device: torch.device,
    bf16: bool,
) -> Dict[str, float]:
    model.eval()
    losses: List[float] = []
    recall_at_1: List[float] = []
    recall_at_5: List[float] = []
    recall_at_10: List[float] = []
    autocast_enabled = bool(bf16) and device.type == "cuda"
    for batch in dataloader:
        batch = {
            key: (value.to(device) if isinstance(value, torch.Tensor) else value)
            for key, value in batch.items()
        }
        labels = torch.arange(batch["source_hidden_states"].size(0), device=device)
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=autocast_enabled,
        ):
            outputs = model.compute_loss_components(
                source_hidden_states=batch["source_hidden_states"],
                source_attention_mask=batch["source_attention_mask"],
                target_hidden_states=batch["target_hidden_states"],
                target_attention_mask=batch["target_attention_mask"],
                labels=labels,
                negative_hidden_states=batch.get("negative_hidden_states"),
                negative_attention_mask=batch.get("negative_attention_mask"),
            )
            logits = outputs["logits"]
            loss = outputs["loss"]
        losses.append(float(loss.detach().cpu().item()))
        for top_k, storage in ((1, recall_at_1), (5, recall_at_5), (10, recall_at_10)):
            effective_k = min(top_k, int(logits.size(1)))
            topk_indices = logits.topk(effective_k, dim=1).indices
            hits = topk_indices.eq(labels.unsqueeze(1)).any(dim=1).float().mean().detach().cpu().item()
            storage.append(float(hits))
    model.train()
    return {
        "eval_loss": float(np.mean(losses)) if losses else float("nan"),
        "eval_recall_at_1": float(np.mean(recall_at_1)) if recall_at_1 else float("nan"),
        "eval_recall_at_5": float(np.mean(recall_at_5)) if recall_at_5 else float("nan"),
        "eval_recall_at_10": float(np.mean(recall_at_10)) if recall_at_10 else float("nan"),
    }


def save_checkpoint(
    output_dir: str,
    step: int,
    model: CooccurrenceContrastiveModel,
    optimizer: torch.optim.Optimizer,
    scheduler,
    train_state: Dict[str, float],
) -> str:
    checkpoint_dir = os.path.join(output_dir, f"checkpoint-{int(step)}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    save_latent_attention_head(
        checkpoint_dir,
        model.cooccurrence_head,
        config=model.export_head_config(),
    )
    state_payload = {
        **train_state,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
    }
    if model.learnable_temperature:
        state_payload["logit_scale"] = model.logit_scale.detach().cpu()
    else:
        state_payload["logit_scale_const"] = model.logit_scale_const.detach().cpu()
    torch.save(state_payload, os.path.join(checkpoint_dir, STAGE2_STATE_NAME))
    return checkpoint_dir


def update_best_checkpoint(output_dir: str, checkpoint_dir: str) -> None:
    best_link = os.path.join(output_dir, "best_checkpoint")
    if os.path.islink(best_link) or os.path.exists(best_link):
        if os.path.islink(best_link):
            os.unlink(best_link)
        elif os.path.isdir(best_link):
            shutil.rmtree(best_link)
        else:
            os.remove(best_link)
    os.symlink(os.path.basename(checkpoint_dir), best_link)


def prune_non_best_checkpoints(output_dir: str, keep_checkpoint_dir: str) -> None:
    keep_realpath = os.path.realpath(keep_checkpoint_dir)
    for name in os.listdir(output_dir):
        checkpoint_dir = os.path.join(output_dir, name)
        if not name.startswith("checkpoint-") or not os.path.isdir(checkpoint_dir):
            continue
        if os.path.realpath(checkpoint_dir) == keep_realpath:
            continue
        shutil.rmtree(checkpoint_dir)


@torch.no_grad()
def encode_cached_item_embeddings(
    head: nn.Module,
    hidden_cache: Dict[str, object],
    item_ids: Sequence[int],
    hidden_dtype: torch.dtype,
    device: torch.device,
    batch_size: int,
    normalize_embeddings: bool = True,
) -> torch.Tensor:
    items = hidden_cache["items"]
    outputs: List[torch.Tensor] = []
    was_training = bool(head.training)
    head.eval()
    head_param = next(head.parameters(), None)
    head_dtype = head_param.dtype if head_param is not None else hidden_dtype
    effective_batch_size = max(1, int(batch_size))
    try:
        for start in range(0, len(item_ids), effective_batch_size):
            batch_item_ids = [int(item_id) for item_id in item_ids[start:start + effective_batch_size]]
            batch_hidden = [items[item_id]["hidden_states"] for item_id in batch_item_ids]
            batch_mask = [items[item_id]["attention_mask"] for item_id in batch_item_ids]
            hidden_tensor, mask_tensor = pad_hidden_states(batch_hidden, batch_mask, hidden_dtype)
            hidden_tensor = hidden_tensor.to(device=device, dtype=head_dtype)
            mask_tensor = mask_tensor.to(device)
            embeddings = head(hidden_tensor, attention_mask=mask_tensor).float()
            if normalize_embeddings:
                embeddings = F.normalize(embeddings, dim=-1)
            outputs.append(embeddings.detach().cpu())
    finally:
        if was_training:
            head.train()
    if not outputs:
        return torch.empty(0, 0, dtype=torch.float32)
    return torch.cat(outputs, dim=0)


def build_positive_adjacency(
    source_item_ids: Sequence[int],
    target_item_ids: Sequence[int],
    pair_mining_mode: str,
) -> Dict[int, Set[int]]:
    adjacency: Dict[int, Set[int]] = {}
    is_symmetric = pair_mining_mode_is_symmetric(pair_mining_mode)
    for source_item_id, target_item_id in zip(source_item_ids, target_item_ids):
        source_item_id = int(source_item_id)
        target_item_id = int(target_item_id)
        adjacency.setdefault(source_item_id, set()).add(target_item_id)
        if is_symmetric:
            adjacency.setdefault(target_item_id, set()).add(source_item_id)
    return adjacency


@torch.no_grad()
def mine_semantic_hard_negative_map(
    item_ids: Sequence[int],
    semantic_embeddings: torch.Tensor,
    positive_adjacency: Dict[int, Set[int]],
    topk: int,
    chunk_size: int,
    min_similarity: float,
) -> Tuple[Dict[int, int], Dict[str, float]]:
    if semantic_embeddings.ndim != 2 or int(semantic_embeddings.size(0)) != len(item_ids):
        raise ValueError("semantic_embeddings shape does not match item_ids")
    if len(item_ids) <= 1:
        return {}, {"hard_negative_item_count": 0, "hard_negative_coverage": 0.0}
    item_ids = [int(item_id) for item_id in item_ids]
    item_id_to_index = {item_id: idx for idx, item_id in enumerate(item_ids)}
    embedding_matrix = semantic_embeddings.float()
    effective_chunk_size = max(1, int(chunk_size))
    effective_topk = max(1, min(int(topk), len(item_ids) - 1))
    hard_negative_map: Dict[int, int] = {}
    selected_similarities: List[float] = []
    for start in range(0, len(item_ids), effective_chunk_size):
        end = min(start + effective_chunk_size, len(item_ids))
        similarity = torch.matmul(embedding_matrix[start:end], embedding_matrix.transpose(0, 1))
        for row_offset, source_index in enumerate(range(start, end)):
            source_item_id = item_ids[source_index]
            similarity[row_offset, source_index] = float("-inf")
            blocked_item_ids = set(positive_adjacency.get(source_item_id, set()))
            blocked_item_ids.add(source_item_id)
            blocked_indices = [item_id_to_index[item_id] for item_id in blocked_item_ids if item_id in item_id_to_index]
            if blocked_indices:
                similarity[row_offset, blocked_indices] = float("-inf")
        top_values, top_indices = similarity.topk(k=effective_topk, dim=1)
        for row_offset, source_index in enumerate(range(start, end)):
            chosen_item_id = 0
            chosen_similarity = float("-inf")
            for value, candidate_index in zip(top_values[row_offset].tolist(), top_indices[row_offset].tolist()):
                if not np.isfinite(value):
                    continue
                if value < float(min_similarity):
                    continue
                chosen_item_id = item_ids[int(candidate_index)]
                chosen_similarity = float(value)
                break
            if chosen_item_id > 0:
                hard_negative_map[item_ids[source_index]] = int(chosen_item_id)
                selected_similarities.append(chosen_similarity)
    coverage = float(len(hard_negative_map) / max(1, len(item_ids)))
    return hard_negative_map, {
        "hard_negative_item_count": int(len(hard_negative_map)),
        "hard_negative_coverage": coverage,
        "hard_negative_mean_similarity": float(np.mean(selected_similarities)) if selected_similarities else float("nan"),
        "hard_negative_min_similarity": float(np.min(selected_similarities)) if selected_similarities else float("nan"),
        "hard_negative_max_similarity": float(np.max(selected_similarities)) if selected_similarities else float("nan"),
    }


def build_train_sampler(pair_weights: Sequence[float]) -> WeightedRandomSampler:
    weight_tensor = torch.tensor([max(float(weight), 1e-8) for weight in pair_weights], dtype=torch.double)
    return WeightedRandomSampler(weight_tensor, num_samples=len(weight_tensor), replacement=True)


def build_train_eval_split(
    source_item_ids: Sequence[int],
    target_item_ids: Sequence[int],
    pair_weights: Optional[Sequence[float]],
    eval_ratio: float,
    max_eval_pairs: int,
    seed: int,
    negative_item_ids: Optional[Sequence[int]] = None,
) -> Tuple[CooccurrencePairDataset, CooccurrencePairDataset]:
    pair_count = len(source_item_ids)
    indices = list(range(pair_count))
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    eval_size = min(max(1, int(round(pair_count * float(eval_ratio)))), max(1, int(max_eval_pairs)))
    eval_indices = set(indices[:eval_size])
    train_source: List[int] = []
    train_target: List[int] = []
    train_weights: List[float] = []
    train_negative: List[int] = []
    eval_source: List[int] = []
    eval_target: List[int] = []
    eval_weights: List[float] = []
    eval_negative: List[int] = []
    for idx, (source_item_id, target_item_id) in enumerate(zip(source_item_ids, target_item_ids)):
        pair_weight = 1.0 if pair_weights is None else float(pair_weights[idx])
        negative_item_id = 0 if negative_item_ids is None else int(negative_item_ids[idx])
        if idx in eval_indices:
            eval_source.append(int(source_item_id))
            eval_target.append(int(target_item_id))
            eval_weights.append(pair_weight)
            if negative_item_ids is not None:
                eval_negative.append(negative_item_id)
        else:
            train_source.append(int(source_item_id))
            train_target.append(int(target_item_id))
            train_weights.append(pair_weight)
            if negative_item_ids is not None:
                train_negative.append(negative_item_id)
    if not train_source or not eval_source:
        raise ValueError("Failed to create non-empty train/eval split for co-occurrence pairs")
    return (
        CooccurrencePairDataset(train_source, train_target, train_weights, train_negative if negative_item_ids is not None else None),
        CooccurrencePairDataset(eval_source, eval_target, eval_weights, eval_negative if negative_item_ids is not None else None),
    )


def write_stage2_config(output_dir: str, config: Dict[str, object]) -> None:
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, STAGE2_CONFIG_NAME), "w", encoding="utf-8") as file:
        json.dump(config, file, ensure_ascii=False, indent=2)


def command_cache_hidden(args: argparse.Namespace) -> None:
    item_settings = resolve_stage1_item_settings(
        base_model_path=args.base_model,
        item_prefix=args.item_prefix,
        item_suffix=args.item_suffix,
        attention_mask_type=args.attention_mask_type,
        max_length=args.max_length,
    )
    item_id_to_title, _ = load_item_titles(args.dataset)
    encoder = FrozenStage1BackboneEncoder(
        base_model_path=args.base_model,
        attention_mask_type=str(item_settings["attention_mask_type"]),
        torch_dtype=torch.bfloat16 if (args.bf16 and torch.cuda.is_available()) else torch.float32,
    )
    sorted_item_ids = list(iter_sorted_item_ids(item_id_to_title))
    texts = [
        compose_item_text(
            item_id_to_title[item_id],
            item_prefix=str(item_settings["item_prefix"]),
            item_suffix=str(item_settings["item_suffix"]),
        )
        for item_id in sorted_item_ids
    ]
    cache_dtype = parse_cache_dtype(args.cache_dtype)
    start_time = time.time()
    encoded = encoder.encode_hidden_states(
        texts=texts,
        batch_size=int(args.batch_size),
        max_length=int(item_settings["max_length"]),
    )
    items: Dict[int, Dict[str, object]] = {}
    for item_id, title, item_text, (hidden_states, attention_mask) in zip(
        sorted_item_ids,
        [item_id_to_title[item_id] for item_id in sorted_item_ids],
        texts,
        encoded,
    ):
        items[int(item_id)] = {
            "title": str(title),
            "item_text": str(item_text),
            "hidden_states": hidden_states.to(dtype=cache_dtype).contiguous(),
            "attention_mask": attention_mask.to(dtype=torch.long).contiguous(),
        }
    payload = {
        "format_version": 1,
        "dataset": args.dataset,
        "base_model": args.base_model,
        "attention_mask_type": item_settings["attention_mask_type"],
        "item_prefix": item_settings["item_prefix"],
        "item_suffix": item_settings["item_suffix"],
        "max_length": int(item_settings["max_length"]),
        "hidden_size": int(encoder.hidden_size),
        "hidden_dtype": str(cache_dtype).replace("torch.", ""),
        "item_count": len(items),
        "items": items,
        "elapsed_seconds": round(time.time() - start_time, 3),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)) or ".", exist_ok=True)
    torch.save(payload, args.output_path)
    print(json.dumps({
        "output_path": os.path.abspath(args.output_path),
        "item_count": len(items),
        "hidden_size": int(encoder.hidden_size),
        "elapsed_seconds": payload["elapsed_seconds"],
    }, ensure_ascii=False, indent=2))


def command_mine_pairs(args: argparse.Namespace) -> None:
    mined_payload = mine_cooccurrence_pairs(
        train_file=args.train_file,
        sample=int(args.sample),
        seed=int(args.seed),
        min_count=int(args.min_count),
        compute_ppmi=as_bool(args.compute_ppmi),
        ppmi_alpha=float(args.ppmi_alpha),
        pair_mining_mode=args.pair_mining_mode,
        window_size=args.window_size,
    )
    pair_counts = mined_payload["pair_counts"]
    if not pair_counts:
        raise ValueError("No co-occurrence pairs mined from the training file")
    sorted_pairs = sorted(pair_counts.items())
    pair_mining_mode = normalize_pair_mining_mode(str(mined_payload.get("pair_mining_mode", args.pair_mining_mode)))
    window_size = mined_payload.get("window_size")
    expand_bidirectional_pairs = bool(as_bool(args.expand_bidirectional_pairs))
    pair_ppmi_scores = dict(mined_payload.get("pair_ppmi_scores") or {})
    expanded_source_item_ids: List[int] = []
    expanded_target_item_ids: List[int] = []
    expanded_counts: List[int] = []
    expanded_ppmi: List[float] = []
    for (source_item_id, target_item_id), raw_count in sorted_pairs:
        source_item_id = int(source_item_id)
        target_item_id = int(target_item_id)
        raw_count = int(raw_count)
        expanded_source_item_ids.append(source_item_id)
        expanded_target_item_ids.append(target_item_id)
        expanded_counts.append(raw_count)
        if pair_ppmi_scores:
            expanded_ppmi.append(float(pair_ppmi_scores.get((source_item_id, target_item_id), 0.0)))
        if expand_bidirectional_pairs:
            expanded_source_item_ids.append(target_item_id)
            expanded_target_item_ids.append(source_item_id)
            expanded_counts.append(raw_count)
            if pair_ppmi_scores:
                expanded_ppmi.append(float(pair_ppmi_scores.get((target_item_id, source_item_id), 0.0)))
    source_item_ids = torch.tensor(expanded_source_item_ids, dtype=torch.long)
    target_item_ids = torch.tensor(expanded_target_item_ids, dtype=torch.long)
    counts = torch.tensor(expanded_counts, dtype=torch.long)
    payload = {
        "format_version": 6,
        "train_file": args.train_file,
        "sample": int(args.sample),
        "seed": int(args.seed),
        "pair_mining_mode": pair_mining_mode,
        "window_size": None if window_size is None else int(window_size),
        "min_count": int(mined_payload.get("min_count", args.min_count)),
        "sequence_count": int(mined_payload.get("sequence_count", 0)),
        "sequence_unit": str(mined_payload.get("sequence_unit", "user")),
        "pre_filter_raw_pair_count": int(mined_payload.get("pre_filter_raw_pair_count", sum(expanded_counts))),
        "pre_filter_unique_pair_count": int(mined_payload.get("pre_filter_unique_pair_count", len(sorted_pairs))),
        "raw_pair_count": int(sum(int(count) for _, count in sorted_pairs)),
        "unique_pair_count": int(len(sorted_pairs)),
        "expanded_pair_count": int(len(expanded_source_item_ids)),
        "expand_bidirectional_pairs": bool(expand_bidirectional_pairs),
        "contains_ppmi": bool(mined_payload.get("contains_ppmi", False)),
        "ppmi_alpha": float(mined_payload.get("ppmi_alpha", args.ppmi_alpha)),
        "source_item_ids": source_item_ids,
        "target_item_ids": target_item_ids,
        "pair_counts": counts,
        "item_popularity": {
            int(item_id): int(count)
            for item_id, count in dict(mined_payload.get("item_popularity") or {}).items()
        },
        "pair_graph_marginals": {
            int(item_id): float(value)
            for item_id, value in dict(mined_payload.get("pair_graph_marginals") or {}).items()
        },
    }
    if expanded_ppmi:
        payload["pair_ppmi"] = torch.tensor(expanded_ppmi, dtype=torch.float32)
    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)) or ".", exist_ok=True)
    torch.save(payload, args.output_path)
    print(json.dumps({
        "output_path": os.path.abspath(args.output_path),
        "pair_mining_mode": payload["pair_mining_mode"],
        "window_size": payload["window_size"],
        "min_count": int(payload["min_count"]),
        "raw_pair_count": int(payload["raw_pair_count"]),
        "unique_pair_count": int(payload["unique_pair_count"]),
        "expanded_pair_count": int(payload["expanded_pair_count"]),
        "pre_filter_raw_pair_count": int(payload["pre_filter_raw_pair_count"]),
        "pre_filter_unique_pair_count": int(payload["pre_filter_unique_pair_count"]),
        "sequence_count": int(payload["sequence_count"]),
        "sequence_unit": payload["sequence_unit"],
        "expand_bidirectional_pairs": bool(payload["expand_bidirectional_pairs"]),
        "contains_ppmi": bool(payload["contains_ppmi"]),
        "ppmi_alpha": float(payload["ppmi_alpha"]),
    }, ensure_ascii=False, indent=2))


def _load_stage2_state(checkpoint_dir: str) -> Dict[str, object]:
    state_path = resolve_stage2_state_path(checkpoint_dir)
    if not os.path.isfile(state_path):
        raise FileNotFoundError(f"Stage2 state not found: {state_path}")
    return torch.load(state_path, map_location="cpu", weights_only=False)


def _infer_no_improve_rounds(output_dir: str, eval_steps: int) -> int:
    checkpoint_dirs = [
        os.path.join(output_dir, name)
        for name in os.listdir(output_dir)
        if name.startswith("checkpoint-") and os.path.isdir(os.path.join(output_dir, name))
    ]
    checkpoint_dirs.sort(key=lambda path: int(os.path.basename(path).split("-")[-1]))
    rows = []
    for checkpoint_dir in checkpoint_dirs:
        state_path = resolve_stage2_state_path(checkpoint_dir)
        if not os.path.isfile(state_path):
            continue
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        step = int(state.get("step", 0))
        if eval_steps > 0 and step % eval_steps != 0:
            continue
        best_eval_recall_at_10 = float(state.get("best_eval_recall_at_10", float("nan")))
        best_eval_loss = float(state.get("best_eval_loss", float("nan")))
        rows.append((step, best_eval_recall_at_10, best_eval_loss))
    no_improve_rounds = 0
    prev_best_recall = float("-inf")
    prev_best_loss = float("inf")
    for _, best_recall, best_loss in rows:
        improved = (
            best_recall > prev_best_recall
            or (best_recall == prev_best_recall and best_loss < prev_best_loss)
        )
        if improved:
            prev_best_recall = best_recall
            prev_best_loss = best_loss
            no_improve_rounds = 0
        else:
            no_improve_rounds += 1
    return no_improve_rounds


def _restore_resume_state(
    model: CooccurrenceContrastiveModel,
    optimizer: torch.optim.Optimizer,
    scheduler,
    resume_checkpoint_dir: str,
    hidden_size: int,
    device: torch.device,
) -> Dict[str, object]:
    resume_head, _, _ = load_latent_head_module(
        resume_checkpoint_dir,
        hidden_size=hidden_size,
        device=device,
        dtype=next(model.cooccurrence_head.parameters()).dtype,
    )
    model.cooccurrence_head.load_state_dict(resume_head.state_dict())
    resume_state = _load_stage2_state(resume_checkpoint_dir)
    if "optimizer" in resume_state:
        optimizer.load_state_dict(resume_state["optimizer"])
    if "scheduler" in resume_state:
        scheduler.load_state_dict(resume_state["scheduler"])
    if model.learnable_temperature and "logit_scale" in resume_state:
        model.logit_scale.data.copy_(resume_state["logit_scale"].to(model.logit_scale.device, dtype=model.logit_scale.dtype))
    return resume_state


def command_train(args: argparse.Namespace) -> None:
    set_seed(int(args.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hidden_cache = load_hidden_cache(args.hidden_cache_path)
    dataset_name = str(hidden_cache.get("dataset", "")).strip()
    max_steps = int(args.max_steps) if args.max_steps is not None else resolve_default_stage2_max_steps(dataset_name)
    early_stopping_patience = int(args.early_stopping_patience)
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if early_stopping_patience < 0:
        raise ValueError("early_stopping_patience must be >= 0")
    pair_payload = torch.load(args.pairs_path, map_location="cpu", weights_only=False)
    pair_mining_mode = normalize_pair_mining_mode(str(pair_payload.get("pair_mining_mode", "cooccurrence")))
    pair_window_size = pair_payload.get("window_size")
    cache_items = hidden_cache["items"]
    source_item_ids = pair_payload["source_item_ids"].tolist()
    target_item_ids = pair_payload["target_item_ids"].tolist()
    pair_counts = pair_payload.get("pair_counts")
    pair_ppmi = pair_payload.get("pair_ppmi")
    filtered_source: List[int] = []
    filtered_target: List[int] = []
    filtered_counts: List[int] = []
    filtered_ppmi: List[float] = []
    for idx, (source_item_id, target_item_id) in enumerate(zip(source_item_ids, target_item_ids)):
        if int(source_item_id) not in cache_items or int(target_item_id) not in cache_items:
            continue
        filtered_source.append(int(source_item_id))
        filtered_target.append(int(target_item_id))
        if pair_counts is not None:
            filtered_counts.append(int(pair_counts[idx]))
        if pair_ppmi is not None:
            filtered_ppmi.append(float(pair_ppmi[idx]))
    if not filtered_source:
        raise ValueError("No co-occurrence pairs remain after filtering against the hidden cache")
    cache_hidden_dtype = parse_cache_dtype(str(hidden_cache.get("hidden_dtype", "float32")))
    hidden_size = int(hidden_cache.get("hidden_size") or next(iter(cache_items.values()))["hidden_states"].shape[-1])
    semantic_head_path = os.path.join(os.path.abspath(args.base_model), "latent_attention_head.pt")
    if not os.path.isfile(semantic_head_path):
        raise FileNotFoundError(f"Semantic head not found under base model: {semantic_head_path}")

    positive_sampling_mode = str(args.positive_sampling_mode).strip().lower()
    if positive_sampling_mode not in {"uniform", "count", "ppmi"}:
        raise ValueError(f"Unsupported positive_sampling_mode: {args.positive_sampling_mode}")
    selected_pair_weights: Optional[List[float]] = None
    if positive_sampling_mode == "count":
        selected_pair_weights = [float(count) for count in filtered_counts] if filtered_counts else None
    elif positive_sampling_mode == "ppmi":
        if not filtered_ppmi:
            raise ValueError("positive_sampling_mode=ppmi requires pair payload with pair_ppmi")
        selected_pair_weights = [max(float(score), 1e-8) for score in filtered_ppmi]

    hard_negative_mode = str(args.hard_negative_mode).strip().lower()
    if hard_negative_mode not in {"none", "semantic"}:
        raise ValueError(f"Unsupported hard_negative_mode: {args.hard_negative_mode}")
    filtered_negative_item_ids: Optional[List[int]] = None
    hard_negative_stats: Dict[str, float] = {
        "hard_negative_item_count": 0,
        "hard_negative_coverage": 0.0,
        "hard_negative_mean_similarity": float("nan"),
        "hard_negative_min_similarity": float("nan"),
        "hard_negative_max_similarity": float("nan"),
    }
    if hard_negative_mode == "semantic":
        positive_adjacency = build_positive_adjacency(
            source_item_ids=filtered_source,
            target_item_ids=filtered_target,
            pair_mining_mode=pair_mining_mode,
        )
        semantic_head, _, _ = load_latent_head_module(
            semantic_head_path,
            hidden_size=hidden_size,
            device=device,
            dtype=torch.float32,
        )
        semantic_head.eval()
        semantic_item_ids = sorted(int(item_id) for item_id in cache_items.keys())
        semantic_embeddings = encode_cached_item_embeddings(
            head=semantic_head,
            hidden_cache=hidden_cache,
            item_ids=semantic_item_ids,
            hidden_dtype=cache_hidden_dtype,
            device=device,
            batch_size=int(args.hard_negative_batch_size),
            normalize_embeddings=True,
        )
        hard_negative_map, hard_negative_stats = mine_semantic_hard_negative_map(
            item_ids=semantic_item_ids,
            semantic_embeddings=semantic_embeddings,
            positive_adjacency=positive_adjacency,
            topk=int(args.hard_negative_topk),
            chunk_size=int(args.hard_negative_chunk_size),
            min_similarity=float(args.hard_negative_min_similarity),
        )
        filtered_negative_item_ids = [int(hard_negative_map.get(int(source_item_id), 0)) for source_item_id in filtered_source]

    train_dataset, eval_dataset = build_train_eval_split(
        source_item_ids=filtered_source,
        target_item_ids=filtered_target,
        pair_weights=selected_pair_weights,
        eval_ratio=float(args.eval_ratio),
        max_eval_pairs=int(args.max_eval_pairs),
        seed=int(args.seed),
        negative_item_ids=filtered_negative_item_ids,
    )
    collator = CooccurrencePairCollator(hidden_cache=hidden_cache, hidden_dtype=cache_hidden_dtype)
    train_sampler = None
    train_shuffle = True
    if positive_sampling_mode != "uniform":
        train_sampler = build_train_sampler(train_dataset.pair_weights)
        train_shuffle = False
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(args.batch_size),
        shuffle=train_shuffle,
        sampler=train_sampler,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        collate_fn=collator,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=int(args.eval_batch_size),
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        collate_fn=collator,
    )
    model = CooccurrenceContrastiveModel(
        semantic_head_path=semantic_head_path,
        hidden_size=hidden_size,
        temperature=float(args.temperature),
        learnable_temperature=as_bool(args.learnable_temperature),
        logit_scale_max=float(args.logit_scale_max),
        normalize_embeddings=as_bool(args.normalize_embeddings),
        forward_chunk_size=int(args.forward_chunk_size),
        gradient_checkpointing=as_bool(args.gradient_checkpointing),
        cooccurrence_head_init=str(args.cooccurrence_head_init),
        device=device,
        dtype=torch.float32,
    )
    model.to(device)
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable_parameters:
        raise ValueError("No trainable parameters found for Stage2 training")
    optimizer = torch.optim.AdamW(trainable_parameters, lr=float(args.learning_rate), weight_decay=float(args.weight_decay))
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=int(max_steps * float(args.warmup_ratio)),
        num_training_steps=max_steps,
    )

    resume_checkpoint_dir = ""
    resume_state: Dict[str, object] = {}
    if str(getattr(args, "resume_from_checkpoint", "")).strip():
        resume_checkpoint_dir = resolve_stage2_checkpoint_path(str(args.resume_from_checkpoint).strip())
        resume_state = _restore_resume_state(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            resume_checkpoint_dir=resume_checkpoint_dir,
            hidden_size=hidden_size,
            device=device,
        )

    metadata = {
        "base_model": args.base_model,
        "hidden_cache_path": args.hidden_cache_path,
        "pairs_path": args.pairs_path,
        "train_file": pair_payload.get("train_file", ""),
        "dataset": dataset_name,
        "pair_mining_mode": pair_mining_mode,
        "window_size": None if pair_window_size is None else int(pair_window_size),
        "min_count": int(pair_payload.get("min_count", 3)),
        "sequence_count": int(pair_payload.get("sequence_count", 0)),
        "sequence_unit": str(pair_payload.get("sequence_unit", "user")),
        "pre_filter_raw_pair_count": int(pair_payload.get("pre_filter_raw_pair_count", pair_payload.get("raw_pair_count", len(filtered_source)))),
        "pre_filter_unique_pair_count": int(pair_payload.get("pre_filter_unique_pair_count", pair_payload.get("unique_pair_count", len(filtered_source)))),
        "raw_pair_count": int(pair_payload.get("raw_pair_count", len(filtered_source))),
        "unique_pair_count": int(pair_payload.get("unique_pair_count", len(filtered_source))),
        "expanded_pair_count": int(pair_payload.get("expanded_pair_count", len(filtered_source))),
        "expand_bidirectional_pairs": bool(pair_payload.get("expand_bidirectional_pairs", False)),
        "contains_ppmi": bool(pair_payload.get("contains_ppmi", False)),
        "ppmi_alpha": float(pair_payload.get("ppmi_alpha", 0.75)),
        "positive_sampling_mode": positive_sampling_mode,
        "hard_negative_mode": hard_negative_mode,
        "hard_negative_topk": int(args.hard_negative_topk),
        "hard_negative_chunk_size": int(args.hard_negative_chunk_size),
        "hard_negative_batch_size": int(args.hard_negative_batch_size),
        "hard_negative_min_similarity": float(args.hard_negative_min_similarity),
        **hard_negative_stats,
        "filtered_pair_count": int(len(filtered_source)),
        "filtered_raw_pair_count": int(sum(filtered_counts)) if filtered_counts else int(len(filtered_source)),
        "filtered_ppmi_mean": float(np.mean(filtered_ppmi)) if filtered_ppmi else float("nan"),
        "train_pair_count": int(len(train_dataset)),
        "eval_pair_count": int(len(eval_dataset)),
        "train_sampler_mode": positive_sampling_mode,
        "batch_size": int(args.batch_size),
        "eval_batch_size": int(args.eval_batch_size),
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "warmup_ratio": float(args.warmup_ratio),
        "max_steps": max_steps,
        "logging_steps": int(args.logging_steps),
        "eval_steps": int(args.eval_steps),
        "save_steps": int(args.save_steps),
        "early_stopping_patience": early_stopping_patience,
        "eval_ratio": float(args.eval_ratio),
        "max_eval_pairs": int(args.max_eval_pairs),
        "temperature": float(args.temperature),
        "learnable_temperature": bool(as_bool(args.learnable_temperature)),
        "logit_scale_max": float(args.logit_scale_max),
        "normalize_embeddings": bool(as_bool(args.normalize_embeddings)),
        "forward_chunk_size": int(args.forward_chunk_size),
        "gradient_checkpointing": bool(as_bool(args.gradient_checkpointing)),
        "seed": int(args.seed),
        "save_only_best": bool(as_bool(args.save_only_best)),
        "semantic_head_source": model.semantic_head_source,
        "cooccurrence_head_init": model.cooccurrence_head_init,
        "cooccurrence_head_init_source": model.cooccurrence_head_source,
        "stage2_method": "tcl",
        "stage2_variant": "stage2_tcl",
        "resume_from_checkpoint": os.path.abspath(resume_checkpoint_dir) if resume_checkpoint_dir else "",
    }
    write_stage2_config(args.output_dir, metadata)

    global_step = int(resume_state.get("step", 0)) if resume_state else 0
    best_eval_recall_at_10 = float(resume_state.get("best_eval_recall_at_10", float("-inf"))) if resume_state else float("-inf")
    best_eval_loss = float(resume_state.get("best_eval_loss", float("inf"))) if resume_state else float("inf")
    best_checkpoint_dir = os.path.realpath(os.path.join(args.output_dir, "best_checkpoint")) if os.path.exists(os.path.join(args.output_dir, "best_checkpoint")) else ""
    no_improve_rounds = _infer_no_improve_rounds(args.output_dir, int(args.eval_steps)) if resume_state else 0
    running_losses: List[float] = []
    start_time = time.time()
    train_iter = iter(train_loader)
    autocast_enabled = bool(as_bool(args.bf16)) and device.type == "cuda"
    optimizer.zero_grad(set_to_none=True)
    model.train()

    while global_step < max_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        batch = {
            key: (value.to(device) if isinstance(value, torch.Tensor) else value)
            for key, value in batch.items()
        }
        labels = torch.arange(batch["source_hidden_states"].size(0), device=device)
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=autocast_enabled,
        ):
            outputs = model.compute_loss_components(
                source_hidden_states=batch["source_hidden_states"],
                source_attention_mask=batch["source_attention_mask"],
                target_hidden_states=batch["target_hidden_states"],
                target_attention_mask=batch["target_attention_mask"],
                labels=labels,
                negative_hidden_states=batch.get("negative_hidden_states"),
                negative_attention_mask=batch.get("negative_attention_mask"),
            )
            loss = outputs["loss"]
        loss.backward()
        if float(args.max_grad_norm) > 0:
            torch.nn.utils.clip_grad_norm_(trainable_parameters, float(args.max_grad_norm))
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        global_step += 1
        running_losses.append(float(loss.detach().cpu().item()))

        if global_step % max(1, int(args.logging_steps)) == 0:
            elapsed = max(1e-6, time.time() - start_time)
            logging_window = max(1, int(args.logging_steps))
            log = {
                "step": int(global_step),
                "train_loss": round(float(np.mean(running_losses[-logging_window:])), 6),
                "lr": optimizer.param_groups[0]["lr"],
                "steps_per_sec": round((global_step - int(resume_state.get("step", 0) or 0)) / elapsed, 4) if resume_state else round(global_step / elapsed, 4),
                "elapsed_sec": round(elapsed, 1),
            }
            print(json.dumps(log, ensure_ascii=False))

        should_eval = int(args.eval_steps) > 0 and (global_step % int(args.eval_steps) == 0)
        if should_eval:
            eval_stats = evaluate(
                model=model,
                dataloader=eval_loader,
                device=device,
                bf16=bool(as_bool(args.bf16)),
            )
            improved = (
                float(eval_stats["eval_recall_at_10"]) > best_eval_recall_at_10
                or (
                    float(eval_stats["eval_recall_at_10"]) == best_eval_recall_at_10
                    and float(eval_stats["eval_loss"]) < best_eval_loss
                )
            )
            print(json.dumps({
                "step": int(global_step),
                **{key: round(float(value), 6) for key, value in eval_stats.items()},
                "best_eval_recall_at_10": None if best_eval_recall_at_10 == float("-inf") else round(float(best_eval_recall_at_10), 6),
                "best_eval_loss": None if best_eval_loss == float("inf") else round(float(best_eval_loss), 6),
                "improved": bool(improved),
            }, ensure_ascii=False))
            if improved:
                best_eval_recall_at_10 = float(eval_stats["eval_recall_at_10"])
                best_eval_loss = float(eval_stats["eval_loss"])
                no_improve_rounds = 0
                train_state = {
                    "step": int(global_step),
                    "best_eval_recall_at_10": float(best_eval_recall_at_10),
                    "best_eval_loss": float(best_eval_loss),
                    "last_eval_loss": float(eval_stats["eval_loss"]),
                    "last_eval_recall_at_1": float(eval_stats["eval_recall_at_1"]),
                    "last_eval_recall_at_5": float(eval_stats["eval_recall_at_5"]),
                    "last_eval_recall_at_10": float(eval_stats["eval_recall_at_10"]),
                    "no_improve_rounds": int(no_improve_rounds),
                }
                best_checkpoint_dir = save_checkpoint(
                    output_dir=args.output_dir,
                    step=global_step,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    train_state=train_state,
                )
                update_best_checkpoint(args.output_dir, best_checkpoint_dir)
                if bool(as_bool(args.save_only_best)):
                    prune_non_best_checkpoints(args.output_dir, best_checkpoint_dir)
            else:
                no_improve_rounds += 1
                if early_stopping_patience > 0 and no_improve_rounds >= early_stopping_patience:
                    print(
                        f"[stage2-llmalign] early stopping at step={global_step}, "
                        f"best_eval_recall_at_10={best_eval_recall_at_10:.6f}"
                    )
                    break

        if (not bool(as_bool(args.save_only_best))) and int(args.save_steps) > 0 and (global_step % int(args.save_steps) == 0):
            train_state = {
                "step": int(global_step),
                "best_eval_recall_at_10": float(best_eval_recall_at_10) if best_eval_recall_at_10 != float("-inf") else float("nan"),
                "best_eval_loss": float(best_eval_loss) if best_eval_loss != float("inf") else float("nan"),
                "last_eval_loss": float("nan"),
                "last_eval_recall_at_1": float("nan"),
                "last_eval_recall_at_5": float("nan"),
                "last_eval_recall_at_10": float("nan"),
                "no_improve_rounds": int(no_improve_rounds),
            }
            _ = save_checkpoint(
                output_dir=args.output_dir,
                step=global_step,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                train_state=train_state,
            )

    if is_baby_dataset_name(dataset_name) and int(global_step) > 0:
        train_state = {
            "step": int(global_step),
            "best_eval_recall_at_10": float(best_eval_recall_at_10) if best_eval_recall_at_10 != float("-inf") else float("nan"),
            "best_eval_loss": float(best_eval_loss) if best_eval_loss != float("inf") else float("nan"),
            "last_eval_loss": float("nan"),
            "last_eval_recall_at_1": float("nan"),
            "last_eval_recall_at_5": float("nan"),
            "last_eval_recall_at_10": float("nan"),
            "no_improve_rounds": int(no_improve_rounds),
        }
        _ = save_checkpoint(
            output_dir=args.output_dir,
            step=int(global_step),
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            train_state=train_state,
        )

    if not best_checkpoint_dir:
        train_state = {
            "step": int(global_step),
            "best_eval_recall_at_10": float(best_eval_recall_at_10) if best_eval_recall_at_10 != float("-inf") else float("nan"),
            "best_eval_loss": float(best_eval_loss) if best_eval_loss != float("inf") else float("nan"),
            "last_eval_loss": float("nan"),
            "last_eval_recall_at_1": float("nan"),
            "last_eval_recall_at_5": float("nan"),
            "last_eval_recall_at_10": float("nan"),
            "no_improve_rounds": int(no_improve_rounds),
        }
        best_checkpoint_dir = save_checkpoint(
            output_dir=args.output_dir,
            step=max(1, int(global_step)),
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            train_state=train_state,
        )
        update_best_checkpoint(args.output_dir, best_checkpoint_dir)
        if bool(as_bool(args.save_only_best)):
            prune_non_best_checkpoints(args.output_dir, best_checkpoint_dir)

    summary = {
        "base_model": args.base_model,
        "output_dir": os.path.abspath(args.output_dir),
        "best_checkpoint": best_checkpoint_dir,
        "pair_mining_mode": pair_mining_mode,
        "window_size": None if pair_window_size is None else int(pair_window_size),
        "positive_sampling_mode": positive_sampling_mode,
        "hard_negative_mode": hard_negative_mode,
        **hard_negative_stats,
        "cooccurrence_head_init": model.cooccurrence_head_init,
        "save_only_best": bool(as_bool(args.save_only_best)),
        "best_eval_recall_at_10": None if best_eval_recall_at_10 == float("-inf") else float(best_eval_recall_at_10),
        "best_eval_loss": None if best_eval_loss == float("inf") else float(best_eval_loss),
        "final_step": int(global_step),
        "elapsed_seconds": round(time.time() - start_time, 3),
        "resume_from_checkpoint": os.path.abspath(resume_checkpoint_dir) if resume_checkpoint_dir else "",
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage2 LLMAlign TCL utilities.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    parser_cache = subparsers.add_parser("cache_hidden", help="Cache frozen backbone hidden states for all item titles.")
    parser_cache.add_argument("--dataset", type=str, required=True)
    parser_cache.add_argument("--base_model", type=str, required=True)
    parser_cache.add_argument("--output_path", type=str, required=True)
    parser_cache.add_argument("--batch_size", type=int, default=32)
    parser_cache.add_argument("--max_length", type=int, default=None)
    parser_cache.add_argument("--item_prefix", type=str, default=None)
    parser_cache.add_argument("--item_suffix", type=str, default=None)
    parser_cache.add_argument("--attention_mask_type", type=str, default=None)
    parser_cache.add_argument("--cache_dtype", type=str, default="fp16")
    parser_cache.add_argument("--bf16", type=int, default=1)
    parser_cache.set_defaults(func=command_cache_hidden)

    parser_pairs = subparsers.add_parser("mine_pairs", help="Mine TCL item-item pairs from the training CSV.")
    parser_pairs.add_argument("--train_file", type=str, required=True)
    parser_pairs.add_argument("--output_path", type=str, required=True)
    parser_pairs.add_argument("--pair_mining_mode", type=str, default="sliding_window", choices=("global", "sliding_window"))
    parser_pairs.add_argument("--window_size", type=int, default=3)
    parser_pairs.add_argument("--min_count", type=int, default=3)
    parser_pairs.add_argument("--expand_bidirectional_pairs", type=int, default=1)
    parser_pairs.add_argument("--compute_ppmi", type=int, default=1)
    parser_pairs.add_argument("--ppmi_alpha", type=float, default=0.75)
    parser_pairs.add_argument("--sample", type=int, default=-1)
    parser_pairs.add_argument("--seed", type=int, default=42)
    parser_pairs.set_defaults(func=command_mine_pairs)

    parser_train = subparsers.add_parser("train", help="Train the TCL head on cached item hidden states.")
    parser_train.add_argument("--base_model", type=str, required=True)
    parser_train.add_argument("--hidden_cache_path", type=str, required=True)
    parser_train.add_argument("--pairs_path", type=str, required=True)
    parser_train.add_argument("--output_dir", type=str, required=True)
    parser_train.add_argument("--batch_size", type=int, default=1024)
    parser_train.add_argument("--eval_batch_size", type=int, default=1024)
    parser_train.add_argument("--learning_rate", type=float, default=1e-4)
    parser_train.add_argument("--weight_decay", type=float, default=0.01)
    parser_train.add_argument("--warmup_ratio", type=float, default=0.05)
    parser_train.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Defaults to 500 for Games/Movies-like datasets and 250 for Baby.",
    )
    parser_train.add_argument("--logging_steps", type=int, default=20)
    parser_train.add_argument("--eval_steps", type=int, default=50)
    parser_train.add_argument("--save_steps", type=int, default=0)
    parser_train.add_argument(
        "--early_stopping_patience",
        type=int,
        default=0,
        help="Set to 0 to disable early stopping.",
    )
    parser_train.add_argument("--eval_ratio", type=float, default=0.05)
    parser_train.add_argument("--max_eval_pairs", type=int, default=50000)
    parser_train.add_argument("--temperature", type=float, default=0.05)
    parser_train.add_argument("--learnable_temperature", type=int, default=1)
    parser_train.add_argument("--logit_scale_max", type=float, default=100.0)
    parser_train.add_argument("--save_only_best", type=int, default=1)
    parser_train.add_argument("--normalize_embeddings", type=int, default=1)
    parser_train.add_argument("--positive_sampling_mode", type=str, default="ppmi", choices=("uniform", "count", "ppmi"))
    parser_train.add_argument("--hard_negative_mode", type=str, default="none", choices=("none", "semantic"))
    parser_train.add_argument("--hard_negative_topk", type=int, default=20)
    parser_train.add_argument("--hard_negative_chunk_size", type=int, default=512)
    parser_train.add_argument("--hard_negative_batch_size", type=int, default=512)
    parser_train.add_argument("--hard_negative_min_similarity", type=float, default=-1.0)
    parser_train.add_argument("--forward_chunk_size", type=int, default=0)
    parser_train.add_argument("--gradient_checkpointing", type=int, default=0)
    parser_train.add_argument(
        "--cooccurrence_head_init",
        type=str,
        default="semantic",
        choices=("semantic", "random"),
        help="Initialize the TCL head from the CPA latent head weights or from a fresh random init with the same architecture.",
    )
    parser_train.add_argument("--bf16", type=int, default=1)
    parser_train.add_argument("--max_grad_norm", type=float, default=1.0)
    parser_train.add_argument("--seed", type=int, default=42)
    parser_train.add_argument("--resume_from_checkpoint", type=str, default="")
    parser_train.set_defaults(func=command_train)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
