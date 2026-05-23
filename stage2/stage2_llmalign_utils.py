import ast
import json
import os
import random
import sys
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

if __package__ in {None, ""}:
    _THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    _REPO_ROOT = os.path.dirname(_THIS_DIR)
    if _REPO_ROOT not in sys.path:
        sys.path.insert(0, _REPO_ROOT)

from stage1.latent_attention import LatentAttentionPooling, load_latent_attention_head


DATASET_NAME_MAPPINGS = {
    "Games_5core": "Video_Games/5-core/downstream",
    "Arts_5core": "Arts_Crafts_and_Sewing/5-core/downstream",
    "Baby_5core": "Baby_Products/5-core/downstream",
}

STAGE1_CONFIG_NAME = "cpa_encoder_config.json"
STAGE2_CONFIG_NAME = "stage2_llmalign_config.json"
STAGE2_STATE_NAME = "stage2_llmalign_state.pt"
PAIR_MINING_MODE_ALIASES = {
    "cooccurrence": "global",
    "global": "global",
    "global_cooccurrence": "global",
    "sliding_window": "sliding_window",
    "sliding_window_cooccurrence": "sliding_window",
    "window": "sliding_window",
}


def as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def is_baby_dataset_name(dataset_name: str) -> bool:
    key = str(dataset_name or "").strip().lower().replace("-", "_")
    return key in {"baby", "baby_5core", "baby_products"} or key.startswith("baby_products/")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalize_pair_mining_mode(pair_mining_mode: str) -> str:
    key = str(pair_mining_mode or "").strip().lower()
    if key not in PAIR_MINING_MODE_ALIASES:
        supported = ", ".join(sorted(PAIR_MINING_MODE_ALIASES.keys()))
        raise ValueError(f"Unsupported pair_mining_mode: {pair_mining_mode}. Supported: {supported}")
    return PAIR_MINING_MODE_ALIASES[key]


def pair_mining_mode_is_symmetric(pair_mining_mode: str) -> bool:
    canonical_mode = normalize_pair_mining_mode(pair_mining_mode)
    return canonical_mode in {"global", "sliding_window"}


def parse_int_list(raw_value) -> List[int]:
    if isinstance(raw_value, list):
        values = raw_value
    elif raw_value is None:
        return []
    else:
        raw_text = str(raw_value).strip()
        if not raw_text:
            return []
        if raw_text.startswith("[") and raw_text.endswith("]"):
            try:
                values = ast.literal_eval(raw_text)
            except (ValueError, SyntaxError):
                return []
        else:
            return []
    output: List[int] = []
    for value in values:
        try:
            output.append(int(value))
        except (TypeError, ValueError):
            continue
    return output


def build_bidirectional_attention_mask_dict(attention_mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    if attention_mask.ndim != 2:
        raise ValueError(
            f"Expected 2D attention_mask [B, L], got shape={tuple(attention_mask.shape)}"
        )
    if not torch.is_floating_point(torch.empty((), dtype=dtype)):
        dtype = torch.float32
    pad_mask = attention_mask.to(dtype=dtype)
    min_value = torch.finfo(dtype).min
    additive_mask = (1.0 - pad_mask).unsqueeze(1).unsqueeze(2) * min_value
    seq_len = attention_mask.size(1)
    return additive_mask.expand(-1, 1, seq_len, -1).contiguous()


def resolve_hidden_size(config) -> int:
    hidden_size = getattr(config, "hidden_size", None)
    if hidden_size is not None:
        return int(hidden_size)
    text_cfg = getattr(config, "text_config", None)
    if text_cfg is not None:
        cfg_hidden = getattr(text_cfg, "hidden_size", None)
        if cfg_hidden is not None:
            return int(cfg_hidden)
        if isinstance(text_cfg, dict) and "hidden_size" in text_cfg:
            return int(text_cfg["hidden_size"])
    raise ValueError(f"Cannot resolve hidden_size from config type={type(config).__name__}")


def ensure_tokenizer_padding(tokenizer) -> None:
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer has no pad/eos token, cannot pad safely.")
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"


def dataset_to_item_titles_path(dataset_name: str) -> str:
    if dataset_name not in DATASET_NAME_MAPPINGS:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    return os.path.join("data", DATASET_NAME_MAPPINGS[dataset_name], "item_titles.json")


def load_item_titles(dataset_name: str) -> Tuple[Dict[int, str], int]:
    path = dataset_to_item_titles_path(dataset_name)
    with open(path, "r", encoding="utf-8") as file:
        payload = json.load(file)
    item_id_to_title: Dict[int, str] = {}
    for key, value in payload.items():
        item_id = int(key)
        if item_id <= 0:
            continue
        item_id_to_title[item_id] = str(value)
    if not item_id_to_title:
        raise ValueError(f"No valid item titles found in {path}")
    return item_id_to_title, max(item_id_to_title)


def resolve_stage1_root(base_model_path: str) -> str:
    normalized = os.path.abspath(base_model_path)
    candidates = [normalized]
    parent = os.path.dirname(normalized)
    if parent and parent not in candidates:
        candidates.append(parent)
    for candidate in candidates:
        if os.path.isfile(os.path.join(candidate, STAGE1_CONFIG_NAME)):
            return candidate
    return parent if os.path.basename(normalized).startswith("checkpoint-") else normalized


def load_stage1_config(base_model_path: str) -> Dict[str, object]:
    root = resolve_stage1_root(base_model_path)
    config_path = os.path.join(root, STAGE1_CONFIG_NAME)
    if not os.path.isfile(config_path):
        return {}
    with open(config_path, "r", encoding="utf-8") as file:
        return json.load(file)


def resolve_stage1_item_settings(
    base_model_path: str,
    item_prefix: Optional[str] = None,
    item_suffix: Optional[str] = None,
    attention_mask_type: Optional[str] = None,
    max_length: Optional[int] = None,
) -> Dict[str, object]:
    cfg = load_stage1_config(base_model_path)
    resolved_attention_mask_type = (
        attention_mask_type
        if attention_mask_type is not None
        else cfg.get("item_attention_mask_type") or cfg.get("attention_mask_type") or "bidirectional"
    )
    resolved_max_length = (
        int(max_length)
        if max_length is not None
        else int(cfg.get("max_item_len", 128))
    )
    return {
        "item_prefix": item_prefix if item_prefix is not None else str(cfg.get("item_prefix", "")),
        "item_suffix": item_suffix if item_suffix is not None else str(cfg.get("item_suffix", "")),
        "attention_mask_type": str(resolved_attention_mask_type).lower().strip(),
        "max_length": int(resolved_max_length),
        "embedding_head_type": str(cfg.get("embedding_head_type", "latent_attention")),
        "latent_num_latents": int(cfg.get("latent_num_latents", 128)),
        "latent_num_cross_heads": int(cfg.get("latent_num_cross_heads", 8)),
        "latent_cross_dim_head": int(cfg.get("latent_cross_dim_head", 64)),
        "latent_ff_mult": int(cfg.get("latent_ff_mult", 4)),
        "latent_dim": int(cfg.get("latent_dim", -1)),
    }


def compose_item_text(title: str, item_prefix: str, item_suffix: str) -> str:
    return f"{item_prefix}{str(title)}{item_suffix}"


def pad_hidden_states(
    hidden_states_list: Sequence[torch.Tensor],
    attention_masks: Sequence[torch.Tensor],
    hidden_dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if len(hidden_states_list) != len(attention_masks):
        raise ValueError("hidden_states_list and attention_masks must have the same length")
    batch_size = len(hidden_states_list)
    if batch_size == 0:
        raise ValueError("Cannot pad an empty batch")
    max_len = max(int(mask.shape[0]) for mask in attention_masks)
    hidden_size = int(hidden_states_list[0].shape[-1])
    padded_hidden = torch.zeros(batch_size, max_len, hidden_size, dtype=hidden_dtype)
    padded_mask = torch.zeros(batch_size, max_len, dtype=torch.long)
    for idx, (hidden_states, attention_mask) in enumerate(zip(hidden_states_list, attention_masks)):
        seq_len = int(attention_mask.shape[0])
        padded_hidden[idx, :seq_len] = hidden_states.to(dtype=hidden_dtype)
        padded_mask[idx, :seq_len] = attention_mask.to(dtype=torch.long)
    return padded_hidden, padded_mask


def _read_stage2_dataset_name(stage2_path: str) -> str:
    normalized = os.path.abspath(stage2_path)
    config_paths: List[str] = []
    if os.path.isdir(normalized):
        config_paths.append(os.path.join(normalized, STAGE2_CONFIG_NAME))
        if os.path.basename(normalized).startswith("checkpoint-"):
            config_paths.append(os.path.join(os.path.dirname(normalized), STAGE2_CONFIG_NAME))
    else:
        parent = os.path.dirname(normalized)
        config_paths.append(os.path.join(parent, STAGE2_CONFIG_NAME))
        config_paths.append(os.path.join(os.path.dirname(parent), STAGE2_CONFIG_NAME))
    for config_path in _dedupe_paths(config_paths):
        if not os.path.isfile(config_path):
            continue
        try:
            with open(config_path, "r", encoding="utf-8") as file:
                payload = json.load(file)
            return str(payload.get("dataset", "")).strip()
        except (OSError, json.JSONDecodeError):
            continue
    return ""


def _sorted_stage2_checkpoints(stage2_dir: str) -> List[str]:
    checkpoints = [
        os.path.join(stage2_dir, name)
        for name in os.listdir(stage2_dir)
        if name.startswith("checkpoint-") and os.path.isdir(os.path.join(stage2_dir, name))
    ]
    checkpoints.sort(key=lambda path: int(os.path.basename(path).split("-")[-1]))
    return checkpoints


def resolve_stage2_checkpoint_path(stage2_path: str, dataset_name: str = "") -> str:
    normalized = os.path.abspath(stage2_path)
    if os.path.isfile(normalized):
        return normalized
    if not os.path.isdir(normalized):
        raise FileNotFoundError(f"Stage2 checkpoint path not found: {stage2_path}")
    if os.path.isfile(os.path.join(normalized, "latent_attention_head.pt")):
        return normalized
    checkpoints = _sorted_stage2_checkpoints(normalized)
    resolved_dataset_name = str(dataset_name or "").strip() or _read_stage2_dataset_name(normalized)
    if checkpoints and is_baby_dataset_name(resolved_dataset_name):
        return checkpoints[-1]
    best_link = os.path.join(normalized, "best_checkpoint")
    if os.path.islink(best_link) or os.path.isdir(best_link):
        return os.path.realpath(best_link)
    if checkpoints:
        return checkpoints[-1]
    raise FileNotFoundError(f"Cannot resolve stage2 checkpoint under {stage2_path}")


def _dedupe_paths(paths: Sequence[str]) -> List[str]:
    ordered_paths: List[str] = []
    seen_paths = set()
    for path in paths:
        normalized = os.path.abspath(path)
        if normalized in seen_paths:
            continue
        seen_paths.add(normalized)
        ordered_paths.append(normalized)
    return ordered_paths


def resolve_stage2_config_candidates(stage2_path: str) -> List[str]:
    normalized = os.path.abspath(stage2_path)
    candidates: List[str] = []
    if os.path.isdir(normalized):
        candidates.append(os.path.join(normalized, STAGE2_CONFIG_NAME))
        checkpoint_path = resolve_stage2_checkpoint_path(normalized)
        checkpoint_root = os.path.dirname(checkpoint_path)
        candidates.append(os.path.join(checkpoint_root, STAGE2_CONFIG_NAME))
    else:
        checkpoint_dir = os.path.dirname(normalized)
        checkpoint_root = os.path.dirname(checkpoint_dir)
        candidates.append(os.path.join(checkpoint_dir, STAGE2_CONFIG_NAME))
        candidates.append(os.path.join(checkpoint_root, STAGE2_CONFIG_NAME))
    return _dedupe_paths(candidates)


def resolve_stage2_config_path(stage2_path: str) -> str:
    for candidate in resolve_stage2_config_candidates(stage2_path):
        if os.path.isfile(candidate):
            return candidate
    normalized = os.path.abspath(stage2_path)
    base_dir = normalized if os.path.isdir(normalized) else os.path.dirname(normalized)
    return os.path.join(base_dir, STAGE2_CONFIG_NAME)


def resolve_stage2_state_path(checkpoint_dir: str) -> str:
    normalized = os.path.abspath(checkpoint_dir)
    return os.path.join(normalized, STAGE2_STATE_NAME)


def load_stage2_config(stage2_path: str) -> Dict[str, object]:
    config_path = resolve_stage2_config_path(stage2_path)
    if os.path.isfile(config_path):
        with open(config_path, "r", encoding="utf-8") as file:
            return json.load(file)
    return {}


class FrozenStage1BackboneEncoder:
    def __init__(
        self,
        base_model_path: str,
        attention_mask_type: str,
        torch_dtype: Optional[torch.dtype] = None,
    ) -> None:
        self.base_model_path = os.path.abspath(base_model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch_dtype is None:
            torch_dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.base_model_path,
                trust_remote_code=True,
            )
        except Exception as exc:
            print(f"[stage2-llmalign][warn] fast tokenizer load failed for {self.base_model_path}: {exc}")
            print("[stage2-llmalign][warn] falling back to slow tokenizer (use_fast=False).")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.base_model_path,
                trust_remote_code=True,
                use_fast=False,
            )
        ensure_tokenizer_padding(self.tokenizer)
        self.attention_mask_type = str(attention_mask_type).lower().strip()
        self.model = AutoModel.from_pretrained(
            self.base_model_path,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )
        self.model.to(self.device)
        self.model.eval()
        self.model_dtype = next(self.model.parameters()).dtype
        self.hidden_size = resolve_hidden_size(self.model.config)

    def encode_hidden_states(
        self,
        texts: Sequence[str],
        batch_size: int,
        max_length: int,
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        outputs: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for start in range(0, len(texts), batch_size):
            batch_texts = [str(text) for text in texts[start:start + batch_size]]
            batch = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
                return_attention_mask=True,
            )
            batch = {key: value.to(self.device) for key, value in batch.items()}
            model_attention_mask = batch["attention_mask"]
            if self.attention_mask_type == "bidirectional":
                model_attention_mask = build_bidirectional_attention_mask_dict(
                    batch["attention_mask"],
                    dtype=self.model_dtype,
                )
            with torch.no_grad():
                model_outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=model_attention_mask,
                    return_dict=True,
                )
            hidden_states = model_outputs.last_hidden_state.detach().cpu()
            attention_masks = batch["attention_mask"].detach().cpu()
            for hidden_state, attention_mask in zip(hidden_states, attention_masks):
                seq_len = int(attention_mask.to(dtype=torch.long).sum().item())
                outputs.append(
                    (
                        hidden_state[:seq_len].contiguous(),
                        attention_mask[:seq_len].contiguous(),
                    )
                )
        return outputs


def load_latent_head_module(
    checkpoint_path: str,
    hidden_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[LatentAttentionPooling, Dict[str, int], str]:
    saved_cfg, state_dict, resolved_path = load_latent_attention_head(checkpoint_path, map_location="cpu")
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
    missing, unexpected = head.load_state_dict(state_dict, strict=False)
    unexpected = [key for key in unexpected if not key.startswith("readout_gate.")]
    if missing or unexpected:
        raise ValueError(
            f"Failed to load latent attention head from {resolved_path}, missing={missing}, unexpected={unexpected}"
        )
    head.to(device=device, dtype=dtype)
    return head, dict(saved_cfg), resolved_path


def _extract_full_sequence_from_row(row) -> List[int]:
    history_item_ids = [
        item_id
        for item_id in parse_int_list(getattr(row, "history_item_id", None))
        if int(item_id) > 0
    ]
    try:
        target_item_id = int(getattr(row, "item_id", -1))
    except (TypeError, ValueError):
        return []
    if target_item_id <= 0:
        return history_item_ids
    return history_item_ids + [target_item_id]


def _collect_user_level_sequences(data: pd.DataFrame) -> List[List[int]]:
    user_to_sequence: Dict[str, Tuple[List[int], int]] = {}
    anonymous_sequences: List[List[int]] = []
    for row in data.itertuples(index=False):
        full_sequence = _extract_full_sequence_from_row(row)
        if len(full_sequence) < 2:
            continue
        raw_user_id = getattr(row, "user_id", None)
        user_id = str(raw_user_id).strip() if raw_user_id is not None else ""
        if not user_id or user_id.lower() == "nan":
            anonymous_sequences.append(full_sequence)
            continue
        try:
            row_timestamp = int(getattr(row, "timestamp", -1))
        except (TypeError, ValueError):
            row_timestamp = -1
        prev = user_to_sequence.get(user_id)
        if prev is None:
            user_to_sequence[user_id] = (full_sequence, row_timestamp)
            continue
        prev_sequence, prev_timestamp = prev
        if len(full_sequence) > len(prev_sequence) or (
            len(full_sequence) == len(prev_sequence) and row_timestamp >= prev_timestamp
        ):
            user_to_sequence[user_id] = (full_sequence, row_timestamp)
    return [sequence for sequence, _ in user_to_sequence.values()] + anonymous_sequences


def _sample_sequences(
    sequences: Sequence[List[int]],
    sample: int,
    seed: int,
) -> List[List[int]]:
    if sample <= 0 or sample >= len(sequences):
        return [list(sequence) for sequence in sequences]
    rng = random.Random(seed)
    selected_indices = sorted(rng.sample(range(len(sequences)), k=int(sample)))
    return [list(sequences[idx]) for idx in selected_indices]


def _update_pair_counts_from_unique_items(
    unique_item_ids: Sequence[int],
    pair_counts: Dict[Tuple[int, int], int],
) -> None:
    for source_idx in range(len(unique_item_ids)):
        source_item_id = int(unique_item_ids[source_idx])
        for target_idx in range(source_idx + 1, len(unique_item_ids)):
            target_item_id = int(unique_item_ids[target_idx])
            if target_item_id <= 0 or target_item_id == source_item_id:
                continue
            pair_key = (source_item_id, target_item_id)
            pair_counts[pair_key] = pair_counts.get(pair_key, 0) + 1


def _iter_sliding_windows(
    sequence: Sequence[int],
    window_size: int,
) -> Iterable[List[int]]:
    effective_window_size = int(window_size)
    if effective_window_size < 2:
        raise ValueError(f"window_size must be >= 2 for sliding_window mining, got {window_size}")
    cleaned_sequence = [int(item_id) for item_id in sequence if int(item_id) > 0]
    if len(cleaned_sequence) < 2:
        return
    if len(cleaned_sequence) <= effective_window_size:
        yield cleaned_sequence
        return
    last_start = len(cleaned_sequence) - effective_window_size
    for start in range(last_start + 1):
        yield cleaned_sequence[start:start + effective_window_size]


def _compute_cooccurrence_ppmi_scores(
    pair_counts: Dict[Tuple[int, int], int],
    smoothing_alpha: float,
) -> Tuple[Dict[Tuple[int, int], float], Dict[int, float]]:
    if not pair_counts:
        return {}, {}
    alpha = float(smoothing_alpha)
    if alpha <= 0:
        raise ValueError(f"ppmi_alpha must be positive, got {smoothing_alpha}")
    item_degree: Dict[int, float] = {}
    total_directed_count = 0.0
    for (source_item_id, target_item_id), raw_count in pair_counts.items():
        count = float(raw_count)
        if count <= 0:
            continue
        item_degree[int(source_item_id)] = item_degree.get(int(source_item_id), 0.0) + count
        item_degree[int(target_item_id)] = item_degree.get(int(target_item_id), 0.0) + count
        total_directed_count += 2.0 * count
    if total_directed_count <= 0:
        return {}, item_degree
    smoothed_context = {
        int(item_id): float(degree) ** alpha
        for item_id, degree in item_degree.items()
        if float(degree) > 0
    }
    smoothed_context_total = float(sum(smoothed_context.values()))
    if smoothed_context_total <= 0:
        return {}, item_degree
    eps = 1e-12
    ppmi_scores: Dict[Tuple[int, int], float] = {}
    for (source_item_id, target_item_id), raw_count in pair_counts.items():
        count = float(raw_count)
        if count <= 0:
            continue
        directed_pair_prob = max(count / total_directed_count, eps)
        source_prob = max(item_degree[int(source_item_id)] / total_directed_count, eps)
        target_prob = max(smoothed_context[int(target_item_id)] / smoothed_context_total, eps)
        reverse_source_prob = max(item_degree[int(target_item_id)] / total_directed_count, eps)
        reverse_target_prob = max(smoothed_context[int(source_item_id)] / smoothed_context_total, eps)
        ppmi_scores[(int(source_item_id), int(target_item_id))] = max(
            0.0,
            float(np.log(directed_pair_prob / (source_prob * target_prob))),
        )
        ppmi_scores[(int(target_item_id), int(source_item_id))] = max(
            0.0,
            float(np.log(directed_pair_prob / (reverse_source_prob * reverse_target_prob))),
        )
    return ppmi_scores, item_degree


def mine_cooccurrence_pairs(
    train_file: str,
    sample: int,
    seed: int,
    min_count: int = 3,
    compute_ppmi: bool = False,
    ppmi_alpha: float = 0.75,
    pair_mining_mode: str = "global",
    window_size: Optional[int] = None,
) -> Dict[str, object]:
    canonical_mode = normalize_pair_mining_mode(pair_mining_mode)
    effective_min_count = max(1, int(min_count))
    use_ppmi = bool(as_bool(compute_ppmi))
    effective_ppmi_alpha = float(ppmi_alpha)
    effective_window_size = None if window_size is None else int(window_size)
    if canonical_mode == "sliding_window" and (effective_window_size is None or effective_window_size < 2):
        raise ValueError(f"window_size must be >= 2 when pair_mining_mode=sliding_window, got {window_size}")
    data = pd.read_csv(train_file)
    pair_counts: Dict[Tuple[int, int], int] = {}
    item_popularity: Dict[int, int] = {}

    sequences = _collect_user_level_sequences(data)
    sequences = _sample_sequences(sequences, sample=sample, seed=seed)
    sequence_count = len(sequences)
    for full_sequence in sequences:
        if canonical_mode == "global":
            unique_item_ids = sorted({int(item_id) for item_id in full_sequence if int(item_id) > 0})
            if len(unique_item_ids) < 2:
                continue
            for item_id in unique_item_ids:
                item_popularity[item_id] = item_popularity.get(item_id, 0) + 1
            _update_pair_counts_from_unique_items(unique_item_ids, pair_counts)
            continue

        for window_item_ids in _iter_sliding_windows(full_sequence, effective_window_size):
            unique_item_ids = sorted({int(item_id) for item_id in window_item_ids if int(item_id) > 0})
            if len(unique_item_ids) < 2:
                continue
            for item_id in unique_item_ids:
                item_popularity[item_id] = item_popularity.get(item_id, 0) + 1
            _update_pair_counts_from_unique_items(unique_item_ids, pair_counts)

    pre_filter_unique_pair_count = len(pair_counts)
    pre_filter_raw_pair_count = int(sum(pair_counts.values()))
    if effective_min_count > 1:
        pair_counts = {
            pair_key: count
            for pair_key, count in pair_counts.items()
            if int(count) >= effective_min_count
        }

    pair_ppmi_scores: Dict[Tuple[int, int], float] = {}
    pair_graph_marginals: Dict[int, float] = {}
    if use_ppmi:
        pair_ppmi_scores, pair_graph_marginals = _compute_cooccurrence_ppmi_scores(
            pair_counts=pair_counts,
            smoothing_alpha=effective_ppmi_alpha,
        )

    return {
        "pair_counts": pair_counts,
        "item_popularity": item_popularity,
        "pair_mining_mode": canonical_mode,
        "window_size": effective_window_size if canonical_mode == "sliding_window" else None,
        "sequence_count": int(sequence_count),
        "sequence_unit": "user",
        "pre_filter_unique_pair_count": int(pre_filter_unique_pair_count),
        "pre_filter_raw_pair_count": int(pre_filter_raw_pair_count),
        "min_count": int(effective_min_count),
        "contains_ppmi": bool(use_ppmi),
        "ppmi_alpha": float(effective_ppmi_alpha),
        "pair_ppmi_scores": pair_ppmi_scores,
        "pair_graph_marginals": {
            int(item_id): float(value)
            for item_id, value in pair_graph_marginals.items()
        },
    }


def load_hidden_cache(cache_path: str) -> Dict[str, object]:
    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid hidden cache payload type: {type(payload)}")
    items = payload.get("items")
    if not isinstance(items, dict) or not items:
        raise ValueError(f"Hidden cache has no items: {cache_path}")
    return payload


def iter_sorted_item_ids(item_id_to_title: Dict[int, str]) -> Iterable[int]:
    return sorted(int(item_id) for item_id in item_id_to_title.keys() if int(item_id) > 0)
