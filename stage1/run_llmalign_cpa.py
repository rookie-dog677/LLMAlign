import ast
import json
import math
import os
import random
import re
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set

import fire
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import transformers
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from safetensors import safe_open
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

if __package__ in {None, ""}:
    _THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    _REPO_ROOT = os.path.dirname(_THIS_DIR)
    if _REPO_ROOT not in sys.path:
        sys.path.insert(0, _REPO_ROOT)

from stage1.latent_attention import (
    LATENT_HEAD_WEIGHTS_NAME,
    LatentAttentionPooling,
    load_latent_attention_head,
    save_latent_attention_head,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def build_bidirectional_attention_mask_dict(
    attention_mask: torch.Tensor, dtype: torch.dtype
) -> torch.Tensor:
    """Build additive 4D padding mask tensor for full (non-causal) self-attention."""
    if attention_mask.ndim != 2:
        raise ValueError(
            f"Expected 2D attention_mask [B, L], got shape={tuple(attention_mask.shape)}"
        )
    if not torch.is_floating_point(torch.empty((), dtype=dtype)):
        dtype = torch.float32

    # 1 -> keep token, 0 -> pad token. Build an already-inverted 4D mask where
    # valid positions are 0 and padding positions are min(dtype), matching HF
    # Qwen2 expectation for custom 4D attention masks.
    pad_mask = attention_mask.to(dtype=dtype)
    min_value = torch.finfo(dtype).min
    additive_mask = (1.0 - pad_mask).unsqueeze(1).unsqueeze(2) * min_value
    seq_len = attention_mask.size(1)
    return additive_mask.expand(-1, 1, seq_len, -1).contiguous()


def parse_title_list(raw_value) -> List[str]:
    if isinstance(raw_value, list):
        return [str(x) for x in raw_value if str(x).strip()]

    if raw_value is None:
        return []

    raw_text = str(raw_value).strip()
    if not raw_text:
        return []

    if raw_text.startswith("[") and raw_text.endswith("]"):
        try:
            values = ast.literal_eval(raw_text)
            if isinstance(values, list):
                return [str(x) for x in values if str(x).strip()]
        except (ValueError, SyntaxError):
            pass

    return [raw_text]


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


LEXICAL_STOPWORDS = {
    "a",
    "an",
    "and",
    "for",
    "from",
    "in",
    "of",
    "on",
    "or",
    "the",
    "to",
    "with",
}


def extract_title_tokens(text: str, min_token_len: int = 2) -> List[str]:
    tokens = re.findall(r"[A-Za-z0-9]+", text.lower())
    return [
        tok
        for tok in tokens
        if len(tok) >= min_token_len and tok not in LEXICAL_STOPWORDS
    ]


class ContrastiveCSFTDataset(Dataset):
    def __init__(
        self,
        csv_file: str,
        sample: int = -1,
        seed: int = 42,
        history_sep: str = ", ",
        history_placeholder: str = "No history.",
        user_prefix: str = "",
        user_suffix: str = "",
        item_prefix: str = "",
        item_suffix: str = "",
    ) -> None:
        self.csv_file = csv_file
        self.history_sep = history_sep
        self.history_placeholder = history_placeholder
        self.user_prefix = user_prefix
        self.user_suffix = user_suffix
        self.item_prefix = item_prefix
        self.item_suffix = item_suffix
        self.seed = seed

        data = pd.read_csv(csv_file)
        if sample > 0 and sample < len(data):
            data = data.sample(n=sample, random_state=seed)
        data = data.reset_index(drop=True)

        self.samples = []
        item_titles = set()
        item_text_freq: Dict[str, int] = {}
        item_text_tokens: Dict[str, List[str]] = {}
        token_to_item_texts: Dict[str, set] = {}
        token_item_df: Dict[str, int] = {}
        item_id_to_title: Dict[int, str] = {}
        item_id_to_item_text: Dict[int, str] = {}

        for row in data.itertuples(index=False):
            history_titles = parse_title_list(getattr(row, "history_item_title", None))
            target_title = str(getattr(row, "item_title", "")).strip()
            raw_item_id = getattr(row, "item_id", -1)

            if not target_title:
                continue

            try:
                item_id = int(raw_item_id)
            except (TypeError, ValueError):
                item_id = -1

            history_text = self.history_sep.join(history_titles) if history_titles else self.history_placeholder
            user_text = f"{self.user_prefix}{history_text}{self.user_suffix}"
            item_text = f"{self.item_prefix}{target_title}{self.item_suffix}"
            history_tokens = extract_title_tokens(" ".join(history_titles))

            self.samples.append(
                {
                    "user_text": user_text,
                    "item_text": item_text,
                    "item_title": target_title,
                    "item_id": item_id,
                    "history_tokens": history_tokens,
                }
            )
            item_titles.add(item_text)
            item_text_freq[item_text] = item_text_freq.get(item_text, 0) + 1
            if item_id > 0 and item_id not in item_id_to_title:
                item_id_to_title[item_id] = target_title
            if item_id > 0 and item_id not in item_id_to_item_text:
                item_id_to_item_text[item_id] = item_text
            if item_text not in item_text_tokens:
                tokens = extract_title_tokens(target_title)
                item_text_tokens[item_text] = tokens
                for token in set(tokens):
                    token_item_df[token] = token_item_df.get(token, 0) + 1
                    if token not in token_to_item_texts:
                        token_to_item_texts[token] = set()
                    token_to_item_texts[token].add(item_text)

        self.item_title_pool = sorted(item_titles)
        self.item_text_freq = item_text_freq
        self.item_text_tokens = item_text_tokens
        self.token_item_df = token_item_df
        self.token_to_item_texts = {
            token: sorted(list(items))
            for token, items in token_to_item_texts.items()
        }
        self.item_id_to_title = item_id_to_title
        self.item_id_to_item_text = item_id_to_item_text

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, str]:
        return self.samples[idx]


@dataclass
class ContrastiveCollator:
    tokenizer: AutoTokenizer
    max_user_len: int = 256
    max_item_len: int = 128
    sampled_negatives: int = 8
    item_title_pool: Optional[List[str]] = None
    item_text_freq: Optional[Dict[str, int]] = None
    item_text_tokens: Optional[Dict[str, List[str]]] = None
    token_to_item_texts: Optional[Dict[str, List[str]]] = None
    token_item_df: Optional[Dict[str, int]] = None
    seed: int = 42
    positive_group_by: str = "item_id"
    negative_sampling_strategy: str = "hard_overlap"
    negative_hard_topk: int = 32
    negative_hard_history_weight: float = 0.35
    negative_hard_max_token_df_ratio: float = 0.2
    negative_hard_candidate_cap: int = 1024
    negative_hard_min_shared_tokens: int = 1
    negative_hard_log_stats: bool = False
    negative_hard_log_every: int = 1
    negative_avoid_batch_positives: bool = True
    negative_dedup_per_query: bool = False
    negative_max_trials: int = 48

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)
        self.negative_sampling_strategy = self.negative_sampling_strategy.lower().strip()
        valid_neg_strategies = {"hard_overlap"}
        if self.negative_sampling_strategy not in valid_neg_strategies:
            raise ValueError(f"Unsupported negative_sampling_strategy: {self.negative_sampling_strategy}")
        if self.negative_hard_topk <= 0:
            raise ValueError("negative_hard_topk must be positive.")
        if not (0.0 <= self.negative_hard_history_weight <= 1.0):
            raise ValueError("negative_hard_history_weight must be in [0, 1].")
        if not (0.0 < self.negative_hard_max_token_df_ratio <= 1.0):
            raise ValueError("negative_hard_max_token_df_ratio must be in (0, 1].")
        if self.negative_hard_candidate_cap <= 0:
            raise ValueError("negative_hard_candidate_cap must be positive.")
        if self.negative_hard_min_shared_tokens <= 0:
            raise ValueError("negative_hard_min_shared_tokens must be positive.")
        if self.negative_hard_log_every <= 0:
            raise ValueError("negative_hard_log_every must be positive.")
        if self.negative_max_trials <= 0:
            raise ValueError("negative_max_trials must be positive.")

        if self.item_title_pool:
            self._pool_list = list(self.item_title_pool)
        else:
            self._pool_list = []

        item_count = max(1, len(self._pool_list))
        self._hard_max_token_df = max(
            1,
            int(round(item_count * self.negative_hard_max_token_df_ratio)),
        )
        self._token_idf = {}
        if self.token_item_df:
            for token, df in self.token_item_df.items():
                clipped_df = max(1, int(df))
                self._token_idf[token] = math.log((1.0 + item_count) / (1.0 + clipped_df)) + 1.0
        self._hard_log_batch_idx = 0
        self._random_fallback_warned = False

    def _sample_random_fallback(
        self,
        pos_item_text: str,
        forbidden_texts: Optional[Set[str]] = None,
        used_texts: Optional[Set[str]] = None,
        max_trials: int = 50,
    ) -> Optional[str]:
        """Fallback to uniform random sampling when hard_overlap yields no candidates."""
        if not self._pool_list:
            return None
        for _ in range(max_trials):
            candidate = self._pool_list[self._rng.randrange(len(self._pool_list))]
            if candidate == pos_item_text:
                continue
            if forbidden_texts is not None and candidate in forbidden_texts:
                continue
            if used_texts is not None and candidate in used_texts:
                continue
            return candidate
        return None

    def _collect_overlap_candidates(
        self,
        pos_item_text: str,
        history_tokens: Optional[List[str]] = None,
    ) -> Set[str]:
        if not self.item_text_tokens or not self.token_to_item_texts:
            return set()

        pos_tokens = self.item_text_tokens.get(pos_item_text, [])
        if not pos_tokens:
            return set()

        history_token_set = {
            token
            for token in (history_tokens or [])
            if token in self.token_to_item_texts
        }
        pos_token_set = set(pos_tokens)
        query_token_set = {
            token
            for token in pos_token_set.union(history_token_set)
            if token in self.token_to_item_texts
        }
        # Shuffle traversal order to keep candidate selection stochastic before
        # candidate_cap truncation.
        query_tokens = list(query_token_set)
        self._rng.shuffle(query_tokens)

        candidate_pool: Set[str] = set()
        for token in query_tokens:
            df = self.token_item_df.get(token, 0) if self.token_item_df else 0
            if df > self._hard_max_token_df:
                continue
            token_candidates = self.token_to_item_texts.get(token, [])
            for candidate in token_candidates:
                if candidate != pos_item_text:
                    candidate_pool.add(candidate)
                if len(candidate_pool) >= self.negative_hard_candidate_cap:
                    break
            if len(candidate_pool) >= self.negative_hard_candidate_cap:
                break
        return candidate_pool

    def _sample_hard_overlap_item(
        self,
        pos_item_text: str,
        history_tokens: Optional[List[str]] = None,
    ) -> Optional[str]:
        ranked_candidates = self._rank_hard_overlap_candidates(
            pos_item_text=pos_item_text,
            history_tokens=history_tokens,
        )
        if not ranked_candidates:
            return None
        topk = max(1, min(self.negative_hard_topk, len(ranked_candidates)))
        return ranked_candidates[self._rng.randrange(topk)]

    def _rank_hard_overlap_candidates(
        self,
        pos_item_text: str,
        history_tokens: Optional[List[str]] = None,
    ) -> List[str]:
        if not self.item_text_tokens or not self.token_to_item_texts:
            return []

        pos_tokens = self.item_text_tokens.get(pos_item_text, [])
        if not pos_tokens:
            return []

        history_token_set = set(history_tokens or [])
        pos_token_set = set(pos_tokens)
        candidate_pool = self._collect_overlap_candidates(
            pos_item_text=pos_item_text,
            history_tokens=history_tokens,
        )
        if not candidate_pool:
            return []

        scored = []
        for candidate in candidate_pool:
            candidate_tokens = self.item_text_tokens.get(candidate, [])
            if not candidate_tokens:
                continue
            candidate_token_set = set(candidate_tokens)
            shared_pos = candidate_token_set.intersection(pos_token_set)
            shared_hist = candidate_token_set.intersection(history_token_set)
            if (len(shared_pos) + len(shared_hist)) < self.negative_hard_min_shared_tokens:
                continue

            pos_score = sum(self._token_idf.get(token, 1.0) for token in shared_pos)
            hist_score = sum(self._token_idf.get(token, 1.0) for token in shared_hist)
            pop_bonus = 0.0
            if self.item_text_freq:
                pop_bonus = 0.05 * math.log1p(float(self.item_text_freq.get(candidate, 1)))
            score = pos_score + (self.negative_hard_history_weight * hist_score) + pop_bonus
            scored.append((score, candidate))

        if not scored:
            return []

        scored.sort(key=lambda x: x[0], reverse=True)
        return [candidate for _, candidate in scored]

    def _sample_from_ranked_candidates(
        self,
        ranked_candidates: List[str],
        sample_topk: int,
        forbidden_texts: Optional[Set[str]] = None,
        used_texts: Optional[Set[str]] = None,
    ) -> Optional[str]:
        if not ranked_candidates:
            return None

        topk = max(1, min(sample_topk, len(ranked_candidates)))
        draw_pool = ranked_candidates[:topk]
        for _ in range(min(self.negative_max_trials, topk * 2)):
            candidate = draw_pool[self._rng.randrange(topk)]
            if forbidden_texts is not None and candidate in forbidden_texts:
                continue
            if used_texts is not None and candidate in used_texts:
                continue
            return candidate

        for candidate in ranked_candidates:
            if forbidden_texts is not None and candidate in forbidden_texts:
                continue
            if used_texts is not None and candidate in used_texts:
                continue
            return candidate
        return None

    def _sample_negative(
        self,
        pos_item_text: str,
        history_tokens: Optional[List[str]] = None,
        ranked_candidates: Optional[List[str]] = None,
        ranked_topk: int = 0,
        forbidden_texts: Optional[Set[str]] = None,
        used_texts: Optional[Set[str]] = None,
        max_trials: Optional[int] = None,
        sampling_stats: Optional[Dict[str, int]] = None,
    ) -> Optional[str]:
        trials = self.negative_max_trials if max_trials is None else max_trials
        for _ in range(trials):
            if ranked_candidates is not None:
                candidate = self._sample_from_ranked_candidates(
                    ranked_candidates=ranked_candidates,
                    sample_topk=ranked_topk or self.negative_hard_topk,
                    forbidden_texts=forbidden_texts,
                    used_texts=used_texts,
                )
            else:
                candidate = self._sample_hard_overlap_item(
                    pos_item_text,
                    history_tokens=history_tokens,
                )

            if candidate is None:
                continue
            if candidate == pos_item_text:
                continue
            if forbidden_texts is not None and candidate in forbidden_texts:
                continue
            if used_texts is not None and candidate in used_texts:
                continue
            if sampling_stats is not None:
                sampling_stats["success"] = sampling_stats.get("success", 0) + 1
            return candidate
        if sampling_stats is not None:
            sampling_stats["failed"] = sampling_stats.get("failed", 0) + 1
        return None

    def _tokenize(self, texts: List[str], max_len: int) -> Dict[str, torch.Tensor]:
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
            return_attention_mask=True,
        )

    def __call__(self, batch: List[Dict[str, str]]) -> Dict[str, torch.Tensor]:
        user_texts = [d["user_text"] for d in batch]
        item_texts = [d["item_text"] for d in batch]
        active_sampled_negatives = self.sampled_negatives

        user_tokens = self._tokenize(user_texts, self.max_user_len)
        item_tokens = self._tokenize(item_texts, self.max_item_len)

        inputs = {
            "user_input_ids": user_tokens["input_ids"],
            "user_attention_mask": user_tokens["attention_mask"],
            "item_input_ids": item_tokens["input_ids"],
            "item_attention_mask": item_tokens["attention_mask"],
        }

        group_mode = self.positive_group_by.lower().strip()
        if group_mode not in {"none", "item_title", "item_text", "item_id"}:
            raise ValueError(f"Unsupported positive_group_by: {self.positive_group_by}")
        if group_mode != "none":
            if group_mode == "item_title":
                group_keys = [str(d.get("item_title", d["item_text"])) for d in batch]
            elif group_mode == "item_text":
                group_keys = [str(d["item_text"]) for d in batch]
            else:
                group_keys = [str(d.get("item_id", -1)) for d in batch]

            group_to_id = {}
            group_ids = []
            for key in group_keys:
                if key not in group_to_id:
                    group_to_id[key] = len(group_to_id)
                group_ids.append(group_to_id[key])

            inputs["pos_group_ids"] = torch.tensor(group_ids, dtype=torch.long)

        if active_sampled_negatives > 0 and self.item_title_pool:
            negative_texts = []
            forbidden_texts = set(item_texts) if self.negative_avoid_batch_positives else None
            sampling_stats: Dict[str, int] = {"success": 0, "failed": 0}
            candidate_pool_sizes: List[int] = []
            for idx, pos_title in enumerate(item_texts):
                count = 0
                used_texts = set()
                sample_history_tokens = batch[idx].get("history_tokens", [])
                ranked_candidates = self._rank_hard_overlap_candidates(
                    pos_item_text=pos_title,
                    history_tokens=sample_history_tokens,
                )
                candidate_pool_sizes.append(len(ranked_candidates))
                ranked_topk = self.negative_hard_topk
                while count < active_sampled_negatives:
                    candidate = self._sample_negative(
                        pos_title,
                        history_tokens=sample_history_tokens,
                        ranked_candidates=ranked_candidates,
                        ranked_topk=ranked_topk,
                        forbidden_texts=forbidden_texts,
                        used_texts=used_texts if self.negative_dedup_per_query else None,
                        sampling_stats=sampling_stats,
                    )
                    if candidate is None and self.negative_dedup_per_query:
                        # Relax dedup first when candidate space is tight.
                        candidate = self._sample_negative(
                            pos_title,
                            history_tokens=sample_history_tokens,
                            ranked_candidates=ranked_candidates,
                            ranked_topk=ranked_topk,
                            forbidden_texts=forbidden_texts,
                            used_texts=None,
                            max_trials=self.negative_max_trials * 2,
                            sampling_stats=sampling_stats,
                        )
                    if candidate is None:
                        # Fallback: random negative from item pool
                        candidate = self._sample_random_fallback(
                            pos_item_text=pos_title,
                            forbidden_texts=forbidden_texts,
                            used_texts=used_texts,
                        )
                        if candidate is not None:
                            sampling_stats["random_fallback"] = sampling_stats.get("random_fallback", 0) + 1
                            if not self._random_fallback_warned:
                                self._random_fallback_warned = True
                                print(
                                    "[hard_overlap][collator] WARNING: hard_overlap candidates exhausted "
                                    "for some samples, falling back to random negatives. "
                                    "This is expected for datasets with short/generic item titles."
                                )
                    if candidate is None:
                        raise RuntimeError(
                            "Failed to sample enough negatives even with random fallback; "
                            "item pool may be too small."
                        )
                    negative_texts.append(candidate)
                    used_texts.add(candidate)
                    count += 1

            neg_tokens = self._tokenize(negative_texts, self.max_item_len)
            inputs["neg_input_ids"] = neg_tokens["input_ids"]
            inputs["neg_attention_mask"] = neg_tokens["attention_mask"]
            if self.negative_hard_log_stats:
                self._hard_log_batch_idx += 1
                if self._hard_log_batch_idx % self.negative_hard_log_every == 0:
                    total_success = sampling_stats.get("success", 0)
                    avg_pool = (
                        float(sum(candidate_pool_sizes)) / float(len(candidate_pool_sizes))
                        if candidate_pool_sizes
                        else 0.0
                    )
                    min_pool = min(candidate_pool_sizes) if candidate_pool_sizes else 0
                    max_pool = max(candidate_pool_sizes) if candidate_pool_sizes else 0
                    zero_pool_ratio = (
                        float(sum(1 for size in candidate_pool_sizes if size == 0))
                        / float(len(candidate_pool_sizes))
                        if candidate_pool_sizes
                        else 0.0
                    )
                    print(
                        "[hard_overlap][collator] "
                        f"batch_idx={self._hard_log_batch_idx} "
                        f"batch_size={len(batch)} "
                        f"sampled_negatives={active_sampled_negatives} "
                        f"avg_pool={avg_pool:.2f} min_pool={min_pool} max_pool={max_pool} "
                        f"zero_pool_ratio={zero_pool_ratio:.4f} "
                        f"failed_draws={sampling_stats.get('failed', 0)} "
                        f"random_fallback={sampling_stats.get('random_fallback', 0)}"
                    )

        return inputs



class ContrastiveTrainer(Trainer):
    def __init__(
        self,
        *args,
        pooling_mode: str = "",
        user_pooling_mode: str = "last_token",
        item_pooling_mode: str = "last_token",
        contrastive_loss_mode: str = "multi_positive_nce",
        temperature: float = 0.07,
        learnable_temperature: bool = True,
        logit_scale_max: float = 100.0,
        normalize_embeddings: bool = True,
        attention_mask_type: str = "bidirectional",
        user_attention_mask_type: str = "",
        item_attention_mask_type: str = "",
        embedding_head_type: str = "latent_attention",
        latent_num_latents: int = 128,
        latent_num_cross_heads: int = 8,
        latent_cross_dim_head: int = 64,
        latent_ff_mult: int = 4,
        latent_dim: int = -1,
        eval_data_collator: Optional[Any] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if pooling_mode:
            user_pooling_mode = pooling_mode
            item_pooling_mode = pooling_mode
        self.user_pooling_mode = user_pooling_mode
        self.item_pooling_mode = item_pooling_mode
        self.contrastive_loss_mode = contrastive_loss_mode
        self.temperature = temperature
        self.learnable_temperature = bool(learnable_temperature)
        self.logit_scale_max = float(logit_scale_max)
        self.normalize_embeddings = normalize_embeddings
        self.attention_mask_type = attention_mask_type.lower().strip()
        self.user_attention_mask_type = (
            user_attention_mask_type.lower().strip()
            if user_attention_mask_type
            else self.attention_mask_type
        )
        self.item_attention_mask_type = (
            item_attention_mask_type.lower().strip()
            if item_attention_mask_type
            else self.attention_mask_type
        )
        if self.contrastive_loss_mode not in {"one_hot_ce", "multi_positive_nce"}:
            raise ValueError(f"Unsupported contrastive_loss_mode: {self.contrastive_loss_mode}")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive.")
        if self.logit_scale_max <= 0:
            raise ValueError("logit_scale_max must be positive.")
        valid_mask_types = {"causal", "bidirectional"}
        if self.attention_mask_type not in valid_mask_types:
            raise ValueError(f"Unsupported attention_mask_type: {self.attention_mask_type}")
        if self.user_attention_mask_type not in valid_mask_types:
            raise ValueError(f"Unsupported user_attention_mask_type: {self.user_attention_mask_type}")
        if self.item_attention_mask_type not in valid_mask_types:
            raise ValueError(f"Unsupported item_attention_mask_type: {self.item_attention_mask_type}")
        self.embedding_head_type = embedding_head_type.lower().strip()
        valid_head_types = {"token_pool", "latent_attention"}
        if self.embedding_head_type not in valid_head_types:
            raise ValueError(f"Unsupported embedding_head_type: {self.embedding_head_type}")
        self.latent_num_latents = latent_num_latents
        self.latent_num_cross_heads = latent_num_cross_heads
        self.latent_cross_dim_head = latent_cross_dim_head
        self.latent_ff_mult = latent_ff_mult
        self.latent_dim = latent_dim
        self.eval_data_collator = eval_data_collator
        if self.learnable_temperature:
            self._ensure_logit_scale_param(self.model)
        if self.embedding_head_type == "latent_attention":
            self._ensure_latent_head(self.model)

    def get_eval_dataloader(self, eval_dataset=None):
        if self.eval_data_collator is None:
            return super().get_eval_dataloader(eval_dataset)
        original_collator = self.data_collator
        self.data_collator = self.eval_data_collator
        try:
            return super().get_eval_dataloader(eval_dataset)
        finally:
            self.data_collator = original_collator

    def _infer_hidden_size(self, model: torch.nn.Module) -> int:
        backbone = self._get_backbone(model)
        hidden_size = getattr(backbone.config, "hidden_size", None)
        if hidden_size is None:
            hidden_size = getattr(getattr(model, "config", None), "hidden_size", None)
        if hidden_size is None:
            raise ValueError("Cannot infer hidden_size for latent attention head.")
        return int(hidden_size)

    def _latent_head_attr(self) -> str:
        return "_llmalign_latent_attention_head"

    @staticmethod
    def _initial_logit_scale(temperature: float) -> float:
        return math.log(1.0 / float(temperature))

    def _logit_scale_attr(self) -> str:
        return "_llmalign_logit_scale"

    def _legacy_logit_scale_attrs(self) -> List[str]:
        return []

    def _resolve_model_holder(self, model: torch.nn.Module) -> torch.nn.Module:
        candidate_attrs = [self._logit_scale_attr(), *self._legacy_logit_scale_attrs()]
        for attr_name in candidate_attrs:
            if hasattr(model, attr_name):
                return model
        module = getattr(model, "module", None)
        if module is not None:
            for attr_name in candidate_attrs:
                if hasattr(module, attr_name):
                    return module
        return model

    def _ensure_logit_scale_param(self, model: torch.nn.Module) -> torch.nn.Parameter:
        holder = self._resolve_model_holder(model)
        current_attr = self._logit_scale_attr()
        existing = getattr(holder, current_attr, None)
        if isinstance(existing, torch.nn.Parameter):
            return existing
        for legacy_attr in self._legacy_logit_scale_attrs():
            legacy_param = getattr(holder, legacy_attr, None)
            if isinstance(legacy_param, torch.nn.Parameter):
                delattr(holder, legacy_attr)
                setattr(holder, current_attr, legacy_param)
                return legacy_param
        param = torch.nn.Parameter(torch.tensor(self._initial_logit_scale(self.temperature), dtype=torch.float32))
        setattr(holder, current_attr, param)
        return param

    def _load_logit_scale_tensor_from_checkpoint(
        self,
        checkpoint_path: str,
    ) -> Optional[tuple[str, torch.Tensor]]:
        if not checkpoint_path or not os.path.isdir(checkpoint_path):
            return None

        candidate_keys = [self._logit_scale_attr(), *self._legacy_logit_scale_attrs()]
        safetensor_path = os.path.join(checkpoint_path, "model.safetensors")
        if os.path.exists(safetensor_path):
            with safe_open(safetensor_path, framework="pt", device="cpu") as handle:
                available_keys = set(handle.keys())
                for key in candidate_keys:
                    if key in available_keys:
                        return key, handle.get_tensor(key)

        torch_bin_path = os.path.join(checkpoint_path, "pytorch_model.bin")
        if os.path.exists(torch_bin_path):
            state_dict = torch.load(torch_bin_path, map_location="cpu")
            for key in candidate_keys:
                if key in state_dict:
                    return key, state_dict[key]

        return None

    def load_logit_scale_checkpoint(self, checkpoint_path: str) -> None:
        if not self.learnable_temperature:
            return
        loaded = self._load_logit_scale_tensor_from_checkpoint(checkpoint_path)
        if loaded is None:
            return
        source_key, tensor = loaded
        param = self._ensure_logit_scale_param(self.model)
        param.data.copy_(tensor.to(device=param.device, dtype=param.dtype))
        if self.args.should_save:
            print(f"Loaded logit scale from {checkpoint_path} ({source_key} -> {self._logit_scale_attr()})")

    def _get_logit_scale(self, model: torch.nn.Module) -> torch.Tensor:
        if self.learnable_temperature:
            logit_scale_param = self._ensure_logit_scale_param(model)
            return logit_scale_param.exp().clamp(max=self.logit_scale_max)
        device = next(self._get_backbone(model).parameters()).device
        return torch.tensor(1.0 / self.temperature, device=device, dtype=torch.float32)

    def _ensure_latent_head(
        self,
        model: torch.nn.Module,
    ) -> Optional[LatentAttentionPooling]:
        if self.embedding_head_type != "latent_attention":
            return None
        attr_name = self._latent_head_attr()
        existing = getattr(model, attr_name, None)
        if isinstance(existing, LatentAttentionPooling):
            return existing

        hidden_size = self._infer_hidden_size(model)
        head = LatentAttentionPooling(
            hidden_size=hidden_size,
            num_latents=self.latent_num_latents,
            latent_dim=self.latent_dim,
            num_cross_heads=self.latent_num_cross_heads,
            cross_dim_head=self.latent_cross_dim_head,
            ff_mult=self.latent_ff_mult,
        )
        backbone = self._get_backbone(model)
        model_param = next(backbone.parameters())
        head.to(device=model_param.device, dtype=model_param.dtype)
        setattr(model, attr_name, head)
        return head

    def _get_latent_head(self, model: torch.nn.Module) -> LatentAttentionPooling:
        head = self._ensure_latent_head(model)
        if head is None:
            raise ValueError("Latent attention head is not enabled.")
        return head

    def _latent_head_config_dict(self, model: torch.nn.Module) -> Dict[str, int]:
        head = self._get_latent_head(model)
        return {
            "embedding_head_type": self.embedding_head_type,
            **head.export_config(),
        }

    def load_latent_head_checkpoint(self, checkpoint_path: str) -> None:
        if self.embedding_head_type != "latent_attention":
            return
        saved_cfg, state_dict, file_path = load_latent_attention_head(checkpoint_path, map_location="cpu")
        if saved_cfg:
            self.latent_num_latents = int(saved_cfg.get("num_latents", self.latent_num_latents))
            self.latent_num_cross_heads = int(saved_cfg.get("num_cross_heads", self.latent_num_cross_heads))
            self.latent_cross_dim_head = int(saved_cfg.get("cross_dim_head", self.latent_cross_dim_head))
            self.latent_ff_mult = int(saved_cfg.get("ff_mult", self.latent_ff_mult))
            self.latent_dim = int(saved_cfg.get("latent_dim", self.latent_dim))
        attr_name = self._latent_head_attr()
        if hasattr(self.model, attr_name):
            delattr(self.model, attr_name)
        head = self._ensure_latent_head(self.model)
        missing, unexpected = head.load_state_dict(state_dict, strict=False)
        unexpected = [key for key in unexpected if not key.startswith("readout_gate.")]
        if missing or unexpected:
            raise ValueError(
                f"Failed to load latent attention head from {file_path}, "
                f"missing={missing}, unexpected={unexpected}"
            )
        if self.args.should_save:
            print(f"Loaded latent attention head from {file_path}")

    def _prepare_model_attention_mask(
        self, attention_mask: torch.Tensor, model_dtype: torch.dtype, attention_mask_type: str
    ):
        if attention_mask_type == "causal":
            return attention_mask
        return build_bidirectional_attention_mask_dict(attention_mask, dtype=model_dtype)

    @staticmethod
    def _get_backbone(model: torch.nn.Module) -> torch.nn.Module:
        base_model = model.get_base_model() if hasattr(model, "get_base_model") else model
        if hasattr(base_model, "model"):
            return base_model.model
        if hasattr(base_model, "transformer"):
            return base_model.transformer
        raise ValueError("Cannot locate backbone encoder from model.")

    def _pool(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        pooling_mode: str,
    ) -> torch.Tensor:
        if pooling_mode == "mean":
            mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
            summed = (hidden_states * mask).sum(dim=1)
            denom = mask.sum(dim=1).clamp(min=1.0)
            return summed / denom

        if pooling_mode != "last_token":
            raise ValueError(f"Unsupported pooling_mode: {pooling_mode}")

        last_pos = attention_mask.to(dtype=torch.long).sum(dim=1) - 1
        last_pos = last_pos.clamp(min=0)
        batch_idx = torch.arange(hidden_states.size(0), device=hidden_states.device)
        return hidden_states[batch_idx, last_pos, :]

    def _encode_text(
        self,
        model: torch.nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pooling_mode: str,
        attention_mask_type: str,
    ) -> torch.Tensor:
        backbone = self._get_backbone(model)
        model_dtype = next(backbone.parameters()).dtype
        model_attention_mask = self._prepare_model_attention_mask(
            attention_mask,
            model_dtype,
            attention_mask_type=attention_mask_type,
        )
        outputs = backbone(
            input_ids=input_ids,
            attention_mask=model_attention_mask,
            use_cache=False,
            return_dict=True,
        )
        hidden = outputs.last_hidden_state
        if self.embedding_head_type == "latent_attention":
            head = self._get_latent_head(model)
            emb = head(hidden, attention_mask=attention_mask).float()
        else:
            emb = self._pool(hidden, attention_mask, pooling_mode=pooling_mode).float()
        if self.normalize_embeddings:
            emb = F.normalize(emb, dim=-1)
        return emb

    @staticmethod
    def _build_positive_mask(group_ids: torch.Tensor) -> torch.Tensor:
        return group_ids.unsqueeze(1).eq(group_ids.unsqueeze(0))

    @staticmethod
    def _multi_positive_nce_per_sample(logits: torch.Tensor, pos_mask: torch.Tensor) -> torch.Tensor:
        min_value = torch.finfo(logits.dtype).min
        masked_logits = logits.masked_fill(~pos_mask, min_value)
        log_pos = torch.logsumexp(masked_logits, dim=1)
        log_all = torch.logsumexp(logits, dim=1)
        return -(log_pos - log_all)

    def _encode_negative_items(
        self,
        model: torch.nn.Module,
        neg_input_ids: torch.Tensor,
        neg_attention_mask: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        del batch_size
        return self._encode_text(
            model,
            neg_input_ids,
            neg_attention_mask,
            pooling_mode=self.item_pooling_mode,
            attention_mask_type=self.item_attention_mask_type,
        )

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        user_emb = self._encode_text(
            model,
            inputs["user_input_ids"],
            inputs["user_attention_mask"],
            pooling_mode=self.user_pooling_mode,
            attention_mask_type=self.user_attention_mask_type,
        )
        pos_item_emb = self._encode_text(
            model,
            inputs["item_input_ids"],
            inputs["item_attention_mask"],
            pooling_mode=self.item_pooling_mode,
            attention_mask_type=self.item_attention_mask_type,
        )

        logit_scale = self._get_logit_scale(model)
        logits_pos = torch.matmul(user_emb, pos_item_emb.transpose(0, 1)) * logit_scale

        logits = logits_pos
        if "neg_input_ids" in inputs:
            neg_item_emb = self._encode_negative_items(
                model,
                inputs["neg_input_ids"],
                inputs["neg_attention_mask"],
                batch_size=user_emb.size(0),
            )
            logits_neg = torch.matmul(user_emb, neg_item_emb.transpose(0, 1)) * logit_scale
            logits = torch.cat([logits_pos, logits_neg], dim=1)

        if self.contrastive_loss_mode == "one_hot_ce":
            labels = torch.arange(logits_pos.size(0), device=logits_pos.device)
            loss = F.cross_entropy(logits, labels)
        else:
            group_ids = inputs.get("pos_group_ids")
            if group_ids is None:
                group_ids = torch.arange(logits_pos.size(0), device=logits_pos.device)
            else:
                group_ids = group_ids.to(logits_pos.device)
            pos_mask_base = self._build_positive_mask(group_ids)

            pos_mask = pos_mask_base
            if logits.size(1) > logits_pos.size(1):
                padding = torch.zeros(
                    logits_pos.size(0),
                    logits.size(1) - logits_pos.size(1),
                    dtype=torch.bool,
                    device=logits.device,
                )
                pos_mask = torch.cat([pos_mask, padding], dim=1)

            if self.contrastive_loss_mode == "multi_positive_nce":
                loss = self._multi_positive_nce_per_sample(logits, pos_mask).mean()
            else:
                raise ValueError(f"Unsupported contrastive_loss_mode: {self.contrastive_loss_mode}")

        if return_outputs:
            return loss, {"logits": logits}
        return loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        # Evaluate with the same contrastive objective used in training.
        # Do not call model(**inputs), because inputs are custom fields rather than LM forward args.
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)
        loss = loss.detach().mean()
        return (loss, None, None)

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        super().save_model(output_dir=output_dir, _internal_call=_internal_call)
        if self.embedding_head_type != "latent_attention" or not self.args.should_save:
            return
        save_dir = output_dir if output_dir is not None else self.args.output_dir
        head = self._get_latent_head(self.model)
        save_latent_attention_head(
            output_dir=save_dir,
            head_module=head,
            config=self._latent_head_config_dict(self.model),
        )

def maybe_apply_lora(
    model: torch.nn.Module,
    base_model_name: str,
    use_lora: bool,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
):
    if not use_lora:
        return model

    if "Qwen" in base_model_name or "Llama" in base_model_name:
        target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    else:
        raise ValueError("LoRA target modules are not configured for this base model.")

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def train(
    base_model: str = "",
    train_file: str = "",
    eval_file: str = "",
    output_dir: str = "./output/cpa_training",
    sample: int = -1,
    eval_sample: int = 2000,
    seed: int = 42,
    per_device_train_batch_size: int = 300,
    per_device_eval_batch_size: int = 300,
    gradient_accumulation_steps: int = 1,
    auto_find_batch_size: bool = True,
    num_epochs: int = 4,
    learning_rate: float = 5e-5,
    weight_decay: float = 0.01,
    warmup_steps: int = 200,
    max_user_len: int = 256,
    max_item_len: int = 128,
    pooling_mode: str = "",
    user_pooling_mode: str = "last_token",
    item_pooling_mode: str = "last_token",
    contrastive_loss_mode: str = "multi_positive_nce",
    positive_group_by: str = "item_id",
    normalize_embeddings: bool = True,
    temperature: float = 0.07,
    learnable_temperature: bool = True,
    logit_scale_max: float = 100.0,
    sampled_negatives: int = 8,
    negative_sampling_strategy: str = "hard_overlap",
    negative_hard_topk: int = 32,
    negative_hard_history_weight: float = 0.35,
    negative_hard_max_token_df_ratio: float = 0.2,
    negative_hard_candidate_cap: int = 1024,
    negative_hard_min_shared_tokens: int = 1,
    negative_hard_log_stats: bool = False,
    negative_hard_log_every: int = 1,
    negative_avoid_batch_positives: bool = True,
    negative_dedup_per_query: bool = False,
    negative_max_trials: int = 48,
    user_prefix: str = "",
    user_suffix: str = "",
    item_prefix: str = "",
    item_suffix: str = "",
    history_sep: str = ", ",
    history_placeholder: str = "No history.",
    logging_steps: int = 10,
    eval_steps: int = 50,
    save_steps: Optional[int] = None,
    save_total_limit: int = 3,
    max_steps: int = -1,
    early_stopping_patience: int = 5,
    early_stopping_threshold: float = 0.0,
    bf16: bool = True,
    fp16: bool = False,
    report_to: str = "none",
    resume_from_checkpoint: Optional[str] = None,
    use_lora: bool = False,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    quantize_4bit: bool = False,
    quantize_8bit: bool = False,
    attention_mask_type: str = "bidirectional",
    user_attention_mask_type: str = "",
    item_attention_mask_type: str = "",
    embedding_head_type: str = "latent_attention",
    latent_num_latents: int = 128,
    latent_num_cross_heads: int = 8,
    latent_cross_dim_head: int = 64,
    latent_ff_mult: int = 4,
    latent_dim: int = -1,
):
    set_seed(seed)

    if save_steps is None:
        save_steps = int(eval_steps)

    if not base_model:
        raise ValueError("base_model must be set to a local HuggingFace model/checkpoint directory.")
    if not os.path.exists(base_model):
        raise FileNotFoundError(f"base_model path not found: {base_model}")
    if not os.path.exists(train_file):
        raise FileNotFoundError(f"train_file not found: {train_file}")
    if not os.path.exists(eval_file):
        raise FileNotFoundError(f"eval_file not found: {eval_file}")

    if temperature <= 0:
        raise ValueError("temperature must be positive.")
    if logit_scale_max <= 0:
        raise ValueError("logit_scale_max must be positive.")
    attention_mask_type = attention_mask_type.lower().strip()
    if attention_mask_type not in {"causal", "bidirectional"}:
        raise ValueError(f"Unsupported attention_mask_type: {attention_mask_type}")
    if user_attention_mask_type:
        user_attention_mask_type = user_attention_mask_type.lower().strip()
    else:
        user_attention_mask_type = attention_mask_type
    if item_attention_mask_type:
        item_attention_mask_type = item_attention_mask_type.lower().strip()
    else:
        item_attention_mask_type = attention_mask_type
    if user_attention_mask_type not in {"causal", "bidirectional"}:
        raise ValueError(f"Unsupported user_attention_mask_type: {user_attention_mask_type}")
    if item_attention_mask_type not in {"causal", "bidirectional"}:
        raise ValueError(f"Unsupported item_attention_mask_type: {item_attention_mask_type}")
    embedding_head_type = embedding_head_type.lower().strip()
    if embedding_head_type not in {"token_pool", "latent_attention"}:
        raise ValueError(f"Unsupported embedding_head_type: {embedding_head_type}")
    if latent_num_latents <= 0:
        raise ValueError("latent_num_latents must be positive.")
    if latent_num_cross_heads <= 0:
        raise ValueError("latent_num_cross_heads must be positive.")
    if latent_cross_dim_head <= 0:
        raise ValueError("latent_cross_dim_head must be positive.")
    if latent_ff_mult <= 0:
        raise ValueError("latent_ff_mult must be positive.")
    positive_group_by = positive_group_by.lower().strip()
    contrastive_loss_mode = contrastive_loss_mode.lower().strip()
    if pooling_mode:
        user_pooling_mode = pooling_mode
        item_pooling_mode = pooling_mode
    valid_pooling_modes = {"last_token", "mean"}
    if user_pooling_mode not in valid_pooling_modes:
        raise ValueError(f"Unsupported user_pooling_mode: {user_pooling_mode}")
    if item_pooling_mode not in valid_pooling_modes:
        raise ValueError(f"Unsupported item_pooling_mode: {item_pooling_mode}")
    valid_group_modes = {"none", "item_title", "item_text", "item_id"}
    if positive_group_by not in valid_group_modes:
        raise ValueError(f"Unsupported positive_group_by: {positive_group_by}")
    valid_loss_modes = {"one_hot_ce", "multi_positive_nce"}
    if contrastive_loss_mode not in valid_loss_modes:
        raise ValueError(f"Unsupported contrastive_loss_mode: {contrastive_loss_mode}")
    negative_sampling_strategy = negative_sampling_strategy.lower().strip()
    valid_neg_strategies = {"hard_overlap"}
    if negative_sampling_strategy not in valid_neg_strategies:
        raise ValueError(f"Unsupported negative_sampling_strategy: {negative_sampling_strategy}")
    if negative_hard_topk <= 0:
        raise ValueError("negative_hard_topk must be positive.")
    if not (0.0 <= negative_hard_history_weight <= 1.0):
        raise ValueError("negative_hard_history_weight must be in [0, 1].")
    if not (0.0 < negative_hard_max_token_df_ratio <= 1.0):
        raise ValueError("negative_hard_max_token_df_ratio must be in (0, 1].")
    if negative_hard_candidate_cap <= 0:
        raise ValueError("negative_hard_candidate_cap must be positive.")
    if negative_hard_min_shared_tokens <= 0:
        raise ValueError("negative_hard_min_shared_tokens must be positive.")
    if negative_hard_log_every <= 0:
        raise ValueError("negative_hard_log_every must be positive.")
    if negative_max_trials <= 0:
        raise ValueError("negative_max_trials must be positive.")
    if early_stopping_patience < 0:
        raise ValueError("early_stopping_patience must be >= 0.")
    if early_stopping_threshold < 0:
        raise ValueError("early_stopping_threshold must be >= 0.")
    os.makedirs(output_dir, exist_ok=True)

    model_dtype = torch.bfloat16 if bf16 and torch.cuda.is_available() else None

    load_kwargs = dict(torch_dtype=model_dtype, trust_remote_code=True)

    if quantize_4bit or quantize_8bit:
        from transformers import BitsAndBytesConfig
        if quantize_4bit:
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        else:
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True,
            )
        load_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(base_model, **load_kwargs)

    if quantize_4bit or quantize_8bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        use_lora = True  # QLoRA requires LoRA

    model = maybe_apply_lora(
        model,
        base_model_name=base_model,
        use_lora=use_lora,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
    )
    model.config.use_cache = False

    try:
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    except Exception as e:
        print(f"[WARN] fast tokenizer load failed for {base_model}: {e}")
        print("[WARN] falling back to slow tokenizer (use_fast=False).")
        tokenizer = AutoTokenizer.from_pretrained(
            base_model,
            trust_remote_code=True,
            use_fast=False,
        )

    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer has no pad_token_id and no eos_token_id.")
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    def _resolve_default_anchor_suffix(current_suffix: str) -> str:
        if current_suffix:
            return current_suffix
        eos_token = getattr(tokenizer, "eos_token", None) or ""
        if eos_token in {"<|endoftext|>", "<|im_end|>"}:
            return eos_token
        return "<|endoftext|>"

    user_suffix = _resolve_default_anchor_suffix(user_suffix)
    item_suffix = _resolve_default_anchor_suffix(item_suffix)

    dataset_kwargs = dict(
        seed=seed,
        history_sep=history_sep,
        history_placeholder=history_placeholder,
        user_prefix=user_prefix,
        user_suffix=user_suffix,
        item_prefix=item_prefix,
        item_suffix=item_suffix,
    )
    train_dataset = ContrastiveCSFTDataset(
        csv_file=train_file,
        sample=sample,
        **dataset_kwargs,
    )
    eval_dataset = ContrastiveCSFTDataset(
        csv_file=eval_file,
        sample=eval_sample,
        **dataset_kwargs,
    )

    train_collator = ContrastiveCollator(
        tokenizer=tokenizer,
        max_user_len=max_user_len,
        max_item_len=max_item_len,
        sampled_negatives=sampled_negatives,
        item_title_pool=train_dataset.item_title_pool,
        item_text_freq=train_dataset.item_text_freq,
        item_text_tokens=train_dataset.item_text_tokens,
        token_to_item_texts=train_dataset.token_to_item_texts,
        token_item_df=train_dataset.token_item_df,
        seed=seed,
        positive_group_by=positive_group_by,
        negative_sampling_strategy=negative_sampling_strategy,
        negative_hard_topk=negative_hard_topk,
        negative_hard_history_weight=negative_hard_history_weight,
        negative_hard_max_token_df_ratio=negative_hard_max_token_df_ratio,
        negative_hard_candidate_cap=negative_hard_candidate_cap,
        negative_hard_min_shared_tokens=negative_hard_min_shared_tokens,
        negative_hard_log_stats=negative_hard_log_stats,
        negative_hard_log_every=negative_hard_log_every,
        negative_avoid_batch_positives=negative_avoid_batch_positives,
        negative_dedup_per_query=negative_dedup_per_query,
        negative_max_trials=negative_max_trials,
    )
    eval_collator = ContrastiveCollator(
        tokenizer=tokenizer,
        max_user_len=max_user_len,
        max_item_len=max_item_len,
        sampled_negatives=sampled_negatives,
        item_title_pool=train_dataset.item_title_pool,
        item_text_freq=train_dataset.item_text_freq,
        item_text_tokens=train_dataset.item_text_tokens,
        token_to_item_texts=train_dataset.token_to_item_texts,
        token_item_df=train_dataset.token_item_df,
        seed=seed,
        positive_group_by=positive_group_by,
        negative_sampling_strategy=negative_sampling_strategy,
        negative_hard_topk=negative_hard_topk,
        negative_hard_history_weight=negative_hard_history_weight,
        negative_hard_max_token_df_ratio=negative_hard_max_token_df_ratio,
        negative_hard_candidate_cap=negative_hard_candidate_cap,
        negative_hard_min_shared_tokens=negative_hard_min_shared_tokens,
        negative_hard_log_stats=negative_hard_log_stats,
        negative_hard_log_every=negative_hard_log_every,
        negative_avoid_batch_positives=negative_avoid_batch_positives,
        negative_dedup_per_query=negative_dedup_per_query,
        negative_max_trials=negative_max_trials,
    )

    if len(train_dataset) == 0:
        raise ValueError("Train dataset is empty after preprocessing.")
    if len(eval_dataset) == 0:
        raise ValueError("Eval dataset is empty after preprocessing.")

    ddp = int(os.environ.get("WORLD_SIZE", "1")) > 1
    report_to_cfg = [] if report_to.lower() in {"none", "false", ""} else [report_to]

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        auto_find_batch_size=auto_find_batch_size,
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        bf16=bf16,
        fp16=fp16,
        logging_steps=logging_steps,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        max_steps=max_steps,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        ddp_find_unused_parameters=False if ddp else None,
        report_to=report_to_cfg,
        remove_unused_columns=False,
        optim="adamw_torch",
        gradient_checkpointing=True,
    )

    callbacks = []
    if early_stopping_patience > 0:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=early_stopping_patience,
                early_stopping_threshold=early_stopping_threshold,
            )
        )

    trainer = ContrastiveTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=train_collator,
        eval_data_collator=eval_collator,
        pooling_mode=pooling_mode,
        user_pooling_mode=user_pooling_mode,
        item_pooling_mode=item_pooling_mode,
        contrastive_loss_mode=contrastive_loss_mode,
        temperature=temperature,
        learnable_temperature=learnable_temperature,
        logit_scale_max=logit_scale_max,
        normalize_embeddings=normalize_embeddings,
        attention_mask_type=attention_mask_type,
        user_attention_mask_type=user_attention_mask_type,
        item_attention_mask_type=item_attention_mask_type,
        embedding_head_type=embedding_head_type,
        latent_num_latents=latent_num_latents,
        latent_num_cross_heads=latent_num_cross_heads,
        latent_cross_dim_head=latent_cross_dim_head,
        latent_ff_mult=latent_ff_mult,
        latent_dim=latent_dim,
        callbacks=callbacks,
    )

    latent_head_checkpoint = ""
    if embedding_head_type == "latent_attention":
        if resume_from_checkpoint:
            latent_head_checkpoint = resume_from_checkpoint
        else:
            latent_head_file = os.path.join(base_model, LATENT_HEAD_WEIGHTS_NAME)
            if os.path.exists(latent_head_file):
                latent_head_checkpoint = base_model
        if latent_head_checkpoint:
            trainer.load_latent_head_checkpoint(latent_head_checkpoint)
    if learnable_temperature:
        logit_scale_checkpoint = resume_from_checkpoint or base_model
        trainer.load_logit_scale_checkpoint(logit_scale_checkpoint)

    trainer.evaluate()
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    metadata = {
        "encoder_mode": "cpa_encoder",
        "pooling_mode": pooling_mode,
        "user_pooling_mode": user_pooling_mode,
        "item_pooling_mode": item_pooling_mode,
        "contrastive_loss_mode": contrastive_loss_mode,
        "positive_group_by": positive_group_by,
        "normalize_embeddings": normalize_embeddings,
        "temperature": temperature,
        "learnable_temperature": learnable_temperature,
        "logit_scale_max": logit_scale_max,
        "sampled_negatives": sampled_negatives,
        "negative_sampling_strategy": negative_sampling_strategy,
        "negative_hard_topk": negative_hard_topk,
        "negative_hard_history_weight": negative_hard_history_weight,
        "negative_hard_max_token_df_ratio": negative_hard_max_token_df_ratio,
        "negative_hard_candidate_cap": negative_hard_candidate_cap,
        "negative_hard_min_shared_tokens": negative_hard_min_shared_tokens,
        "negative_hard_log_stats": negative_hard_log_stats,
        "negative_hard_log_every": negative_hard_log_every,
        "negative_avoid_batch_positives": negative_avoid_batch_positives,
        "negative_dedup_per_query": negative_dedup_per_query,
        "negative_max_trials": negative_max_trials,
        "early_stopping_patience": early_stopping_patience,
        "early_stopping_threshold": early_stopping_threshold,
        "attention_mask_type": attention_mask_type,
        "user_attention_mask_type": user_attention_mask_type,
        "item_attention_mask_type": item_attention_mask_type,
        "embedding_head_type": embedding_head_type,
        "latent_num_latents": latent_num_latents,
        "latent_num_cross_heads": latent_num_cross_heads,
        "latent_cross_dim_head": latent_cross_dim_head,
        "latent_ff_mult": latent_ff_mult,
        "latent_dim": latent_dim,
        "max_user_len": max_user_len,
        "max_item_len": max_item_len,
        "user_prefix": user_prefix,
        "user_suffix": user_suffix,
        "item_prefix": item_prefix,
        "item_suffix": item_suffix,
        "history_sep": history_sep,
        "history_placeholder": history_placeholder,
    }
    with open(os.path.join(output_dir, "cpa_encoder_config.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    fire.Fire(train)
