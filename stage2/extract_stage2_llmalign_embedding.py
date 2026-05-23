import argparse
import json
import os
import os.path as op
import sys
from typing import Dict, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

if __package__ in {None, ""}:
    _THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    _REPO_ROOT = os.path.dirname(_THIS_DIR)
    if _REPO_ROOT not in sys.path:
        sys.path.insert(0, _REPO_ROOT)

from stage2.stage2_llmalign_utils import (
    FrozenStage1BackboneEncoder,
    STAGE2_CONFIG_NAME,
    STAGE2_STATE_NAME,
    compose_item_text,
    iter_sorted_item_ids,
    load_hidden_cache,
    load_item_titles,
    load_latent_head_module,
    load_stage2_config,
    pad_hidden_states,
    resolve_stage1_item_settings,
    resolve_stage2_checkpoint_path,
)


class LLMAlignStage2Extractor:
    def __init__(
        self,
        base_model_path: str,
        stage2_path: str,
        hidden_cache_path: str,
        batch_size: int,
        dataset_name: str,
        attention_mask_type: str,
        max_length: int,
        item_prefix: str,
        item_suffix: str,
        fusion_mode: str,
        fusion_alpha: float,
        normalize_embeddings: bool,
        whitening_mode: str,
        whitening_alpha: float,
        whitening_variance_ratio: float,
        bf16: bool,
    ) -> None:
        self.base_model_path = os.path.abspath(base_model_path)
        self.stage2_config = load_stage2_config(stage2_path)
        checkpoint_dataset = str(self.stage2_config.get("dataset", "") or dataset_name).strip()
        self.stage2_checkpoint = resolve_stage2_checkpoint_path(stage2_path, dataset_name=checkpoint_dataset)
        self.stage2_state_path = os.path.join(self.stage2_checkpoint, STAGE2_STATE_NAME)
        self.stage2_state = (
            torch.load(self.stage2_state_path, map_location="cpu", weights_only=False)
            if os.path.isfile(self.stage2_state_path)
            else {}
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = int(batch_size)
        self.attention_mask_type = str(attention_mask_type).lower().strip()
        self.max_length = int(max_length)
        self.item_prefix = str(item_prefix)
        self.item_suffix = str(item_suffix)
        self.fusion_mode = str(fusion_mode).lower().strip()
        self.fusion_alpha = float(fusion_alpha)
        self.normalize_embeddings = bool(normalize_embeddings)
        self.whitening_mode = str(whitening_mode).lower().strip()
        self.whitening_alpha = float(whitening_alpha)
        self.whitening_variance_ratio = float(whitening_variance_ratio)
        self.whitening_eps = 1.0e-5
        self.whitening_metadata: Dict[str, object] = {}
        self.hidden_cache = load_hidden_cache(hidden_cache_path) if hidden_cache_path else None
        supported_fusion_modes = {
            "concat",
            "semantic_repeat",
            "semantic_only",
            "cooccurrence_only",
            "alpha_mix",
            "whitened",
        }
        if self.fusion_mode not in supported_fusion_modes:
            raise ValueError(f"Unsupported fusion_mode: {fusion_mode}")
        if self.fusion_mode == "whitened":
            if self.whitening_mode not in {"truncated", "relaxed"}:
                raise ValueError(f"Unsupported whitening_mode: {whitening_mode}")
            if self.whitening_alpha < 0:
                raise ValueError(f"whitening_alpha must be non-negative, got {self.whitening_alpha}")
            if not 0.0 < self.whitening_variance_ratio <= 1.0:
                raise ValueError(
                    f"whitening_variance_ratio must be in (0, 1], got {self.whitening_variance_ratio}"
                )

        self.encoder: Optional[FrozenStage1BackboneEncoder] = None
        self.hidden_size: Optional[int] = None
        if self.hidden_cache is not None:
            self.hidden_size = int(
                self.hidden_cache.get("hidden_size")
                or next(iter(self.hidden_cache["items"].values()))["hidden_states"].shape[-1]
            )
            self.model_dtype = torch.float32
        else:
            self.encoder = FrozenStage1BackboneEncoder(
                base_model_path=self.base_model_path,
                attention_mask_type=self.attention_mask_type,
                torch_dtype=torch.bfloat16 if (bf16 and torch.cuda.is_available()) else torch.float32,
            )
            self.hidden_size = int(self.encoder.hidden_size)
            self.model_dtype = self.encoder.model_dtype

        semantic_head_path = os.path.join(self.base_model_path, "latent_attention_head.pt")
        if not os.path.isfile(semantic_head_path):
            raise FileNotFoundError(f"Semantic head not found under base model: {semantic_head_path}")
        self.semantic_head, _, semantic_resolved_path = load_latent_head_module(
            semantic_head_path,
            hidden_size=int(self.hidden_size),
            device=self.device,
            dtype=self.model_dtype,
        )
        self.semantic_head.eval()
        self.semantic_head_path = semantic_resolved_path

        self.cooccurrence_head = None
        self.cooccurrence_head_path = ""
        if self.fusion_mode != "whitened":
            self.cooccurrence_head, _, cooccurrence_resolved_path = load_latent_head_module(
                self.stage2_checkpoint,
                hidden_size=int(self.hidden_size),
                device=self.device,
                dtype=self.model_dtype,
            )
            self.cooccurrence_head.eval()
            self.cooccurrence_head_path = cooccurrence_resolved_path


    def output_dim(self) -> int:
        if self.fusion_mode in {"concat", "semantic_repeat", "whitened"}:
            return int(self.hidden_size) * 2
        return int(self.hidden_size)

    def _compose_embeddings(self, semantic_emb: torch.Tensor, aux_emb: torch.Tensor) -> torch.Tensor:
        if self.fusion_mode in {"concat", "semantic_repeat", "whitened"}:
            if self.normalize_embeddings:
                semantic_emb = F.normalize(semantic_emb, dim=-1)
                aux_emb = F.normalize(aux_emb, dim=-1)
            if self.fusion_mode == "semantic_repeat":
                return torch.cat([semantic_emb, semantic_emb], dim=-1)
            return torch.cat([semantic_emb, aux_emb], dim=-1)
        if self.fusion_mode == "semantic_only":
            final_emb = semantic_emb
        elif self.fusion_mode == "cooccurrence_only":
            final_emb = aux_emb
        else:
            if self.normalize_embeddings:
                semantic_emb = F.normalize(semantic_emb, dim=-1)
                aux_emb = F.normalize(aux_emb, dim=-1)
            final_emb = (self.fusion_alpha * semantic_emb) + ((1.0 - self.fusion_alpha) * aux_emb)
        if self.normalize_embeddings:
            final_emb = F.normalize(final_emb, dim=-1)
        return final_emb

    def _encode_semantic_from_hidden_batch(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        with torch.no_grad():
            return self.semantic_head(hidden_states, attention_mask=attention_mask).float()

    def _encode_from_hidden_batch(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        item_ids: Sequence[int],
    ) -> torch.Tensor:
        semantic_emb = self._encode_semantic_from_hidden_batch(hidden_states, attention_mask)
        if self.fusion_mode == "whitened":
            return semantic_emb
        if self.cooccurrence_head is None:
            raise RuntimeError("cooccurrence_head is not loaded for the current fusion mode")
        with torch.no_grad():
            cooccurrence_emb = self.cooccurrence_head(hidden_states, attention_mask=attention_mask).float()
            final_emb = self._compose_embeddings(semantic_emb, cooccurrence_emb)
        return final_emb

    def _encode_from_cache(self, item_ids: Sequence[int]) -> np.ndarray:
        items = self.hidden_cache["items"]
        hidden_list = [items[int(item_id)]["hidden_states"] for item_id in item_ids]
        mask_list = [items[int(item_id)]["attention_mask"] for item_id in item_ids]
        hidden_states, attention_mask = pad_hidden_states(hidden_list, mask_list, self.model_dtype)
        hidden_states = hidden_states.to(self.device)
        attention_mask = attention_mask.to(self.device)
        return self._encode_from_hidden_batch(hidden_states, attention_mask, item_ids).detach().cpu().numpy()

    def _encode_live(self, item_ids: Sequence[int], titles: Sequence[str]) -> np.ndarray:
        texts = [compose_item_text(title, self.item_prefix, self.item_suffix) for title in titles]
        hidden_pairs = self.encoder.encode_hidden_states(
            texts=texts,
            batch_size=self.batch_size,
            max_length=self.max_length,
        )
        hidden_list = [pair[0] for pair in hidden_pairs]
        mask_list = [pair[1] for pair in hidden_pairs]
        hidden_states, attention_mask = pad_hidden_states(hidden_list, mask_list, self.model_dtype)
        hidden_states = hidden_states.to(self.device)
        attention_mask = attention_mask.to(self.device)
        return self._encode_from_hidden_batch(hidden_states, attention_mask, item_ids).detach().cpu().numpy()

    def _encode_semantic_from_cache(self, item_ids: Sequence[int]) -> np.ndarray:
        items = self.hidden_cache["items"]
        hidden_list = [items[int(item_id)]["hidden_states"] for item_id in item_ids]
        mask_list = [items[int(item_id)]["attention_mask"] for item_id in item_ids]
        hidden_states, attention_mask = pad_hidden_states(hidden_list, mask_list, self.model_dtype)
        hidden_states = hidden_states.to(self.device)
        attention_mask = attention_mask.to(self.device)
        return self._encode_semantic_from_hidden_batch(hidden_states, attention_mask).detach().cpu().numpy()

    def _encode_semantic_live(self, titles: Sequence[str]) -> np.ndarray:
        texts = [compose_item_text(title, self.item_prefix, self.item_suffix) for title in titles]
        hidden_pairs = self.encoder.encode_hidden_states(
            texts=texts,
            batch_size=self.batch_size,
            max_length=self.max_length,
        )
        hidden_list = [pair[0] for pair in hidden_pairs]
        mask_list = [pair[1] for pair in hidden_pairs]
        hidden_states, attention_mask = pad_hidden_states(hidden_list, mask_list, self.model_dtype)
        hidden_states = hidden_states.to(self.device)
        attention_mask = attention_mask.to(self.device)
        return self._encode_semantic_from_hidden_batch(hidden_states, attention_mask).detach().cpu().numpy()

    def _encode_semantic_items(self, item_id_to_title: Dict[int, str]) -> np.ndarray:
        max_item_id = max(item_id_to_title)
        output = np.zeros((max_item_id + 1, int(self.hidden_size)), dtype=np.float32)
        item_ids = list(iter_sorted_item_ids(item_id_to_title))
        for start in range(0, len(item_ids), self.batch_size):
            batch_item_ids = item_ids[start:start + self.batch_size]
            if self.hidden_cache is not None:
                batch_emb = self._encode_semantic_from_cache(batch_item_ids)
            else:
                batch_titles = [item_id_to_title[item_id] for item_id in batch_item_ids]
                batch_emb = self._encode_semantic_live(batch_titles)
            for row_idx, item_id in enumerate(batch_item_ids):
                output[int(item_id)] = batch_emb[row_idx]
        return output

    def _compute_whitened_view(
        self,
        semantic_output: np.ndarray,
        item_ids: Sequence[int],
    ) -> torch.Tensor:
        semantic_tensor = torch.from_numpy(semantic_output[[int(item_id) for item_id in item_ids]]).to(dtype=torch.float64)
        item_count = int(semantic_tensor.shape[0])
        if item_count <= 1:
            raise ValueError("Need at least 2 item embeddings to compute whitening transform")
        mean = semantic_tensor.mean(dim=0, keepdim=True)
        centered = semantic_tensor - mean
        covariance = centered.transpose(0, 1).matmul(centered) / float(item_count)
        u, singular_values, _ = torch.linalg.svd(covariance, full_matrices=False)

        if self.whitening_mode == "relaxed":
            scaling = torch.pow(torch.rsqrt(singular_values + self.whitening_eps), self.whitening_alpha)
            whitening_matrix = (u * scaling.unsqueeze(0)).matmul(u.transpose(0, 1))
            effective_rank = int(singular_values.numel())
            explained_variance = 1.0
        else:
            cumulative = torch.cumsum(singular_values, dim=0) / torch.clamp(singular_values.sum(), min=self.whitening_eps)
            threshold = torch.tensor(self.whitening_variance_ratio, dtype=cumulative.dtype)
            effective_rank = int(torch.searchsorted(cumulative, threshold, right=False).item()) + 1
            effective_rank = min(effective_rank, int(singular_values.numel()))
            u_k = u[:, :effective_rank]
            scaling = torch.rsqrt(singular_values[:effective_rank] + self.whitening_eps)
            whitening_matrix = (u_k * scaling.unsqueeze(0)).matmul(u_k.transpose(0, 1))
            explained_variance = float(cumulative[effective_rank - 1].item())

        whitened = centered.matmul(whitening_matrix).to(dtype=torch.float32)
        self.whitening_metadata = {
            "whitening_mode": self.whitening_mode,
            "whitening_alpha": float(self.whitening_alpha),
            "whitening_variance_ratio": float(self.whitening_variance_ratio),
            "whitening_eps": float(self.whitening_eps),
            "whitening_effective_rank": int(effective_rank),
            "whitening_item_count": int(item_count),
            "whitening_feature_dim": int(semantic_tensor.shape[1]),
            "whitening_mean_norm": float(mean.norm().item()),
            "whitening_explained_variance": float(explained_variance),
            "whitening_min_singular_value": float(singular_values.min().item()),
            "whitening_max_singular_value": float(singular_values.max().item()),
        }
        return whitened

    def _encode_whitened_items(self, item_id_to_title: Dict[int, str]) -> np.ndarray:
        semantic_output = self._encode_semantic_items(item_id_to_title)
        item_ids = list(iter_sorted_item_ids(item_id_to_title))
        semantic_tensor = torch.from_numpy(semantic_output[[int(item_id) for item_id in item_ids]]).to(dtype=torch.float32)
        whitened_tensor = self._compute_whitened_view(semantic_output, item_ids)
        final_tensor = self._compose_embeddings(semantic_tensor, whitened_tensor).cpu().numpy().astype(np.float32)
        max_item_id = max(item_id_to_title)
        output = np.zeros((max_item_id + 1, self.output_dim()), dtype=np.float32)
        for row_idx, item_id in enumerate(item_ids):
            output[int(item_id)] = final_tensor[row_idx]
        return output

    def encode_items(self, item_id_to_title: Dict[int, str]) -> np.ndarray:
        if self.fusion_mode == "whitened":
            return self._encode_whitened_items(item_id_to_title)
        max_item_id = max(item_id_to_title)
        output = np.zeros((max_item_id + 1, self.output_dim()), dtype=np.float32)
        item_ids = list(iter_sorted_item_ids(item_id_to_title))
        for start in range(0, len(item_ids), self.batch_size):
            batch_item_ids = item_ids[start:start + self.batch_size]
            if self.hidden_cache is not None:
                batch_emb = self._encode_from_cache(batch_item_ids)
            else:
                batch_titles = [item_id_to_title[item_id] for item_id in batch_item_ids]
                batch_emb = self._encode_live(batch_item_ids, batch_titles)
            for row_idx, item_id in enumerate(batch_item_ids):
                output[int(item_id)] = batch_emb[row_idx]
        return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract Stage2 TCL dual-view item embeddings.")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--base_model", type=str, default="", help="Stage1 checkpoint directory. If empty, try stage2 config.")
    parser.add_argument("--stage2_path", type=str, required=True, help="Stage2 TCL output dir or checkpoint dir.")
    parser.add_argument("--hidden_cache_path", type=str, default="", help="Optional cached backbone hidden states.")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--save_info", type=str, required=True)
    parser.add_argument(
        "--fusion_mode",
        type=str,
        default="concat",
        help="concat | semantic_repeat | semantic_only | cooccurrence_only | alpha_mix | whitened",
    )
    parser.add_argument("--fusion_alpha", type=float, default=0.5)
    parser.add_argument("--normalize_embeddings", type=int, default=0)
    parser.add_argument("--whitening_mode", type=str, default="relaxed", help="truncated | relaxed")
    parser.add_argument("--whitening_alpha", type=float, default=0.5)
    parser.add_argument("--whitening_variance_ratio", type=float, default=0.95)
    parser.add_argument("--item_prefix", type=str, default=None)
    parser.add_argument("--item_suffix", type=str, default=None)
    parser.add_argument("--attention_mask_type", type=str, default=None)
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--bf16", type=int, default=1)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    stage2_config = load_stage2_config(args.stage2_path)
    base_model = args.base_model or str(stage2_config.get("base_model", "")).strip()
    if not base_model:
        raise ValueError("base_model is required unless it can be resolved from stage2 config")
    item_settings = resolve_stage1_item_settings(
        base_model_path=base_model,
        item_prefix=args.item_prefix,
        item_suffix=args.item_suffix,
        attention_mask_type=args.attention_mask_type,
        max_length=args.max_length,
    )
    item_id_to_title, _ = load_item_titles(args.dataset)
    extractor = LLMAlignStage2Extractor(
        base_model_path=base_model,
        stage2_path=args.stage2_path,
        hidden_cache_path=args.hidden_cache_path,
        batch_size=int(args.batch_size),
        dataset_name=args.dataset,
        attention_mask_type=str(item_settings["attention_mask_type"]),
        max_length=int(item_settings["max_length"]),
        item_prefix=str(item_settings["item_prefix"]),
        item_suffix=str(item_settings["item_suffix"]),
        fusion_mode=args.fusion_mode,
        fusion_alpha=float(args.fusion_alpha),
        normalize_embeddings=bool(args.normalize_embeddings),
        whitening_mode=args.whitening_mode,
        whitening_alpha=float(args.whitening_alpha),
        whitening_variance_ratio=float(args.whitening_variance_ratio),
        bf16=bool(args.bf16),
    )
    item_embs = extractor.encode_items(item_id_to_title)
    save_dir = op.join("item_info", args.dataset)
    os.makedirs(save_dir, exist_ok=True)
    save_path = op.join(save_dir, f"{args.save_info}_title_item_embs.npy")
    np.save(save_path, item_embs)
    meta = {
        "dataset": args.dataset,
        "base_model": base_model,
        "stage2_path": args.stage2_path,
        "resolved_stage2_checkpoint": extractor.stage2_checkpoint,
        "stage2_state_path": extractor.stage2_state_path,
        "semantic_head_path": extractor.semantic_head_path,
        "cooccurrence_head_path": extractor.cooccurrence_head_path,
        "hidden_cache_path": args.hidden_cache_path if args.hidden_cache_path else "",
        "fusion_mode": args.fusion_mode,
        "fusion_alpha": float(args.fusion_alpha),
        "normalize_embeddings": bool(args.normalize_embeddings),
        "whitening_mode": args.whitening_mode,
        "whitening_alpha": float(args.whitening_alpha),
        "whitening_variance_ratio": float(args.whitening_variance_ratio),
        "attention_mask_type": item_settings["attention_mask_type"],
        "item_prefix": item_settings["item_prefix"],
        "item_suffix": item_settings["item_suffix"],
        "max_length": int(item_settings["max_length"]),
        "output_path": save_path,
        "shape": list(item_embs.shape),
        "stage2_config_path": os.path.join(os.path.dirname(extractor.stage2_checkpoint), STAGE2_CONFIG_NAME),
    }
    if extractor.whitening_metadata:
        meta.update(extractor.whitening_metadata)
    with open(f"{save_path}.meta.json", "w", encoding="utf-8") as file:
        json.dump(meta, file, ensure_ascii=False, indent=2)
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
