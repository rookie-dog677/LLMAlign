import pandas as pd
import torch
import torch.nn.functional as F
import numpy as np
import os
import os.path as op
import sys
import json
import argparse
from transformers import AutoModel, AutoTokenizer

if __package__ in {None, ""}:
    _THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    _REPO_ROOT = os.path.dirname(_THIS_DIR)
    if _REPO_ROOT not in sys.path:
        sys.path.insert(0, _REPO_ROOT)

from stage1.latent_attention import (
    LatentAttentionPooling,
    load_latent_attention_head,
)


dataset_name_mappings = {
    # 5-core filtered datasets
    "Games_5core": "Video_Games/5-core/downstream",
    "Arts_5core": "Arts_Crafts_and_Sewing/5-core/downstream",
    "Baby_5core": "Baby_Products/5-core/downstream",
}


def l2_normalize_np(embeddings: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / np.clip(norms, eps, None)


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

    pad_mask = attention_mask.to(dtype=dtype)
    min_value = torch.finfo(dtype).min
    additive_mask = (1.0 - pad_mask).unsqueeze(1).unsqueeze(2) * min_value
    seq_len = attention_mask.size(1)
    return additive_mask.expand(-1, 1, seq_len, -1).contiguous()


class CPAEncoder:
    """Encoder that loads a CPA-trained model (with optional LoRA adapter) for item embedding extraction."""

    def __init__(
        self,
        model_path,
        peft_model_name_or_path=None,
        pooling_mode="last_token",
        attention_mask_type="bidirectional",
        item_attention_mask_type="",
        max_length=128,
        normalize_embeddings=False,
        embedding_head_type="latent_attention",
        latent_head_path="",
        latent_num_latents=128,
        latent_num_cross_heads=8,
        latent_cross_dim_head=64,
        latent_ff_mult=4,
        latent_dim=-1,
    ):
        self.pooling_mode = pooling_mode
        resolved_attention_mask_type = item_attention_mask_type or attention_mask_type
        self.attention_mask_type = str(resolved_attention_mask_type).lower().strip()
        self.max_length = max_length
        self.normalize_embeddings = bool(normalize_embeddings)
        self.embedding_head_type = str(embedding_head_type).lower().strip()
        self.latent_head_path = latent_head_path
        if self.embedding_head_type not in {"token_pool", "latent_attention"}:
            raise ValueError(f"Unsupported embedding_head_type: {self.embedding_head_type}")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.attention_mask_type not in {"causal", "bidirectional"}:
            raise ValueError(f"Unsupported attention_mask_type: {self.attention_mask_type}")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
            )
        except Exception as e:
            print(f"[WARN] fast tokenizer load failed for {model_path}: {e}")
            print("[WARN] falling back to slow tokenizer (use_fast=False).")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
                use_fast=False,
            )
        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token_id is None:
                raise ValueError("Tokenizer has no pad/eos token, cannot pad safely.")
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "right"

        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        if peft_model_name_or_path is not None:
            try:
                from peft import PeftModel, LoraConfig

                _cfg_path = os.path.join(peft_model_name_or_path, "adapter_config.json")
                with open(_cfg_path) as _f:
                    _adapter_cfg = json.load(_f)
                _auto_mapping = _adapter_cfg.get("auto_mapping") or {}
                _base_cls = _auto_mapping.get("base_model_class", "")
                if "ConditionalGeneration" in _base_cls:
                    raise RuntimeError(
                        "PEFT adapters trained against ConditionalGeneration wrappers are not "
                        "supported by this open-source extraction path. Please provide an "
                        "AutoModel-compatible text encoder adapter or a merged text encoder checkpoint."
                    )

                self.model = AutoModel.from_pretrained(
                    model_path, dtype=dtype, trust_remote_code=True
                )
                peft_config = LoraConfig.from_pretrained(peft_model_name_or_path)
                peft_config.task_type = None
                peft_config.auto_mapping = None
                _peft_model = PeftModel.from_pretrained(self.model, peft_model_name_or_path, config=peft_config)
                self.model = _peft_model.merge_and_unload()
            except Exception as e:
                raise RuntimeError(f"Failed to load PEFT adapter for contrastive encoder: {e}")
        else:
            self.model = AutoModel.from_pretrained(
                model_path,
                dtype=dtype,
                trust_remote_code=True,
            )

        self.model.to(self.device)
        self.model.eval()
        self.model_dtype = next(self.model.parameters()).dtype
        self.latent_head = None
        if self.embedding_head_type == "latent_attention":
            self.latent_head = self._build_and_load_latent_head(
                model_path=model_path,
                peft_model_name_or_path=peft_model_name_or_path,
                latent_head_path=latent_head_path,
                latent_num_latents=latent_num_latents,
                latent_num_cross_heads=latent_num_cross_heads,
                latent_cross_dim_head=latent_cross_dim_head,
                latent_ff_mult=latent_ff_mult,
                latent_dim=latent_dim,
            )

    @staticmethod
    def _resolve_head_source(model_path, peft_model_name_or_path, latent_head_path):
        if latent_head_path:
            return latent_head_path
        if os.path.isdir(model_path):
            candidate = os.path.join(model_path, "latent_attention_head.pt")
            if os.path.exists(candidate):
                return candidate
        if peft_model_name_or_path and os.path.isdir(peft_model_name_or_path):
            candidate = os.path.join(peft_model_name_or_path, "latent_attention_head.pt")
            if os.path.exists(candidate):
                return candidate
        return None

    def _build_and_load_latent_head(
        self,
        model_path,
        peft_model_name_or_path,
        latent_head_path,
        latent_num_latents,
        latent_num_cross_heads,
        latent_cross_dim_head,
        latent_ff_mult,
        latent_dim,
    ):
        source = self._resolve_head_source(model_path, peft_model_name_or_path, latent_head_path)
        if source is None:
            raise FileNotFoundError(
                "embedding_head_type=latent_attention requires latent_attention_head.pt "
                "under model_path (or --latent_head_path override)."
            )

        saved_cfg, state_dict, resolved_path = load_latent_attention_head(source, map_location="cpu")
        hidden_size = int(getattr(self.model.config, "hidden_size"))
        num_latents = int(saved_cfg.get("num_latents", latent_num_latents))
        num_cross_heads = int(saved_cfg.get("num_cross_heads", latent_num_cross_heads))
        cross_dim_head = int(saved_cfg.get("cross_dim_head", latent_cross_dim_head))
        ff_mult = int(saved_cfg.get("ff_mult", latent_ff_mult))
        cfg_latent_dim = int(saved_cfg.get("latent_dim", latent_dim))
        if cfg_latent_dim <= 0:
            cfg_latent_dim = hidden_size

        head = LatentAttentionPooling(
            hidden_size=hidden_size,
            num_latents=num_latents,
            latent_dim=cfg_latent_dim,
            num_cross_heads=num_cross_heads,
            cross_dim_head=cross_dim_head,
            ff_mult=ff_mult,
        )
        missing, unexpected = head.load_state_dict(state_dict, strict=False)
        unexpected = [key for key in unexpected if not key.startswith("readout_gate.")]
        if missing or unexpected:
            raise ValueError(
                f"Failed to load latent attention head from {resolved_path}, "
                f"missing={missing}, unexpected={unexpected}"
            )
        head.to(device=self.device, dtype=self.model_dtype)
        head.eval()
        print(f"[cpa_encoder] loaded latent attention head: {resolved_path}")
        return head

    def _pool(self, hidden_states, attention_mask):
        if self.pooling_mode == "mean":
            mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
            summed = (hidden_states * mask).sum(dim=1)
            denom = mask.sum(dim=1).clamp(min=1.0)
            return summed / denom

        if self.pooling_mode != "last_token":
            raise ValueError(f"Unsupported pooling mode: {self.pooling_mode}")

        last_pos = attention_mask.to(dtype=torch.long).sum(dim=1) - 1
        last_pos = last_pos.clamp(min=0)
        batch_idx = torch.arange(hidden_states.size(0), device=hidden_states.device)
        return hidden_states[batch_idx, last_pos, :]

    @staticmethod
    def _to_plain_text(text) -> str:
        if isinstance(text, (list, tuple, np.ndarray)):
            parts = [str(x).strip() for x in text if str(x).strip()]
            return " ".join(parts)
        return str(text)

    def encode(self, sentences, batch_size):
        all_embeds = []
        for i in range(0, len(sentences), batch_size):
            batch_text = [self._to_plain_text(x) for x in sentences[i:i + batch_size]]
            batch = self.tokenizer(
                batch_text,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
                return_attention_mask=True,
            )
            batch = {k: v.to(self.device) for k, v in batch.items()}
            model_attention_mask = batch["attention_mask"]
            if self.attention_mask_type == "bidirectional":
                model_attention_mask = build_bidirectional_attention_mask_dict(
                    batch["attention_mask"], dtype=self.model_dtype
                )
            with torch.no_grad():
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=model_attention_mask,
                    return_dict=True,
                )
                if self.embedding_head_type == "latent_attention":
                    pooled = self.latent_head(
                        outputs.last_hidden_state,
                        attention_mask=batch["attention_mask"],
                    ).float()
                else:
                    pooled = self._pool(outputs.last_hidden_state, batch["attention_mask"]).float()
                if self.normalize_embeddings:
                    pooled = F.normalize(pooled, dim=-1)
            all_embeds.append(pooled.detach().cpu().numpy())

        return np.concatenate(all_embeds, axis=0)


def apply_item_affixes(prompts, item_prefix: str, item_suffix: str):
    if not item_prefix and not item_suffix:
        return prompts

    if isinstance(prompts, np.ndarray) and prompts.ndim == 2:
        prompts = prompts.copy()
        prompts[:, 1] = np.array(
            [f"{item_prefix}{str(x)}{item_suffix}" for x in prompts[:, 1]],
            dtype=object,
        )
        return prompts

    return np.array([f"{item_prefix}{str(x)}{item_suffix}" for x in prompts], dtype=object)


def extract_item_embedding_with_prompts(
    dataset_name,
    model_path,
    peft_path,
    batch_size,
    prompt_type,
    save_info=None,
    encoder_mode="cpa_encoder",
    pooling_mode="last_token",
    attention_mask_type="bidirectional",
    item_attention_mask_type="",
    normalize_embeddings=False,
    item_prefix="",
    item_suffix="",
    max_length=128,
    embedding_head_type="latent_attention",
    latent_head_path="",
    latent_num_latents=128,
    latent_num_cross_heads=8,
    latent_cross_dim_head=64,
    latent_ff_mult=4,
    latent_dim=-1,
):
    peft_path = peft_path or None

    raw_dataset_name = dataset_name_mappings[dataset_name]
    with open(f"./data/{raw_dataset_name}/item_titles.json", 'r', encoding='utf-8') as file:
        item_metadata = json.load(file)

    item_ids = [int(int_id) for int_id in item_metadata.keys()]
    max_item_id = max(item_ids)
    assert 0 not in item_ids, "Item IDs should not contain 0"

    item_titles = ["Null"]
    for i in range(1, max_item_id + 1):
        item_titles.append(item_metadata[str(i)])

    if encoder_mode == "cpa_encoder":
        model = CPAEncoder(
            model_path,
            peft_model_name_or_path=peft_path,
            pooling_mode=pooling_mode,
            attention_mask_type=attention_mask_type,
            item_attention_mask_type=item_attention_mask_type,
            max_length=max_length,
            normalize_embeddings=normalize_embeddings,
            embedding_head_type=embedding_head_type,
            latent_head_path=latent_head_path,
            latent_num_latents=latent_num_latents,
            latent_num_cross_heads=latent_num_cross_heads,
            latent_cross_dim_head=latent_cross_dim_head,
            latent_ff_mult=latent_ff_mult,
            latent_dim=latent_dim,
        )
    else:
        raise ValueError(f"Unsupported encoder_mode: {encoder_mode}")

    item_infos = np.array(item_titles)
    if prompt_type == "direct":
        prompts = generate_direct_item_prompt_pog(item_infos)
    elif prompt_type == "title":
        prompts = generate_title_item_prompt_pog(item_infos)
    else:
        raise ValueError(f"Unsupported item_prompt_type: {prompt_type}")

    prompts = apply_item_affixes(prompts, item_prefix=item_prefix, item_suffix=item_suffix)

    item_embeddings = model.encode(prompts, batch_size)

    save_path = f"./item_info/{dataset_name}/"
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    if save_info is not None:
        model_name = f"{save_info}"
    else:
        model_name = model_path.replace("/", "_")
    np.save(op.join(save_path, f"{model_name}_{prompt_type}_item_embs.npy"), item_embeddings)


def generate_direct_item_prompt_pog(item_info):
    instruct = "To recommend this item to users, this item can be described as: "
    instructs = np.repeat(instruct, len(item_info))
    prompts = item_info

    outputs = np.concatenate((instructs[:, np.newaxis], prompts[:, np.newaxis]), axis=1)
    return outputs


def generate_title_item_prompt_pog(item_info):
    prompts = item_info
    return prompts


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract item embeddings with prompts.")
    parser.add_argument('--dataset', type=str, default="Arts_5core", help="Name of the dataset")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size for processing")
    parser.add_argument('--model_path', type=str, default='./', help="Path to the model")
    parser.add_argument('--peft_path', type=str, default=None, help="Path to the PEFT model")
    parser.add_argument('--item_prompt_type', type=str, default="title", help="Type of item prompt")
    parser.add_argument('--save_info', type=str, default="Test-only", help="Save information identifier")
    parser.add_argument('--encoder_mode', type=str, default="cpa_encoder", help="Encoder mode")
    parser.add_argument('--pooling_mode', type=str, default="last_token", help="Pooling mode: last_token or mean")
    parser.add_argument('--attention_mask_type', type=str, default="bidirectional", help="Attention mask type: causal or bidirectional")
    parser.add_argument('--item_attention_mask_type', type=str, default="", help="Optional item-side attention mask type override: causal or bidirectional")
    parser.add_argument('--embedding_head_type', type=str, default="latent_attention", help="Embedding head type: token_pool or latent_attention")
    parser.add_argument('--latent_head_path', type=str, default="", help="Optional explicit path to latent_attention_head.pt")
    parser.add_argument('--latent_num_latents', type=int, default=128, help="Latent token count for latent_attention head")
    parser.add_argument('--latent_num_cross_heads', type=int, default=8, help="Cross-attention heads for latent_attention head")
    parser.add_argument('--latent_cross_dim_head', type=int, default=64, help="Per-head dim for latent_attention head")
    parser.add_argument('--latent_ff_mult', type=int, default=4, help="FFN width multiplier for latent_attention head")
    parser.add_argument('--latent_dim', type=int, default=-1, help="Latent hidden dim for latent_attention head; <=0 means model hidden size")
    parser.add_argument('--normalize_embeddings', type=int, default=0, help="Whether to L2 normalize embeddings: 1 or 0")
    parser.add_argument('--item_prefix', type=str, default="", help="Optional prefix added before each item title")
    parser.add_argument('--item_suffix', type=str, default="", help="Optional suffix added after each item title")
    parser.add_argument('--max_length', type=int, default=128, help="Max token length for encoder")
    args = parser.parse_args()
    extract_item_embedding_with_prompts(
        dataset_name=args.dataset,
        model_path=args.model_path,
        peft_path=args.peft_path,
        batch_size=args.batch_size,
        prompt_type=args.item_prompt_type,
        save_info=args.save_info,
        encoder_mode=args.encoder_mode,
        pooling_mode=args.pooling_mode,
        attention_mask_type=args.attention_mask_type,
        item_attention_mask_type=args.item_attention_mask_type,
        normalize_embeddings=bool(args.normalize_embeddings),
        item_prefix=args.item_prefix,
        item_suffix=args.item_suffix,
        max_length=args.max_length,
        embedding_head_type=args.embedding_head_type,
        latent_head_path=args.latent_head_path,
        latent_num_latents=args.latent_num_latents,
        latent_num_cross_heads=args.latent_num_cross_heads,
        latent_cross_dim_head=args.latent_cross_dim_head,
        latent_ff_mult=args.latent_ff_mult,
        latent_dim=args.latent_dim,
    )


if __name__ == "__main__":
    main()
