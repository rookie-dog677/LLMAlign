# LLMAlign: Making Large Language Models Effective Embedding Models for Sequential Recommendation

## Introduction

This is the implementation of the submission "LLMAlign: Making Large Language Models Effective Embedding Models for Sequential Recommendation".

## Environments

- `torch == 2.6.0`
- `transformers == 5.3.0`
- `peft == 0.18.1`
- `accelerate == 1.13.0`

```bash
pip install -r requirements.txt
```

## Datasets

The preprocessed datasets (`Games_5core`, `Arts_5core`, `Baby_5core` from Amazon Reviews 2023) can be downloaded from this [link](https://drive.google.com/file/d/1RxxEGCT3ZGGmYX0649XiwmObyiXq6L8S/view). Please unzip the files under directory `./data`.

## Training

LLMAlign follows a two-stage training pipeline. Download a local copy of the backbone first (e.g. [Qwen2-0.5B](https://huggingface.co/Qwen/Qwen2-0.5B)) — `BASE_MODEL` must point to a local directory.

```bash
# Stage 1: Contrastive Preference Alignment
BASE_MODEL=/path/to/local/Qwen2-0.5B LLMALIGN_DATASET=Games_5core bash run_LLMAlign_CPA

# Stage 2: Temporal/Co-occurrence Contrastive Learning
LLMALIGN_DATASET=Games_5core bash run_LLMAlign_TCL
```

## Evaluation

Extract the fused item embedding from a Stage 2 checkpoint, then evaluate with a downstream recommender:

```bash
# Extract dual-view embedding (1792d concat by default)
python stage2/extract_stage2_llmalign_embedding.py \
    --dataset Games_5core \
    --stage2_path output/<stage2_run>/best_checkpoint \
    --save_info llmalign_games \
    --fusion_mode concat

# Downstream evaluation
python evaluate_with_seqrec.py \
    --model SASRec \
    --dataset Games_5core \
    --embedding item_info/Games_5core/llmalign_games_title_item_embs.npy \
    --item_adapter_type=moe \
    --moe_n_exps=8 --moe_top_k=2 --moe_balance_weight=1.0e-3 \
    --rand_seed=2024
```

## License

Released under the [Apache License 2.0](LICENSE).
