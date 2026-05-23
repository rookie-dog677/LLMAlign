# LLMAlign 仓库协作指南

## 项目概述

LLMAlign 是一个两阶段双视图对比对齐框架，利用 LLM 为推荐系统生成高质量 item embedding。核心思路：通过 item-level 对比学习，将用户偏好信号（user-item）和物品共现信号（item-item）逐阶段注入 LLM 表征空间，再将语义视图与共现视图接入下游序列推荐模型。

- **Stage 1 - CPA（Contrastive Preference Alignment）**：微调 LLM backbone 与 Latent Attention head，以 user-item InfoNCE 注入偏好信号。
- **Stage 2 - TCL**：冻结 Stage 1 backbone 与 CPA head，只训练独立的 Co-occurrence head，以 item-item InfoNCE 注入共现信号。
- **融合**：默认抽取 `concat(cpa_emb, cooccurrence_emb)`。以 Qwen2-0.5B hidden size 896 为例，输出维度为 `1792d`，下游通常通过 MoE adapter 融合。

---

## 当前项目结构

```text
LLMAlign/
├── stage1/                         # Stage 1 CPA
│   ├── run_llmalign_cpa.py         #   训练主逻辑（fire.Fire(train)）
│   ├── latent_attention.py         #   Latent Attention Pooling head
│   └── extract_llm_embedding.py    #   Stage 1 item embedding 抽取
├── stage2/                         # Stage 2 TCL
│   ├── run_llmalign_tcl.py         #   cache_hidden / mine_pairs / train 三阶段实现
│   ├── stage2_llmalign_utils.py    #   数据集映射、hidden cache、pair mining、checkpoint 工具
│   └── extract_stage2_llmalign_embedding.py  # Stage 2 embedding 抽取与融合
├── seqrec/                         # 下游序列推荐框架
│   ├── runner.py                   #   评估/训练入口，wandb 集成
│   ├── trainer.py / evaluator.py   #   训练与评估逻辑
│   ├── modules.py                  #   通用模块（MoE adapter 等）
│   ├── models/                     #   SASRec / BERT4Rec / SRGNN
│   ├── recdata.py                  #   下游数据加载与 cooccurrence bucket eval
│   └── default.yaml                #   默认超参配置
├── data/                           # 当前本地数据
│   ├── Video_Games/
│   ├── Arts_Crafts_and_Sewing/
│   └── Baby_Products/
├── evaluate_with_seqrec.py         # 单次下游评估
├── evaluate_with_seqrec_single_seed.py # 单 seed 评估并写 JSON
├── requirements.txt                # 当前环境依赖版本
├── run_LLMAlign_CPA                # Stage 1 顶层入口脚本
└── run_LLMAlign_TCL                # Stage 2 顶层入口脚本（一键 cache -> mine -> train）
```

说明：当前 checkout 只保留 `Games_5core`、`Arts_5core`、`Baby_5core` 三个本地数据集映射，对应 `Video_Games`、`Arts_Crafts_and_Sewing`、`Baby_Products`。`script_extract_and_evaluate*.sh`、`parallel_evaluate_with_seqrec.py`、`run_batch_experiments.py` 当前不存在，不要在新流程中引用。

---

## 运行流程

### Stage 1 训练（CPA）

```bash
BASE_MODEL=/path/to/local/Qwen2-0.5B bash run_LLMAlign_CPA

# 环境变量覆盖示例
BASE_MODEL=/path/to/local/Qwen2-0.5B LLMALIGN_DATASET=Baby_Products LEARNING_RATE=5e-5 bash run_LLMAlign_CPA
```

`BASE_MODEL` / `LLMALIGN_BASE_MODEL` 必须显式指向本地 HuggingFace 模型或 checkpoint 目录；顶层脚本不解析 HuggingFace model id。`LLMALIGN_DATASET` / `DATASET` 支持 `Video_Games|Games_5core`、`Arts_Crafts_and_Sewing|Arts_5core`、`Baby_Products|Baby_5core` 等别名，并自动从 `data/<raw_dataset>/5-core/{train,valid}` 解析 CSV。

### Stage 2 训练（TCL）

```bash
# 一键运行：cache_hidden -> mine_pairs -> train
bash run_LLMAlign_TCL

# 分阶段运行
python stage2/run_llmalign_tcl.py cache_hidden ...
python stage2/run_llmalign_tcl.py mine_pairs ...
python stage2/run_llmalign_tcl.py train ...
```

`run_LLMAlign_TCL` 默认 `DATASET=Games_5core`，会优先使用 `BASE_MODEL` / `LLMALIGN_BASE_MODEL`；未设置时，会在 `output/` 下按当前 raw dataset 自动寻找最近的 Stage 1 checkpoint。默认 Co-occurrence head 从 CPA latent head 初始化（`COOCCURRENCE_HEAD_INIT=semantic`，run tag 为 `cpainit`）。随机初始化只用于消融，必须同时设置：

```bash
COOCCURRENCE_HEAD_INIT=random ALLOW_RANDOM_COOCCURRENCE_HEAD_INIT=1 bash run_LLMAlign_TCL
```

### 抽取 embedding

```bash
# Stage 1 CPA embedding
python stage1/extract_llm_embedding.py \
  --dataset Games_5core \
  --model_path <stage1_checkpoint_dir> \
  --save_info <name> \
  --embedding_head_type latent_attention \
  --normalize_embeddings 0

# Stage 2 双视图 embedding，默认 concat
python stage2/extract_stage2_llmalign_embedding.py \
  --dataset Games_5core \
  --stage2_path <stage2_output_or_checkpoint_dir> \
  --hidden_cache_path <optional_hidden_cache.pt> \
  --save_info <name> \
  --fusion_mode concat \
  --normalize_embeddings 0
```

Stage 2 抽取支持 `concat`、`semantic_repeat`、`semantic_only`、`cooccurrence_only`、`alpha_mix`、`whitened`。输出统一写到 `item_info/<dataset>/<save_info>_title_item_embs.npy`，并生成相邻的 `.meta.json`。

### 下游评估

```bash
python evaluate_with_seqrec.py \
  --model SASRec \
  --dataset Games_5core \
  --embedding item_info/Games_5core/<name>_title_item_embs.npy \
  --item_adapter_type=moe \
  --moe_n_exps=8 \
  --moe_top_k=2 \
  --moe_balance_weight=1.0e-3 \
  --rand_seed=2024 \
  --wandb_group=SASRec_Games_llmalign
```

`evaluate_with_seqrec_single_seed.py` 与上面参数一致，但额外要求 `--out_json <path>`，会把 `test_result`、最终 config 和合并后的命令行参数写入 JSON。长时间训练或评估任务统一用 `tmux` 脱离会话启动，不要依赖当前 agent 前台 `exec` 或 `nohup`。

---

## 主线超参

### Stage 1（CPA）

以 `AGENTS.md` 中这套主线超参为准；`run_LLMAlign_CPA` 与 `stage1/run_llmalign_cpa.py` 默认值应保持同步。

| 类别 | 当前默认 |
|------|----------|
| 训练 | `per_device_train_batch_size=300`、`per_device_eval_batch_size=300`、`num_epochs=4`、`max_steps=-1`、`learning_rate=5e-5`、`warmup_steps=200`、`early_stopping_patience=5` |
| 长度 | `max_user_len=256`、`max_item_len=128` |
| 编码 | `attention_mask_type=bidirectional`（user/item 均继承） |
| Head | `embedding_head_type=latent_attention`、`latent_num_latents=128`、`latent_num_cross_heads=8`、`latent_cross_dim_head=64`、`latent_ff_mult=4` |
| 对比学习 | `contrastive_loss_mode=multi_positive_nce`、`temperature=0.07`、`learnable_temperature=True`、`logit_scale_max=100.0`、`normalize_embeddings=True` |
| 正样本归组 | `positive_group_by=item_id` |
| 负采样 | `sampled_negatives=8`、`negative_sampling_strategy=hard_overlap`、`negative_hard_topk=32`、`negative_hard_history_weight=0.35`、`negative_avoid_batch_positives=True`、`negative_dedup_per_query=False` |
| 文本 | `history_sep=', '`；`user_suffix` / `item_suffix` 自动取 tokenizer eos（只接受 `<\|endoftext\|>` 或 `<\|im_end\|>`，否则回退 `<\|endoftext\|>`） |
| 保存/日志 | `eval_steps=50`、`save_steps=eval_steps`、`save_total_limit=3`、`report_to=none` |
| 精度 | `bf16=True` |

### Stage 2（TCL）

| 类别 | 当前默认 |
|------|----------|
| 冻结策略 | Stage 1 backbone 不参与训练；CPA semantic head 只作为初始化/抽取语义视图；训练 Co-occurrence head 与 learnable temperature |
| Base model | `BASE_MODEL` 显式优先；否则从 `output/` 自动查找当前 dataset 的 Stage 1 checkpoint |
| Hidden cache | `cache_batch_size=32`、`cache_dtype=fp16`、`cache_bf16=1`、item 文本设置继承 Stage 1 `cpa_encoder_config.json` |
| Pair mining | `pair_mining_mode=sliding_window`、`window_size=3`、`min_count=3`、`expand_bidirectional_pairs=1`、`compute_ppmi=1`、`ppmi_alpha=0.75`、`sample=-1` |
| 训练 | `batch_size=1024`、`eval_batch_size=1024`、`lr=1e-4`、`weight_decay=0.01`、`warmup_ratio=0.05`、`max_steps=500`（Baby 为 `250`）、`max_grad_norm=1.0` |
| 对比学习 | `temperature=0.05`、`learnable_temperature=1`、`logit_scale_max=100.0`、`normalize_embeddings=1` |
| 正样本采样 | `positive_sampling_mode=ppmi`（要求 pairs payload 含 `pair_ppmi`） |
| 负样本 | `hard_negative_mode=none`；可选 `semantic`，配套 `hard_negative_topk=20` |
| Head 初始化 | `cooccurrence_head_init=semantic`（CPA head 权重初始化）；`random` 必须显式解锁 |
| 评估/保存 | `eval_steps=50`、`eval_ratio=0.05`、`max_eval_pairs=50000`、`save_steps=0`、`save_only_best=1`、`early_stopping_patience=0` |
| 精度 | `bf16=1` |

### 下游 SeqRec

- 支持模型：`SASRec`、`BERT4Rec`、`SRGNN`。
- 输入 embedding：`.npy` shape 必须是 `[item_num + 1, dim]`，index 0 为 padding 占位。
- 默认模型配置里的 `item_adapter_type=linear`；使用 LLMAlign 1792d embedding 时通常要显式传 `--item_adapter_type=moe`。
- 当前 MoE 默认：`moe_n_exps=8`、`moe_top_k=2`、`moe_dropout=0.0`、`moe_noise=False`、`moe_temperature=1.0`、`moe_balance_weight=1.0e-3`。
- `item_adapter_type=none` 要求预训练 embedding 维度等于模型 `hidden_size`（默认 128）。
- `item_adapter_type=llmemb_bottleneck` 用于高维单视图 LLM embedding 到 128d 的 bottleneck adapter。
- 可通过 `--cooccurrence_bucket_eval=True` 开启按训练共现次数分桶的评估，默认 `cooccurrence_count_mode=sliding_window`、`cooccurrence_window_size=3`。

---

## 接口约束

- Stage 1 输出保持 HuggingFace 标准目录结构，并额外保存 `latent_attention_head.pt` 与 `cpa_encoder_config.json`。
- Stage 2 输出目录包含 `stage2_llmalign_config.json`、`best_checkpoint`、以及至少一个 `checkpoint-*`。
- Stage 2 checkpoint 中保存 `latent_attention_head.pt`（co-occurrence head 权重）与 `stage2_llmalign_state.pt`。
- `best_checkpoint` 是优先解析目标；如果不存在，Stage 2 抽取会回退到最新的 `checkpoint-*`。
- Stage 2 中间产物由顶层脚本自动生成到 `output/`：`*_hidden_cache_<DownstreamDataset>.pt` 与 `*_pairs_<RawDataset>_mc3_ppmi.pt`。
- Stage 2 默认 `SAVE_ONLY_BEST=1`，非 best checkpoint 会被清理；需要保留所有 checkpoint 时显式设 `SAVE_ONLY_BEST=0` 并设置 `SAVE_STEPS`。
- 抽取后的 item embedding 命名使用 `<save_info>_title_item_embs.npy`；保持 `best_raw` 这类下游实验命名时使用下划线，不要改成 `best-raw`。

---

## WandB 规范

- 统一 project：`LLMAlign_Eval`。
- `group` 用于聚合同一组实验的多个 seed，推荐格式：`{Model}_{DatasetShort}_{method}`，例如 `SASRec_Games_llmalign`。
- 传参方式：通过 `--wandb_group=SASRec_Games_llmalign` 传入；`seqrec/runner.py` 中 `wandb.init(group=config.get('wandb_group'))` 读取。
- `name` 由 `seqrec/utils.py:get_file_name()` 自动生成，包含 `run_id`、命令行片段、时间戳和 config hash。
- 当前 checkout 没有内置批量实验脚本；多 seed 批量运行请用外部脚本或 tmux，确保每个 seed 显式传 `--rand_seed`、`--run_id` 和 `--wandb_group`。

### 下游 adapter 对照

| 方法 | embedding 维度 | item_adapter_type | 额外参数 |
|------|---------------|-------------------|----------|
| clean | 无 | 不传 `--embedding` | - |
| 128d embedding | 128d | `none` | 维度必须等于 `hidden_size` |
| LLMEmb | 896d | `llmemb_bottleneck` | 高维单视图到 128d |
| LLMAlign | 1792d | `moe` | `moe_n_exps=8`、`moe_top_k=2`、`moe_balance_weight=1.0e-3` |
