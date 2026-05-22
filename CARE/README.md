# CARE: Coldness-Adaptive Reranking Engine

A lightweight multimodal recommendation system that uses CLIP text embeddings as visual-semantic proxies for cold-start reranking.

**Core insight**: CLIP's text encoder produces embeddings in a joint vision-language space -- text descriptions serve as effective visual-semantic proxies without requiring image processing at runtime.

## Architecture

```
CARE (models/vara.py)
└── CF Backbone: user_emb[n_users, d] + item_emb[n_items, d]
    └── score_cf = normalized dot product / τ

Training: pure BPR loss on CF backbone scores.
Inference: CF scores are combined with raw CLIP text cosine similarity
via post-hoc coldness-gated fusion:

  w_txt = 1 / (1 + α · item_count)
  fused = (1 - w_txt) · cf_score + w_txt · clip_cos_sim

Cold items (count → 0) get w_txt → 1.0 (pure text).
Warm items (count → ∞) get w_txt → 0.0 (pure CF).
α controls transition steepness.
```

## Requirements

- Python 3.10+
- CUDA-compatible GPU (recommended: 24GB+ VRAM)
- Packages: see `requirements.txt`

```bash
pip install -r requirements.txt
```

## Data Acquisition

### Amazon Review Datasets

CARE uses the Amazon Reviews 2023 dataset (McAuley Lab):

1. Download from [https://amazon-reviews-2023.github.io/](https://amazon-reviews-2023.github.io/)
2. Place raw files in `data/raw/`:

| Dataset               | Interactions File                  | Metadata File                  |
|-----------------------|------------------------------------|--------------------------------|
| Baby Products         | `Baby_Products.jsonl`              | `meta_Baby_Products.jsonl`     |
| Office Products       | `Office_Products.jsonl`            | `meta_Office_Products.jsonl`   |
| Sports & Outdoors     | `Sports_and_Outdoors.jsonl`        | `meta_Sports_and_Outdoors.jsonl` |
| All Beauty            | `All_Beauty.jsonl`                 | `meta_All_Beauty.jsonl`        |

### Yelp Dataset

1. Download from [https://www.yelp.com/dataset](https://www.yelp.com/dataset)
2. Place `review.json` and `business.json` in `data/raw/yelp/`

## Data Preprocessing

### Amazon datasets (standard pipeline)

```bash
# Single dataset
bash scripts/run_pipeline_new.sh baby
bash scripts/run_pipeline_new.sh office
bash scripts/run_pipeline_new.sh sports

# All datasets
bash scripts/run_pipeline_new.sh all
```

Pipeline steps:
1. `scripts/preprocess.py` — K-core filtering + Leave-One-Out temporal split + ID mapping
2. `scripts/extract_meta.py` — Metadata extraction + CLIP text assembly (title + description + reviews)
3. `scripts/cache_clip_features.py` — CLIP ViT-L/14 text feature extraction → `cache/{dataset}/clip_text.npy`
4. `scripts/pretrain_lightgcn.py` — LightGCN collaborative embedding pretraining (optional, for baselines)

### With image download (optional)

```bash
bash scripts/run_preprocess.sh baby
# Set SKIP_DOWNLOAD=1 to skip image download
```

### Yelp dataset

```bash
python scripts/preprocess_yelp.py
python scripts/cache_clip_features.py --dataset yelp
```

### Cache files generated

```
cache/{dataset}/
├── clip_text.npy     [N_items, 768] — CLIP text features (required)
├── clip_cls.npy      [N_items, 768] — CLIP visual features (optional, only if images downloaded)
├── item_emb.npy      [N_items, 64]  — LightGCN item embeddings (optional)
├── user_emb.npy      [N_users, 64]  — LightGCN user embeddings (optional)
├── item_ids.json     — Ordered item ID list
└── user_ids.json     — Ordered user ID list
```

## Training

```bash
# Train CARE on any dataset
python train.py --dataset baby --seeds 42 123 456
python train.py --dataset office --seeds 42 123 456
python train.py --dataset sports --seeds 42 123 456

# Smoke test (2 epochs, quick verification)
python train.py --dataset baby --smoke_test
```

Key flags: `--d` (embedding dim), `--lr`, `--margin`, `--batch_size`, `--epochs`

## Evaluation

```bash
# Full evaluation (warm + cold-start + stratified)
python evaluate.py --dataset baby --seeds 42 123 456
```

Output per dataset:
- `results/{dataset}_vara_results.json` — Full per-seed metrics + stratified analysis

## Experiments

### Baselines

```bash
# Original 5 baselines (BPR-MF, LightGCN, VBPR, BM3, FREEDOM)
python experiments/baselines/run_baselines.py --dataset baby --seeds 42 123 456

# Cold-start specialized baselines (DropoutNet, CLCRec, CCFCRec, MARec)
python experiments/baselines/run_coldstart_baselines.py --dataset baby --seeds 42 123 456

# New multimodal baselines (LGMRec, PEARL, PromptMM)
python experiments/baselines/run_new_baselines.py --dataset baby --seeds 42 123 456

# Stratified evaluation for all models
python experiments/baselines/run_stratified_baselines.py --dataset baby --seeds 42 123 456

# MAMEX-style modality gating baseline
python experiments/baselines/run_mamex.py --dataset baby --seeds 42 123 456

# Meta-learning cold-start baselines (MetaEmb, ProtoNet)
python experiments/baselines/meta_coldstart.py --dataset baby --seeds 42 123 456
```

### Ablations

```bash
# A1-A4: VARA, w/o Text, SBERT Text, Coldness Gate
python experiments/ablations/run_ablation.py --dataset baby --seeds 42 123 456

# Gate function design space comparison
python experiments/ablations/gate_functions.py --dataset baby

# Learned gate (2-param / 4-param)
python experiments/ablations/learned_gate.py --dataset baby

# Per-bucket alpha analysis
python experiments/ablations/per_bucket_alpha.py --dataset baby
```

### Full pipeline

```bash
bash experiments/run_all.sh baby
```

## Project Structure

```
CARE/
├── README.md
├── requirements.txt
├── train.py                    — Training entry point
├── evaluate.py                 — Evaluation entry point
├── models/
│   ├── vara.py                 — CARE model (BPR-MF backbone)
│   └── __init__.py
├── losses/
│   ├── vara_loss.py            — BPR loss for CARE
│   ├── bpr.py                  — BPR + in-batch BPR loss
│   └── __init__.py
├── data_utils/
│   ├── dataset.py              — MARADataset + CacheManager
│   └── __init__.py
├── evaluation/
│   ├── evaluator.py            — Fast batched CF evaluator
│   ├── metrics.py              — HR@K, NDCG@K
│   ├── full_rank.py            — Full-ranking evaluation
│   └── __init__.py
├── config/
│   ├── baby.yaml
│   └── beauty_sub.yaml
├── scripts/
│   ├── preprocess.py           — K-core + Leave-One-Out split
│   ├── preprocess_yelp.py      — Yelp preprocessing
│   ├── extract_meta.py         — Metadata + CLIP text assembly
│   ├── cache_clip_features.py  — CLIP feature extraction
│   ├── pretrain_lightgcn.py    — LightGCN pretraining
│   ├── verify_data.py          — Data verification
│   ├── verify_cache.py         — Cache verification
│   ├── run_preprocess.sh       — Full pipeline (with images)
│   ├── run_pipeline_new.sh     — VARA pipeline (text-only)
│   └── run_cache.sh            — Feature caching pipeline
└── experiments/
    ├── eval_protocol.py        — High-speed eval protocol
    ├── run_all.sh              — Full experiment pipeline
    ├── baselines/
    │   ├── models/
    │   │   ├── baselines.py    — BPR-MF, LightGCN, VBPR, BM3, FREEDOM
    │   │   ├── coldstart_models.py — DropoutNet, CLCRec, CCFCRec, MARec
    │   │   ├── graph_models.py — LGMRec, PEARL
    │   │   └── prompt_models.py — PromptMM
    │   ├── run_baselines.py
    │   ├── run_coldstart_baselines.py
    │   ├── run_new_baselines.py
    │   ├── run_stratified_baselines.py
    │   ├── run_mamex.py
    │   ├── mamex_style.py
    │   └── meta_coldstart.py
    ├── ablations/
    │   ├── run_ablation.py     — VARA ablation variants (A1-A4)
    │   ├── gate_functions.py   — Gate design space
    │   ├── learned_gate.py     — Learned gate (2/4 param)
    │   └── per_bucket_alpha.py — Per-bucket alpha
    └── analysis/
        ├── alpha_selection_rule.py
        ├── alpha_visualize.py
        ├── cross_model_report.py
        ├── full_rank_alpha_sweep.py
        ├── generate_paper_figures.py
        ├── make_tables.py
        ├── score_distribution.py
        └── variance_decomposition.py
```

## Key Design Decisions

- **Pure BPR-MF backbone**: Only user_emb + item_emb, no text adapter, no fusion weights. Training stays simple and stable.
- **Post-hoc text fusion**: CLIP text similarity applied only at inference — zero-shot, parameter-free.
- **τ_score=1.0**: Cosine similarity directly for BPR, same as BPR-MF baseline.
- **Coldness-gated fusion (CARE)**: Per-item dynamic weight `w = 1/(1+α·count)` smoothly transitions from text-reliant (cold) to CF-dominant (warm).
- **No images at runtime**: All features precomputed offline.

## Default Config

| Parameter | Value | Description |
|-----------|-------|-------------|
| `d` | 64 | Embedding dimension |
| `lr` | 1e-3 | Learning rate |
| `weight_decay` | 1e-5 | Weight decay |
| `margin` | 0.1 | BPR margin |
| `tau_score` | 1.0 | Score temperature |
| `max_epochs` | 50 | Max training epochs |
| `patience` | 6 | Early stopping patience |
| `batch_size` | 16384 | Auto-scaled per dataset |
| `k_list` | [5, 10, 20] | Evaluation cutoffs |
| `n_neg` | 99 | Val/test negative samples |

## Citation

If you use CARE in your research, please cite:

```bibtex
@article{CARE,
  title     = {CARE: Coldness-Adaptive Reranking Engine for Multimodal Recommendation},
  author    = {},
  journal   = {},
  year      = {2025},
}
```

## License

This project is released for research purposes.
