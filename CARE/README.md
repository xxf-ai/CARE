# CARE: a coldness-aware reliability calibration method for multimodal cold-start recommendation

CARE is a coldness-aware reliability calibration method for multimodal cold-start recommendation. Instead of building a new fusion module or backbone, CARE addresses a more specific question: **when an item has zero training interactions, should the collaborative filtering (CF) score still participate in the final ranking?**

**Core answer**: On zero-interaction items, CF is not just a weak signal — it can be *actively harmful*, contaminating otherwise usable semantic rankings from text and image. CARE uses a single-parameter, inference-time gate to deterministically remove CF when `c(i) = 0` and gradually restore it as interaction evidence accumulates.

## Research Overview

### Problem: Channel-Trust Calibration

Traditional multimodal cold-start recommendation treats CF as universally informative and supplements it with side information (text, images, metadata). CARE reframes this as a **channel-trust calibration** problem:

- **Collaborative channel**: CF scores learned from interaction data — reliable for warm items, unreliable for cold items.
- **Semantic channel** (text / image): Zero-shot semantic evidence from CLIP — does not depend on target-item interactions, remains informative even at zero interactions.

The key question is not *how to fuse more signals*, but **how to judge which signals are trustworthy at each item coldness level**.

### Core Formula

```
ŝ(u,i) = (1 − w(i)) · s_cf(u,i)  +  w(i) · s_sem(u,i)

w(i) = 1 / (1 + α · c(i))
```

- `c(i)`: item `i`'s interaction count in training data
- `w(i)`: semantic channel weight
- `1 − w(i)`: collaborative channel weight
- `s_sem ∈ {s_text, s_image}`: semantic score from text or image

**Critical boundary condition**: When `c(i) = 0`, `w(i) = 1` and `1 − w(i) = 0`. CARE *deterministically removes* CF from the score — not just down-weights it.

As `c(i)` grows, CF weight gradually recovers. CARE never permanently relies on semantic channels; it adapts to the evidence available per item.

### Architecture

```
CARE (models/care.py)
└── CF Backbone: user_emb[n_users, d] + item_emb[n_items, d]
    └── score_cf = normalized dot product / τ

Training: pure BPR loss on CF backbone scores (no text adapter, no fusion weights).
Inference: coldness-gated fusion is applied post-hoc.
```

### Key Design Properties

| Property | Implication |
|----------|-------------|
| Training unchanged | CF backbone trains on collaborative signals only; no interference from multimodal features |
| Semantic encoder frozen | CLIP features precomputed offline; zero-shot, no fine-tuning |
| Inference-time gate only | The gate is not a trained module; it applies a deterministic function of item count |
| `w(0) = 1` boundary | Zero-interaction items get pure semantic score — CF is structurally excluded |
| One parameter (`α`) | Controls the transition steepness from text-reliant to CF-dominant |
| Text and image both supported | `s_sem` can be CLIP text or CLIP visual; the gate rule is modality-agnostic |

## Research Questions

The experiments are organized around four Research Questions:

### RQ1: Where and How Does CF Hurt?

Prove that CF collapse at zero interactions is real, and that semantic channels (text / image) remain usable. CARE-Text and CARE-Image both recover cold-start performance.

**Core evidence**: Stratified evaluation across coldness buckets:
- **L0** (zero-shot): CF nearly collapses; text/image still provide usable signal; CARE recovers performance.
- **L1–L2** (cold/warm): CARE smoothly transitions from semantic to CF.
- **L3** (hot): CF recovers strong signal; CARE no longer over-relies on semantics.

### RQ2: Is This Only a Weak-Backbone Artifact?

Prove that L0 CF collapse is not specific to BPR-MF. Tested across 11 models including LightGCN, VBPR, BM3, LGMRec, PEARL, PromptMM, DropoutNet, CLCRec, CCFCRec, MARec.

Even recent multimodal models and cold-start-specialized models expose unreliable CF scores on zero-interaction items. The problem is about *scoring reliability*, not backbone choice.

### RQ3: Robustness of Coldness-Aware Trust Calibration

Prove the boundary condition is not naturally learned by alternatives:
- **Fixed-weight fusion**: Improves cold-start but cannot distinguish L0 from L3.
- **Learned gate**: Under aggregate full-set objective, a 2-parameter learned gate retains 73%–84% CF weight even at `c(i) = 0` — it does not naturally discover `w(0) = 1`.
- **Full-set metrics**: Aggregate NDCG changes minimally, while L0 NDCG changes dramatically — aggregate evaluation masks cold-tail failure.
- **Full-ranking**: Full-ranking objective favors warm majority; the optimal α differs from sampled evaluation.
- **Cross-domain (Yelp)**: The same CF reliability pattern holds across domains.

### RQ4: Why Is One Parameter Sufficient?

If CF estimator uncertainty decreases with item interaction count, and semantic estimator variance is relatively independent of target-item interactions, then the minimum-variance linear combination naturally raises CF weight as coldness decreases. CARE's inverse gate is a simple, interpretable approximation of this reliability transition.

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
- `results/{dataset}_care_results.json` — Full per-seed metrics + stratified analysis

## Experiments

### Baselines

```bash
# Original 5 baselines (BPR-MF, LightGCN, VBPR, BM3, FREEDOM)
python experiments/baselines/run_baselines.py --dataset baby --seeds 42 123 456

# Cold-start specialized baselines (DropoutNet, CLCRec, CCFCRec, MARec)
python experiments/baselines/run_coldstart_baselines.py --dataset baby --seeds 42 123 456

# New multimodal baselines (LGMRec, PEARL, PromptMM)
python experiments/baselines/run_new_baselines.py --dataset baby --seeds 42 123 456

# Stratified evaluation for all models (RQ2: cross-model diagnostic)
python experiments/baselines/run_stratified_baselines.py --dataset baby --seeds 42 123 456

# MAMEX-style modality gating baseline
python experiments/baselines/run_mamex.py --dataset baby --seeds 42 123 456

# Meta-learning cold-start baselines (MetaEmb, ProtoNet)
python experiments/baselines/meta_coldstart.py --dataset baby --seeds 42 123 456
```

### Ablations

```bash
# A1-A4: CARE, w/o Text, SBERT Text, Coldness Gate
python experiments/ablations/run_ablation.py --dataset baby --seeds 42 123 456

# Gate function design space comparison (RQ3)
python experiments/ablations/gate_functions.py --dataset baby

# Learned gate (2-param / 4-param) — RQ3 diagnostic
python experiments/ablations/learned_gate.py --dataset baby

# Per-bucket alpha analysis (RQ4)
python experiments/ablations/per_bucket_alpha.py --dataset baby
```

### Analysis Tools

```bash
# Alpha selection rule analysis
python experiments/analysis/alpha_selection_rule.py --dataset baby

# Gate activation visualization
python experiments/analysis/alpha_visualize.py --dataset baby --seed 42

# Cross-model stratified report (RQ2)
python experiments/analysis/cross_model_report.py --dataset baby

# Full-ranking alpha sweep (RQ3: full-ranking validation)
python experiments/analysis/full_rank_alpha_sweep.py --dataset baby

# Variance decomposition analysis
python experiments/analysis/variance_decomposition.py --dataset baby

# Paper figure generation
python experiments/analysis/generate_paper_figures.py

# Paper table generation
python experiments/analysis/make_tables.py --dataset baby
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
├── evaluate.py                 — Evaluation entry point (coldness-stratified)
├── models/
│   ├── care.py                 — CARE model (BPR-MF backbone)
│   └── __init__.py
├── losses/
│   ├── care_loss.py            — BPR loss for CARE
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
│   ├── run_pipeline_new.sh     — CARE pipeline (text-only)
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
    │   ├── run_ablation.py     — CARE ablation variants (A1-A4)
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

- **Pure BPR-MF backbone**: Only user_emb + item_emb, no text adapter, no fusion weights. Training stays simple and stable — the CF backbone learns clean collaborative signals without interference from multimodal features.
- **Post-hoc semantic fusion**: CLIP text/image similarity applied only at inference — zero-shot, parameter-free. This keeps the method deployable without retraining and proves the benefit comes from the gate, not from a better fusion module.
- **τ_score=1.0**: Cosine similarity directly for BPR, same as BPR-MF baseline.
- **Coldness-gated fusion**: Per-item dynamic weight `w = 1/(1+α·count)` smoothly transitions from semantic-reliant (cold) to CF-dominant (warm) without a hard threshold.
- **Deterministic boundary**: `w(0) = 1` guarantees zero-interaction items get pure semantic score — CF is structurally excluded, not just down-weighted.
- **Semantic channel abstraction**: `s_sem ∈ {s_text, s_image}` — the gate rule is modality-agnostic. Both text and image are instances of the same semantic channel.
- **No images at runtime**: All features precomputed offline.
- **Coldness-stratified evaluation**: Metrics reported per interaction-count bucket (L0–L3) to prevent aggregate metrics from masking cold-tail failure.

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
