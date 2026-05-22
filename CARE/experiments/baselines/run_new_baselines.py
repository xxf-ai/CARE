"""
experiments/baselines/run_new_baselines.py — 新通用多模态基线模型训练 + 分层评估

支持 3 个 General multimodal CF 基线模型:
  LGMRec   (AAAI 2024)  — Local/Global Graph Learning for Multimodal Recommendation
  PEARL    (2025)       — Dual-layer Graph Learning for Multimodal Recommendation
  PromptMM (WWW 2024)   — Multi-Modal Knowledge Distillation with Prompt-Tuning

冷启动专用模型 (CCFCRec, MARec) → run_coldstart_baselines.py

用法:
  python experiments/baselines/run_new_baselines.py --dataset baby --seeds 42 123 456
  python experiments/baselines/run_new_baselines.py --dataset baby --models LGMRec --seeds 42
  python experiments/baselines/run_new_baselines.py --dataset baby --seeds 42 --skip_train
"""

import sys, json, argparse, time, datetime
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import autocast
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from data_utils.dataset import MARADataset, CacheManager
from evaluation.evaluator import Evaluator
from evaluate import build_user_text_pref
from experiments.baselines.models.graph_models import LGMRec, PEARL
from experiments.baselines.models.prompt_models import PromptMM

DATA_DIR   = ROOT / "data"
CACHE_DIR  = ROOT / "cache"
CKPT_DIR   = ROOT / "checkpoints"
RESULT_DIR = ROOT / "experiments" / "baselines" / "results"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

COLD_THRESHOLD = 5
K_LIST = [5, 10, 20]
BASE_CFG = {
    "d": 64, "text_dim": 768,
    "lr": 1e-3, "weight_decay": 1e-5,
    "max_epochs": 30, "patience": 6,
    "batch_size": 16384, "eval_batch": 16384, "neg_batch": 64,
    "k_list": [5, 10, 20], "n_neg": 99,
    "val_every": 3,
    "use_amp": True, "compile": True,
}

MODEL_CLASSES = {
    "LGMRec": LGMRec, "PEARL": PEARL, "PromptMM": PromptMM,
}
CKPT_SUBDIRS = {
    "LGMRec": "lgmrec", "PEARL": "pearl", "PromptMM": "promptmm",
}

args_dataset = None


def set_seed(seed):
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _save_ckpt(model, optimizer, epoch, best_metric, ckpt_path):
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": epoch, "model": model.state_dict(),
        "optimizer": optimizer.state_dict(), "best_metric": best_metric,
    }, ckpt_path)


def load_cache_gpu(cache_dir, data_dir, device):
    mgr = CacheManager(cache_dir, data_dir, device, pin_memory=False)
    cache = {}
    for k, v in mgr.tensors.items():
        if isinstance(v, np.ndarray):
            cache[k] = torch.from_numpy(np.array(v, dtype=np.float32)).to(device)
        else:
            cache[k] = v.to(device)
    return cache


# ══════════════════════════════════════════════════════════════════════════════
# 统一训练循环 (AMP + torch.compile + pin_memory, 对齐 train.py 风格)
# ══════════════════════════════════════════════════════════════════════════════

def run_training(model, device, data_dir, seed, cfg, model_name, clip_text):
    """统一训练循环 (AMP + torch.compile + pin_memory)。"""
    train_df = pd.read_csv(data_dir / "train_indexed.csv")
    train_u = torch.tensor(train_df["user_id"].values, dtype=torch.long, device=device)
    train_i = torch.tensor(train_df["item_id"].values, dtype=torch.long, device=device)
    n_train = len(train_u)
    n_items = clip_text.size(0)

    val_set = MARADataset(data_dir, split="val", seed=seed, n_neg=cfg["n_neg"])
    val_loader = DataLoader(val_set, batch_size=cfg["eval_batch"],
                            shuffle=False, num_workers=4,
                            pin_memory=True, persistent_workers=True)
    evaluator = Evaluator(model, {}, cfg["k_list"], device,
                          cfg["eval_batch"], cfg["neg_batch"])

    # torch.compile
    use_compile = cfg.get("compile", True) and hasattr(torch, "compile")
    if use_compile:
        print("  torch.compile 编译中 (mode=reduce-overhead)...")
        model = torch.compile(model, mode="reduce-overhead")

    # AMP
    use_amp = cfg.get("use_amp", True) and device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)
    print(f"  AMP={'FP16' if use_amp else 'FP32'}  compile={'ON' if use_compile else 'OFF'}")

    optimizer = optim.AdamW(model.parameters(), lr=cfg["lr"],
                            weight_decay=cfg["weight_decay"])
    ckpt_dir = CKPT_DIR / args_dataset / CKPT_SUBDIRS[model_name] / f"seed{seed}"
    best_ndcg, patience = 0.0, 0

    for epoch in range(1, cfg["max_epochs"] + 1):
        model.train()
        perm = torch.randperm(n_train, device=device)
        eu, ei = train_u[perm], train_i[perm]
        ep_loss = 0.0; nb = 0

        for start in range(0, n_train, cfg["batch_size"]):
            end = min(start + cfg["batch_size"], n_train)
            users, pos = eu[start:end], ei[start:end]
            neg = torch.randint(0, n_items, (users.size(0),), device=device)

            with autocast("cuda", enabled=use_amp):
                loss = _forward_loss(model, model_name, users, pos, neg,
                                     clip_text)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            ep_loss += loss.item(); nb += 1

        if epoch % cfg["val_every"] == 0 or epoch <= 5:
            val_m = evaluator.evaluate(val_loader, desc=f"{model_name} ep{epoch}")
            ndcg = val_m.get("ndcg@10", 0)
            print(f"    ep{epoch:2d}  loss={ep_loss/max(nb,1):.4f}  NDCG@10={ndcg:.4f}")
            if ndcg > best_ndcg:
                best_ndcg, patience = ndcg, 0
                _save_ckpt(model, optimizer, epoch, best_ndcg, ckpt_dir / "best_model.pt")
            else:
                patience += 1
                if patience >= cfg["patience"]:
                    print(f"    早停 @ epoch {epoch}")
                    break

    if not (ckpt_dir / "best_model.pt").exists():
        _save_ckpt(model, optimizer, epoch, best_ndcg, ckpt_dir / "best_model.pt")
    return ckpt_dir / "best_model.pt", best_ndcg


def _forward_loss(model, model_name, users, pos, neg, clip_text):
    """根据模型名自动选择 forward 参数。"""
    if model_name == "LGMRec":
        return model.forward_full(users, pos, neg, clip_text)
    else:  # PEARL, PromptMM — standard BPR forward
        return model(users, pos, neg, clip_text[pos], clip_text[neg])


# ══════════════════════════════════════════════════════════════════════════════
# 分层评估
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def stratified_eval_new(model, model_name, cache, item_counts, n_items,
                        device, seed, data_dir, alpha=1.0):
    """对单个模型运行 L0/L1/L2/L3 分层评估 (标准协议)。"""
    tau = 1.0
    buckets = [
        ("L0 (zero-shot)", 0, 0),
        ("L1 (1-4)",       1, 4),
        ("L2 (5-20)",      5, 20),
        ("L3 (>20)",      21, 1_000_000),
    ]

    test_set = MARADataset(data_dir, split="test", seed=seed, n_neg=99)
    loader = DataLoader(test_set, batch_size=4096, shuffle=False,
                        num_workers=2, pin_memory=True)
    strategies = ["cf_only", "text_only", "gate"]
    bucket_ranks = {bname: {s: [] for s in strategies} for bname, _, _ in buckets}

    base_model = getattr(model, "_orig_mod", model)
    can_extract_cf = (hasattr(base_model, 'get_user_cf_repr') and
                      hasattr(base_model, 'get_all_item_cf_repr'))

    clip_text = cache["clip_text"]
    user_text_pref = cache["user_text_pref"]
    if can_extract_cf:
        all_item_cf = base_model.get_all_item_cf_repr()

    for users, pos_items, neg_items in loader:
        users     = users.to(device)
        pos_items = pos_items.to(device)
        neg_items = neg_items.to(device)
        B = users.size(0)
        cand = torch.cat([pos_items.unsqueeze(1), neg_items], dim=1)
        cand_flat = cand.reshape(-1)

        # Text scores
        u_txt = F.normalize(user_text_pref[users], dim=-1)
        i_txt = F.normalize(clip_text[cand_flat], dim=-1).reshape(B, 100, -1)
        txt_scores = (u_txt.unsqueeze(1) * i_txt).sum(-1) / tau
        txt_ranks = ((txt_scores > txt_scores[:, 0:1]).sum(dim=1) + 1).cpu().numpy()

        cf_ranks = gate_ranks = None
        if can_extract_cf:
            u_cf = base_model.get_user_cf_repr(users)
            i_cf = all_item_cf[cand_flat].reshape(B, 100, -1)
            cf_scores = (u_cf.unsqueeze(1) * i_cf).sum(-1) / tau
            cf_ranks = ((cf_scores > cf_scores[:, 0:1]).sum(dim=1) + 1).cpu().numpy()

            cand_counts = item_counts[cand]
            w_txt = 1.0 / (1.0 + alpha * cand_counts)
            gate_scores = (1.0 - w_txt) * cf_scores + w_txt * txt_scores
            gate_ranks = ((gate_scores > gate_scores[:, 0:1]).sum(dim=1) + 1).cpu().numpy()

        pos_counts = item_counts[pos_items].cpu().numpy()
        for j in range(B):
            cnt = int(pos_counts[j])
            for bname, lo, hi in buckets:
                if lo <= cnt <= hi:
                    bucket_ranks[bname]["text_only"].append(int(txt_ranks[j]))
                    if can_extract_cf:
                        bucket_ranks[bname]["cf_only"].append(int(cf_ranks[j]))
                        bucket_ranks[bname]["gate"].append(int(gate_ranks[j]))
                    break

    result = {}
    for bname, _, _ in buckets:
        bres = {}
        for strategy in strategies:
            ranks_arr = np.array(bucket_ranks[bname][strategy])
            if len(ranks_arr) == 0:
                bres[strategy] = None
                continue
            m = {"n": len(ranks_arr)}
            for k in K_LIST:
                m[f"hr@{k}"]   = float((ranks_arr <= k).mean())
                m[f"ndcg@{k}"] = float(
                    np.where(ranks_arr <= k, 1.0 / np.log2(ranks_arr + 1), 0.0).mean())
            bres[strategy] = m
        result[bname] = bres
    return result


def _print_stratified(model_name, seed, result, alpha):
    print(f"\n  Stratified (α={alpha}):")
    print(f"  {'Bucket':<18} {'n':>6}  {'CF N@10':>8}  {'TXT N@10':>8}  {'GATE N@10':>8}")
    print(f"  {'─'*58}")
    for bname in ["L0 (zero-shot)", "L1 (1-4)", "L2 (5-20)", "L3 (>20)"]:
        b = result.get(bname, {})
        cf  = b.get("cf_only", {})
        txt = b.get("text_only", {})
        gt  = b.get("gate", {})
        n = cf.get("n", 0) if cf else 0
        cfn = f"{cf.get('ndcg@10', 0):.4f}" if cf else "N/A"
        tn  = f"{txt.get('ndcg@10', 0):.4f}" if txt else "N/A"
        gn  = f"{gt.get('ndcg@10', 0):.4f}" if gt else "N/A"
        print(f"  {bname:<18} {n:>6}  {cfn:>8}  {tn:>8}  {gn:>8}")


def _print_multi_seed_summary(all_results):
    print(f"\n{'='*80}")
    print(f"  多 Seed 汇总 — NDCG@10 mean ± std")
    print(f"{'='*80}")
    buckets_order = ["L0 (zero-shot)", "L1 (1-4)", "L2 (5-20)", "L3 (>20)"]
    strategies = ["cf_only", "text_only", "gate"]

    for model_name, seeds_dict in all_results.items():
        print(f"\n  [{model_name}]")
        header = f"  {'Bucket':<18}"
        for s in strategies:
            header += f"  {s:>18}"
        print(header)
        print(f"  {'─'*76}")
        for bname in buckets_order:
            row = f"  {bname:<18}"
            for strategy in strategies:
                vals = []
                for seed, data in seeds_dict.items():
                    b = data["result"].get(bname, {})
                    m = b.get(strategy, {}) if b else {}
                    if m and m.get("ndcg@10") is not None:
                        vals.append(m["ndcg@10"])
                if len(vals) >= 2:
                    row += f"  {np.mean(vals):.4f}±{np.std(vals):.4f}"
                elif len(vals) == 1:
                    row += f"  {vals[0]:.4f}"
                else:
                    row += f"  {'N/A':>18}"
            print(row)
    print(f"{'='*80}")


# ══════════════════════════════════════════════════════════════════════════════
# 主入口
# ══════════════════════════════════════════════════════════════════════════════

def main():
    global args_dataset
    parser = argparse.ArgumentParser(description="新基线模型训练 + 分层评估")
    parser.add_argument("--dataset", default="baby",
                        choices=["baby", "beauty_sub", "office", "sports", "yelp"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    parser.add_argument("--models", nargs="+",
                        default=["LGMRec", "PEARL", "PromptMM"])
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_eval", action="store_true")
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    args = parser.parse_args()

    args_dataset = args.dataset
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir  = DATA_DIR / args.dataset
    cache_dir = CACHE_DIR / args.dataset

    cfg = dict(BASE_CFG)
    if args.batch_size:
        cfg["batch_size"] = args.batch_size
    if args.epochs:
        cfg["max_epochs"] = args.epochs
    if args.lr:
        cfg["lr"] = args.lr

    with open(data_dir / "stats.json") as f:
        stats = json.load(f)
    n_users, n_items = stats["n_users"], stats["n_items"]

    if args.alpha is not None:
        alpha = args.alpha
    else:
        try:
            vr = json.load(open(ROOT / "results" / f"{args.dataset}_vara_results.json"))
            alpha = vr["per_seed"][0].get("gate_best_alpha", 1.0)
        except Exception:
            alpha = 1.0 if args.dataset in ("baby", "yelp") else 0.5

    # 限制 batch_size 防止 OOM
    if cfg["batch_size"] > n_items:
        cfg["batch_size"] = max(4096, n_items // 2)

    print(f"\n{'='*70}")
    print(f"  新基线模型训练 + 分层评估  dataset={args.dataset}  seeds={args.seeds}")
    print(f"  models={args.models}  α={alpha}  batch_size={cfg['batch_size']}")
    print(f"{'='*70}")

    # ── 加载 cache ──────────────────────────────────────────────────
    cache = {}
    if not args.skip_train or not args.skip_eval:
        print("\n  加载特征缓存...")
        cache = load_cache_gpu(cache_dir, data_dir, device)

    # ── 训练 ─────────────────────────────────────────────────────────
    train_results = {}
    if not args.skip_train:
        for model_name in args.models:
            if model_name not in MODEL_CLASSES:
                print(f"\n  [SKIP] 未知模型: {model_name}")
                continue

            print(f"\n{'─'*50}")
            print(f"  Train: {model_name}")
            print(f"{'─'*50}")
            train_results[model_name] = {}

            for seed in args.seeds:
                set_seed(seed)
                t0 = time.time()
                print(f"  seed={seed} ...")

                ModelClass = MODEL_CLASSES[model_name]
                model = ModelClass(n_users, n_items, text_dim=768, d=cfg["d"]).to(device)

                # Graph 模型需要构建图
                if model_name in ("LGMRec", "PEARL"):
                    tdf = pd.read_csv(data_dir / "train_indexed.csv")
                    edge_u = torch.tensor(tdf["user_id"].values, dtype=torch.long, device=device)
                    edge_i = torch.tensor(tdf["item_id"].values, dtype=torch.long, device=device)
                    edge_index = torch.stack([edge_u, edge_i], dim=0)
                    if model_name == "LGMRec":
                        model.set_graph(edge_index, n_users, n_items)
                    elif model_name == "PEARL":
                        model.set_graphs(edge_index, cache["clip_text"].to(device), n_users, n_items)

                ckpt_path, best_ndcg = run_training(
                    model, device, data_dir, seed, cfg, model_name,
                    cache["clip_text"].to(device))

                elapsed = time.time() - t0
                print(f"  seed={seed}: NDCG@10={best_ndcg:.4f}  ({elapsed:.0f}s)")
                train_results[model_name][seed] = {
                    "ckpt": str(ckpt_path) if ckpt_path else None,
                    "best_val_ndcg10": best_ndcg,
                }

    # ── 分层评估 ─────────────────────────────────────────────────────
    if not args.skip_eval:
        print(f"\n{'='*70}")
        print(f"  分层评估")
        print(f"{'='*70}")

        cache["user_text_pref"] = build_user_text_pref(
            data_dir / "train_indexed.csv", cache["clip_text"], n_users, device)
        train_df = pd.read_csv(data_dir / "train_indexed.csv")
        item_counts = torch.zeros(n_items, device=device)
        for iid in train_df["item_id"].values:
            item_counts[int(iid)] += 1

        all_stratified = {}

        for model_name in args.models:
            if model_name not in CKPT_SUBDIRS:
                continue

            print(f"\n  [{model_name}]")
            subdir = CKPT_SUBDIRS[model_name]

            strat_results = {}
            for seed in args.seeds:
                ckpt_path = CKPT_DIR / args.dataset / subdir / f"seed{seed}" / "best_model.pt"
                if not ckpt_path.exists():
                    print(f"  seed={seed}: checkpoint 不存在 {ckpt_path}")
                    continue

                print(f"  seed={seed}: {ckpt_path}")

                ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
                ModelClass = MODEL_CLASSES[model_name]
                model = ModelClass(n_users, n_items, text_dim=768, d=cfg["d"]).to(device)

                # Graph 模型加载后需要重建图
                if model_name == "LGMRec":
                    edge_u = torch.tensor(train_df["user_id"].values, dtype=torch.long, device=device)
                    edge_i = torch.tensor(train_df["item_id"].values, dtype=torch.long, device=device)
                    model.set_graph(torch.stack([edge_u, edge_i], dim=0), n_users, n_items)
                elif model_name == "PEARL":
                    edge_u = torch.tensor(train_df["user_id"].values, dtype=torch.long, device=device)
                    edge_i = torch.tensor(train_df["item_id"].values, dtype=torch.long, device=device)
                    model.set_graphs(torch.stack([edge_u, edge_i], dim=0),
                                     cache["clip_text"].to(device), n_users, n_items)

                state_dict = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model"].items()}
                model.load_state_dict(state_dict, strict=False)
                model.eval()

                result = stratified_eval_new(
                    model, model_name, cache, item_counts, n_items,
                    device, seed, data_dir, alpha=alpha)
                if result is not None:
                    strat_results[seed] = {
                        "result": result, "alpha": alpha, "ckpt": str(ckpt_path)}
                    _print_stratified(model_name, seed, result, alpha)

            if strat_results:
                all_stratified[model_name] = strat_results

        if all_stratified:
            _print_multi_seed_summary(all_stratified)

            out_path = RESULT_DIR / f"{args.dataset}_new_multimodal_baselines.json"

            # 读取已有结果并在 seed 级别合并
            existing = {}
            if out_path.exists():
                try:
                    with open(out_path) as f:
                        prev = json.load(f)
                    existing = prev.get("results", {})
                except Exception:
                    pass

            merged_results = dict(existing)
            for mname, seeds_dict in all_stratified.items():
                if mname not in merged_results:
                    merged_results[mname] = {}
                for seed_str, data in seeds_dict.items():
                    merged_results[mname][seed_str] = data

            all_models = sorted(merged_results.keys())

            with open(out_path, "w") as f:
                json.dump({
                    "dataset": args.dataset,
                    "models": all_models,
                    "timestamp": datetime.datetime.now().isoformat(),
                    "results": merged_results,
                }, f, indent=2, ensure_ascii=False)
            print(f"\n  结果已保存: {out_path}")

    print(f"\n{'='*70}")
    print(f"  完成")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
