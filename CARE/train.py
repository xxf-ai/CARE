"""
train.py — VARA training entry point

Simplified training loop:
  - CF backbone (BPR-MF) + text-semantic adapter
  - Joint optimization: BPR loss + text-CF alignment loss
  - All features precomputed (clip_text.npy cached offline)
  - Zero VLM at runtime
  - Logs saved to logs/{dataset}/train_seed{seed}_{timestamp}.log
"""

import argparse
import json
import time
import sys
import datetime
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torch.utils.data import DataLoader
from torch.amp import autocast
from torch.cuda.amp import GradScaler

from models.vara          import VARA
from data_utils.dataset   import MARADataset, CacheManager
from evaluation.evaluator import Evaluator
from losses.vara_loss     import vara_total_loss

ROOT_DIR  = Path(__file__).resolve().parent
DATA_DIR  = ROOT_DIR / "data"
CACHE_DIR = ROOT_DIR / "cache"
CKPT_DIR  = ROOT_DIR / "checkpoints"
LOG_DIR   = ROOT_DIR / "logs"


class Logger:
    """同时输出到 stdout 和日志文件"""
    def __init__(self, log_path: Path):
        self.file = open(log_path, "w", encoding="utf-8")
        self.terminal = sys.stdout

    def write(self, message):
        self.terminal.write(message)
        self.file.write(message)
        self.file.flush()

    def flush(self):
        self.terminal.flush()
        self.file.flush()

    def close(self):
        self.file.close()

DEFAULT_CFG = {
    "d": 64,
    "lr": 1e-3, "weight_decay": 1e-5,
    "margin": 0.1, "tau_score": 1.0,
    "max_epochs": 50, "patience": 6, "warmup_ratio": 0.10,
    "batch_size": 16384, "num_workers": 12, "eval_batch": 16384, "neg_batch": 64,
    "k_list": [5, 10, 20], "n_neg": 99,
    "margin_warmup_start": 5, "margin_warmup_end": 30,
    "val_every": 3,
    "use_amp": True, "compile": True,
}


def set_seed(seed):
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_cache_to_gpu(cache_mgr, device):
    cache = {}
    gpu_gb = 0.0
    for k, v in cache_mgr.tensors.items():
        if isinstance(v, np.ndarray):
            t = torch.from_numpy(np.array(v, dtype=np.float32)).to(device, non_blocking=True)
        else:
            t = v.to(device, non_blocking=True)
        cache[k] = t
        gb = t.element_size() * t.nelement() / 1e9
        gpu_gb += gb
        print(f"    {k:<22} {tuple(t.shape)}  {gb:.2f}GB  [GPU]")
    print(f"    {'GPU合计':<22} {gpu_gb:.2f}GB")
    return cache


def _get_margin(epoch, cfg):
    s, e, t = cfg["margin_warmup_start"], cfg["margin_warmup_end"], cfg["margin"]
    if epoch <= s: return 0.0
    if epoch >= e: return t
    return t * (epoch - s) / (e - s)


def train(args, cfg=None):
    set_seed(args.seed)
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True  # 自动选择最优 kernel
    data_dir  = DATA_DIR  / args.dataset
    cache_dir = CACHE_DIR / args.dataset
    ckpt_dir  = CKPT_DIR  / args.dataset / f"seed{args.seed}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── 日志文件 ──────────────────────────────────────────────────
    log_dir = LOG_DIR / args.dataset
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"train_seed{args.seed}_{ts}.log"
    logger = Logger(log_path)
    sys.stdout = logger
    print(f"  日志文件: {log_path}")

    run_cfg = dict(DEFAULT_CFG)
    if cfg: run_cfg.update(cfg)

    # ── Auto-scale batch_size based on dataset size ──────────────
    # 目标: 每个 epoch 约 40-80 个 step（大数据集用大 batch）
    with open(data_dir / "stats.json") as f:
        pre_stats = json.load(f)
    n_train_approx = pre_stats.get("n_train", 100000)
    auto_bs = run_cfg["batch_size"]
    if n_train_approx > 1_000_000:
        auto_bs = 65536   # sports (2.8M) → ~43 steps/epoch
    elif n_train_approx > 500_000:
        auto_bs = 32768   # office (1.4M) → ~43 steps/epoch
    run_cfg["batch_size"] = auto_bs

    for attr, key in [("d", "d"), ("batch_size", "batch_size"),
                      ("epochs", "max_epochs"), ("lr", "lr"), ("margin", "margin")]:
        val = getattr(args, attr, None)
        if val is not None: run_cfg[key] = val

    print(f"\n{'='*60}")
    print(f"  VARA 训练  数据集={args.dataset}  seed={args.seed}")
    print(f"  d={run_cfg['d']}  batch={run_cfg['batch_size']}  lr={run_cfg['lr']}")
    print(f"  margin={run_cfg['margin']}  τ={run_cfg['tau_score']}")
    print(f"{'='*60}")

    # ── Training data: 一次性上 GPU，消除 CPU 瓶颈 ────────────────
    import pandas as pd
    train_df = pd.read_csv(data_dir / "train_indexed.csv")
    train_users_gpu = torch.from_numpy(
        train_df["user_id"].values.astype(np.int64)).to(device)
    train_items_gpu = torch.from_numpy(
        train_df["item_id"].values.astype(np.int64)).to(device)
    n_train = len(train_users_gpu)

    # ── Validation data ───────────────────────────────────────────
    val_set    = MARADataset(data_dir, split="val", seed=args.seed, n_neg=run_cfg["n_neg"])
    val_loader = DataLoader(val_set, batch_size=run_cfg["eval_batch"],
                            shuffle=False, num_workers=4,
                            pin_memory=True, persistent_workers=True,
                            prefetch_factor=2)

    # ── Cache ─────────────────────────────────────────────────────
    print("\n  加载特征缓存...")
    cache_mgr = CacheManager(cache_dir, data_dir, device, pin_memory=False)
    cache = load_cache_to_gpu(cache_mgr, device)

    # ── Stats ─────────────────────────────────────────────────────
    with open(data_dir / "stats.json") as f:
        stats = json.load(f)
    n_users, n_items = stats["n_users"], stats["n_items"]

    # ── Model ─────────────────────────────────────────────────────
    model = VARA(n_users, n_items, run_cfg).to(device)
    n_params = model.count_params()["trainable"]
    print(f"\n  可训练参数: {n_params/1e6:.2f}M")

    if run_cfg.get("compile", True) and hasattr(torch, "compile"):
        print("  torch.compile 编译中...")
        model = torch.compile(model, mode="reduce-overhead")
        print("  编译完成")

    use_amp = run_cfg.get("use_amp", True) and device.type == "cuda"
    scaler  = GradScaler(enabled=use_amp)
    print(f"  AMP={'开启' if use_amp else '关闭'}")

    # ── Optimizer ─────────────────────────────────────────────────
    optimizer = optim.AdamW(model.parameters(), lr=run_cfg["lr"],
                            weight_decay=run_cfg["weight_decay"])

    steps_per_epoch = (n_train + run_cfg["batch_size"] - 1) // run_cfg["batch_size"]
    total_steps     = steps_per_epoch * run_cfg["max_epochs"]
    warmup_steps    = max(1, int(total_steps * run_cfg["warmup_ratio"]))
    scheduler = SequentialLR(optimizer, schedulers=[
        LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps),
        CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps,
                          eta_min=run_cfg["lr"] * 0.01),
    ], milestones=[warmup_steps])

    # ── Evaluator ─────────────────────────────────────────────────
    evaluator = Evaluator(model, cache, run_cfg["k_list"], device,
                          run_cfg["eval_batch"], run_cfg["neg_batch"])

    best_ndcg10, best_epoch, patience_cnt = 0.0, 0, 0
    warmup_epoch_end = max(1, int(run_cfg["max_epochs"] * run_cfg["warmup_ratio"]))

    print(f"\n  开始训练（最多 {run_cfg['max_epochs']} epoch，patience={run_cfg['patience']}）\n")

    for epoch in range(1, run_cfg["max_epochs"] + 1):
        model.train()
        ep_loss = ep_bpr = ep_gap = 0.0
        t0 = time.time()
        grad_clip = 0.5 if epoch <= warmup_epoch_end else 1.0
        current_margin = _get_margin(epoch, run_cfg)

        # ── GPU 侧 shuffle + batch + 负采样（零 CPU 开销）───────
        perm = torch.randperm(n_train, device=device)
        epoch_u = train_users_gpu[perm]
        epoch_i = train_items_gpu[perm]

        for start in range(0, n_train, run_cfg["batch_size"]):
            end       = min(start + run_cfg["batch_size"], n_train)
            users     = epoch_u[start:end]
            pos_items = epoch_i[start:end]
            neg_items = torch.randint(0, n_items, (users.size(0),), device=device)

            with autocast("cuda", enabled=use_amp):
                outputs = model(users, pos_items, neg_items, cache)
                loss, info = vara_total_loss(
                    outputs,
                    margin=current_margin,
                )

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            ep_loss += info["loss_total"]
            ep_bpr  += info["loss_bpr"]
            with torch.no_grad():
                ep_gap += (outputs["score_cf_pos"] - outputs["score_cf_neg"]).mean().item()

        nb = (n_train + run_cfg["batch_size"] - 1) // run_cfg["batch_size"]
        elapsed = time.time() - t0

        # ── 打印训练指标（每轮）────────────────────────────────
        print(f"  Epoch {epoch:3d}  loss={ep_loss/nb:.4f}  "
              f"bpr={ep_bpr/nb:.4f}  gap={ep_gap/nb:.3f}  "
              f"{elapsed:.0f}s  lr={optimizer.param_groups[0]['lr']:.2e}", end="")

        # ── 评估（每 val_every 轮，epoch≤5 时每轮都评）───────
        val_every = run_cfg.get("val_every", 3)
        do_eval = (epoch % val_every == 0) or (epoch <= 5)

        if do_eval:
            val_m  = evaluator.evaluate(val_loader, desc=f"Ep{epoch}")
            ndcg10 = val_m.get("ndcg@10", 0.0)
            hr10   = val_m.get("hr@10",   0.0)
            print(f"  NDCG@10={ndcg10:.4f}  HR@10={hr10:.4f}")

            if ndcg10 > best_ndcg10:
                best_ndcg10, best_epoch, patience_cnt = ndcg10, epoch, 0
                torch.save(
                    {"epoch": epoch, "model": model.state_dict(),
                     "val_ndcg10": ndcg10, "cfg": run_cfg},
                    ckpt_dir / "best_model.pt",
                )
            else:
                patience_cnt += 1
                if patience_cnt >= run_cfg["patience"]:
                    print(f"\n  早停：{run_cfg['patience']} epoch 无提升")
                    break
        else:
            # 跳过评估时用上次 best 更新 patience
            print()  # 换行
            patience_cnt += 1

        if getattr(args, "smoke_test", False) and epoch >= 2:
            print("  Smoke test 完成")
            break

    peak = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    print(f"\n  训练完成  最佳 NDCG@10={best_ndcg10:.4f} @ Epoch {best_epoch}"
          f"  显存峰值={peak:.1f}GB")

    with open(ckpt_dir / "train_summary.json", "w") as f:
        json.dump({"dataset": args.dataset, "seed": args.seed,
                   "best_ndcg10": best_ndcg10, "best_epoch": best_epoch,
                   "params": n_params, "cfg": run_cfg}, f, indent=2)

    # ── 关闭日志文件，恢复 stdout ─────────────────────────────────
    sys.stdout = logger.terminal
    logger.close()
    print(f"  日志已保存: {log_path}")

    return best_ndcg10


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",    default="baby", choices=["baby", "beauty_sub", "office", "sports", "yelp"])
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--seeds",      type=int,   nargs="+", default=None)
    parser.add_argument("--d",          type=int,   default=None)
    parser.add_argument("--batch_size", type=int,   default=None)
    parser.add_argument("--epochs",     type=int,   default=None)
    parser.add_argument("--lr",         type=float, default=None)
    parser.add_argument("--margin",     type=float, default=None)
    parser.add_argument("--smoke_test", action="store_true")
    args = parser.parse_args()

    seeds   = args.seeds or [args.seed]
    results = {}
    for seed in seeds:
        args.seed = seed
        results[seed] = train(args)

    if len(seeds) > 1:
        vals = list(results.values())
        print(f"\n  多种子汇总  均值={np.mean(vals):.4f}  std={np.std(vals):.4f}")


if __name__ == "__main__":
    main()
