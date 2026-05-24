"""
experiments/analysis/gate_visualize.py — CARE gate activation analysis

Visualizes:
  1. Uncertainty gate activation rate by item popularity
  2. Learned fusion weights (α, β) over training
"""

import sys, argparse, json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

from models.care import CARE

RESULTS_DIR = ROOT / "experiments" / "analysis" / "figures"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

try:
    rcParams["font.family"] = "SimSun"
except:
    pass


def load_model(dataset, seed, device):
    ckpt_path = ROOT / "checkpoints" / dataset / f"seed{seed}" / "best_model.pt"
    if not ckpt_path.exists():
        return None, None

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt.get("cfg", {})

    data_dir = ROOT / "data" / dataset
    with open(data_dir / "stats.json") as f:
        stats = json.load(f)

    model = CARE(stats["n_users"], stats["n_items"], cfg).to(device)
    state_dict = ckpt["model"]
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model, stats


def plot_fusion_weights(model, save_path):
    """Plot learned α/β fusion weights"""
    alpha = torch.sigmoid(model.log_alpha).item()
    beta  = torch.sigmoid(model.log_beta).item()
    w_sum = alpha + beta + 1e-8

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(["α (CF weight)", "β (Text weight)"],
                  [alpha / w_sum, beta / w_sum],
                  color=["#4C72B0", "#55A868"])
    for bar, v in zip(bars, [alpha / w_sum, beta / w_sum]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{v:.3f}", ha="center", fontsize=12)
    ax.set_ylabel("Normalized Weight", fontsize=12)
    ax.set_title(f"CARE Fusion Weights (α={alpha:.3f}, β={beta:.3f})", fontsize=13)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Fusion weights plot saved: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="baby")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, stats = load_model(args.dataset, args.seed, device)

    if model is None:
        print(f"No checkpoint found for {args.dataset}/seed{args.seed}")
        return

    plot_fusion_weights(
        model,
        RESULTS_DIR / f"{args.dataset}_seed{args.seed}_fusion_weights.png",
    )

    print(f"\n  Learned parameters:")
    print(f"    α (CF)  = {torch.sigmoid(model.log_alpha).item():.4f}")
    print(f"    β (Txt) = {torch.sigmoid(model.log_beta).item():.4f}")
    print(f"    δ (gate threshold) = {model.delta:.4f}")


if __name__ == "__main__":
    main()
