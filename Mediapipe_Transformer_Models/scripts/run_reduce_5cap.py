"""
rq2_runner_v2.py  —  RQ2: Progressive Training Data Reduction with Rotating Windows
=====================================================================================
Analyzes the relationship between training data size and signer-independent
recognition performance under the LOSO protocol.

KEY DIFFERENCE FROM v1:
    For each cap value, training is repeated ROUNDS_PER_CAP=3 times, each time
    using a DIFFERENT (non-overlapping) window of videos. This ensures we test
    data efficiency with genuinely different video selections, not just the same
    videos evaluated once.

Window-based subsampling strategy:
    For each (cap, rotation_round) pair, a deterministic window offset is used
    to select which videos to draw from per user per word.

    Example for cap=3, 3 rotation rounds:
        Rotation 0: try to take videos [0:3]   (positions 0,1,2)
        Rotation 1: try to take videos [3:6]   (positions 3,4,5)
        Rotation 2: try to take videos [6:9]   (positions 6,7,8)

    If the user has fewer videos than needed for a window, videos wrap around
    (cyclic indexing) so we never run out — but the code logs when wrapping
    happens for transparency.

    The sorted order of videos within a user/word bucket is fixed. The window
    start = rotation_round * cap (mod total_available if wrapping needed).

Structure:
    caps          = [5, 4, 3, 2, 1]          (5 cap values)
    rotation_rounds = [0, 1, 2]              (3 rotations per cap)
    loso_folds    = 15                        (one per signer)
    ─────────────────────────────────────────
    Total runs    = 5 × 3 × 15 = 225

Reproducibility:
    Video ordering within each (user, word) bucket is sorted by filename, so
    the same files always map to the same positions regardless of OS/filesystem.
    Seed for any within-window shuffling = base_seed + cap * 10000 + rotation_round * 1000 + fold_idx.

Test set:
    Fixed across ALL caps and rotations — held-out user's complete recordings.
    Never subsampled or rotated.

Outputs (saved to RESULTS_DIR):
    cap_5/
        rotation_0/
            fold_0/ ... fold_14/
                final_model.pt
                norm_mean.npy, norm_std.npy
                eval/
                    confusion_matrix.png
                    top_confused_pairs.png/.csv
                    per_class_accuracy.csv/_bar.png
                    classification_report.txt
                    metrics.txt
            rotation_summary.csv     ← 15-fold summary for this (cap, rotation)
        rotation_1/  ...
        rotation_2/  ...
        cap_summary.csv              ← averaged over all 3 rotations for this cap
    cap_4/ ... cap_1/
    rq2_summary.csv                  ← one row per (cap, rotation)
    rq2_cap_averaged_summary.csv     ← one row per cap (averaged over rotations)
    rq2_accuracy_curve.png           ← main curve (cap-averaged)
    rq2_per_rotation_curves.png      ← one line per rotation
    rq2_boxplot.png                  ← distribution across folds+rotations per cap
    rq2_heatmap.png                  ← cap × rotation heatmap of top-1
    rq2_fold_variance.png            ← per-fold consistency across caps
    rq2_window_coverage.csv          ← logs wrapping events
    class_names.npy
    config.json
    rq2_log_TIMESTAMP.txt

Run:
    python rq2_runner_v2.py
    python rq2_runner_v2.py --cap_start 5 --cap_end 1 --rotations 3 --results_dir results_rq2_v2
"""

import os
import sys
import csv
import json
import random
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

from datetime import datetime
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    confusion_matrix, f1_score,
    precision_score, recall_score,
)
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.dataset_class import (
    find_h5_files, label_from_filename,
    user_from_path, H5SignDataset,
)
from src.model import TransformerClassifier, collate_fn_packed
from src.train import (
    train_one_fold, Tee,
    collect_preds, top_k_accuracy,
    save_confusion_matrix, save_top_confused_pairs,
    save_per_class_accuracy, print_worst_best_classes,
    save_classification_report,
)
# ============================================================
# CONFIG  (matches main runner.py exactly)
# ============================================================

DATA_ROOT      = "/mnt/9a528fe4-4fe8-4dff-9a0c-8b1a3cf3d7ba/popsign/ISL_Goa_Data_h5_30fps"
RESULTS_DIR    = "results_rq2_v2"

CAP_START      = 5    # maximum reps per word per user
CAP_END        = 1    # minimum reps per word per user
ROTATIONS      = 3    # how many non-overlapping video windows to try per cap

config = {
    "max_frames"      : 150,
    "batch_size"      : 32,
    "epochs"          : 100,
    "warmup_epochs"   : 5,
    "min_lr_ratio"    : 0.05,
    "lr"              : 1e-4,
    "weight_decay"    : 1e-3,
    "label_smoothing" : 0.1,
    "num_workers"     : 4,
    "use_mixup"       : True,
    "mixup_alpha"     : 0.2,
    "drop_path_rate"  : 0.1,
    "model_dim"       : 256,
    "nhead"           : 8,
    "num_layers"      : 4,
    "dim_feedforward" : 512,
    "dropout"         : 0.3,
    "seed"            : 42,
}


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ============================================================
# FILE GROUPING
# ============================================================

def group_by_user_and_word(paths):
    """
    Returns dict[user][word] -> list of h5 paths.
    Paths within each bucket are SORTED by filename for reproducibility.
    Sorting ensures position 0, 1, 2, ... always refer to the same files
    regardless of filesystem traversal order.
    """
    uwp = defaultdict(lambda: defaultdict(list))
    for p in paths:
        uwp[user_from_path(p)][label_from_filename(p)].append(p)

    # Sort each bucket by basename so positions are deterministic
    for user in uwp:
        for word in uwp[user]:
            uwp[user][word] = sorted(uwp[user][word], key=os.path.basename)

    return uwp


# ============================================================
# WINDOW-BASED SUBSAMPLING
# ============================================================

def sample_train_paths_windowed(train_users, user_word_paths, cap,
                                 rotation_round, fold_idx, base_seed):
    """
    Selects `cap` videos per (user, word) using a sliding window approach.

    Window Logic
    ─────────────
    For a given (user, word) bucket with N videos sorted by filename:

        window_start = rotation_round * cap
        window_end   = window_start + cap

    If window_end <= N  → take videos[window_start : window_end]  (no overlap)
    If window_end >  N  → wrap cyclically: indices taken modulo N
                          This ensures we always get exactly `cap` videos
                          even when the bucket is small. The wrap is logged.

    Why cyclic wrapping?
        For small cap values (e.g. cap=1) with 3 rotations, a user might only
        have 2 videos for a word. Rotation 0 takes video[0], rotation 1 takes
        video[1], rotation 2 must wrap and takes video[0] again. This is
        intentional and logged — it is the least-bad option vs. skipping the
        word entirely or crashing.

    Parameters
    ──────────
    train_users     : list[str]   — user IDs in training split
    user_word_paths : dict        — user → word → sorted list[path]
    cap             : int         — videos to select per (user, word)
    rotation_round  : int         — which rotation window (0-based)
    fold_idx        : int         — current LOSO fold (0-based)
    base_seed       : int         — base seed (used for any tie-breaking)

    Returns
    ───────
    selected     : list[str]  — flat list of selected h5 paths
    stats        : dict       — counts + wrapping diagnostics for logging
    wrap_events  : list[dict] — one entry per (user, word) that needed wrapping
    """
    selected       = []
    per_word_counts = defaultdict(int)
    wrap_events    = []

    for user in sorted(train_users):
        for word in sorted(user_word_paths[user].keys()):
            paths         = user_word_paths[user][word]
            n             = len(paths)
            window_start  = rotation_round * cap
            window_end    = window_start + cap

            if window_end <= n:
                # Clean non-overlapping window — ideal case
                chosen = paths[window_start:window_end]
                wrapped = False
            else:
                # Need to wrap cyclically
                # Build index list using modulo
                indices = [(window_start + i) % n for i in range(cap)]
                chosen  = [paths[i] for i in indices]
                wrapped = True
                wrap_events.append({
                    "user"         : user,
                    "word"         : word,
                    "n_available"  : n,
                    "cap"          : cap,
                    "rotation"     : rotation_round,
                    "window_start" : window_start,
                    "window_end"   : window_end,
                    "indices_used" : indices,
                })

            selected.extend(chosen)
            per_word_counts[word] += len(chosen)

    counts = list(per_word_counts.values())
    stats  = {
        "total_files"     : len(selected),
        "mean_per_word"   : float(np.mean(counts)),
        "min_per_word"    : int(np.min(counts)),
        "max_per_word"    : int(np.max(counts)),
        "n_wrap_events"   : len(wrap_events),
        "words_at_nominal": int(sum(
            1 for c in counts if c == cap * len(train_users)
        )),
    }
    return selected, stats, wrap_events


# ============================================================
# FOLD EVALUATION
# ============================================================

def evaluate_fold(fold_idx, fold_log_dir, test_paths,
                  label_encoder, class_names, device, config):
    """
    Load final_model.pt and run full evaluation.
    Saves all artifacts to fold_log_dir/eval/
    Returns metrics dict.
    """
    eval_dir = os.path.join(fold_log_dir, "eval")
    os.makedirs(eval_dir, exist_ok=True)

    norm_stats = {
        "mean": np.load(os.path.join(fold_log_dir, "norm_mean.npy")),
        "std" : np.load(os.path.join(fold_log_dir, "norm_std.npy")),
    }

    test_set = H5SignDataset(
        test_paths, label_encoder,
        n               = config["max_frames"],
        normalize_stats = norm_stats,
        augment         = False,
    )
    test_loader = DataLoader(
        test_set,
        batch_size  = config["batch_size"],
        shuffle     = False,
        collate_fn  = collate_fn_packed,
        num_workers = config["num_workers"],
        pin_memory  = True,
    )

    num_classes = len(class_names)
    model = TransformerClassifier(
        input_size      = H5SignDataset.FEATURE_DIM,
        num_classes     = num_classes,
        model_dim       = config["model_dim"],
        nhead           = config["nhead"],
        num_layers      = config["num_layers"],
        dim_feedforward = config["dim_feedforward"],
        dropout         = config["dropout"],
        drop_path_rate  = config.get("drop_path_rate", 0.1),
    ).to(device)

    model.load_state_dict(
        torch.load(
            os.path.join(fold_log_dir, "final_model.pt"),
            map_location=device, weights_only=True,
        )
    )
    model.eval()

    y_true, y_pred, y_topk, test_loss = collect_preds(
        model, test_loader, device, top_k=5
    )

    top1  = (y_true == y_pred).mean() * 100.0
    top5  = top_k_accuracy(y_true, y_topk)
    mac_p = precision_score(y_true, y_pred, average="macro",    zero_division=0) * 100
    mac_r = recall_score(   y_true, y_pred, average="macro",    zero_division=0) * 100
    mac_f = f1_score(       y_true, y_pred, average="macro",    zero_division=0) * 100
    wtd_f = f1_score(       y_true, y_pred, average="weighted", zero_division=0) * 100

    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    save_confusion_matrix(
        cm, class_names,
        os.path.join(eval_dir, "confusion_matrix.png"),
        title=f"Confusion Matrix — Fold {fold_idx}",
    )
    save_top_confused_pairs(
        cm, class_names,
        os.path.join(eval_dir, "top_confused_pairs.png"), n=20,
    )
    rows = save_per_class_accuracy(y_true, y_pred, class_names, eval_dir)
    print_worst_best_classes(rows, n=5)
    save_classification_report(
        y_true, y_pred, class_names,
        os.path.join(eval_dir, "classification_report.txt"),
    )

    with open(os.path.join(eval_dir, "metrics.txt"), "w") as f:
        f.write(f"Fold        : {fold_idx}\n")
        f.write(f"Top-1       : {top1:.2f}%\n")
        f.write(f"Top-5       : {top5:.2f}%\n")
        f.write(f"Macro P     : {mac_p:.2f}%\n")
        f.write(f"Macro R     : {mac_r:.2f}%\n")
        f.write(f"Macro F1    : {mac_f:.2f}%\n")
        f.write(f"Weighted F1 : {wtd_f:.2f}%\n")
        f.write(f"Test Loss   : {test_loss:.4f}\n")

    return {
        "top1_acc"   : top1,
        "top5_acc"   : top5,
        "macro_f1"   : mac_f,
        "weighted_f1": wtd_f,
        "test_loss"  : test_loss,
        "y_true"     : y_true,
        "y_pred"     : y_pred,
    }


# ============================================================
# PLOTTING — all plots generated from round_summaries
# ============================================================

def plot_main_curve(cap_summaries, save_path):
    """
    Top-1, Top-5, Macro F1 vs cap value (averaged over all 3 rotations).
    Shaded band = ±1 std of Top-1 across folds × rotations.
    """
    caps  = [s["cap"]           for s in cap_summaries]
    top1s = [s["mean_top1"]     for s in cap_summaries]
    top5s = [s["mean_top5"]     for s in cap_summaries]
    f1s   = [s["mean_macro_f1"] for s in cap_summaries]
    stds  = [s["std_top1"]      for s in cap_summaries]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(caps, top1s, "o-", color="#2563EB", lw=2.5, ms=8,
            label="Top-1 Accuracy (avg over rotations)")
    ax.fill_between(
        caps,
        [t - s for t, s in zip(top1s, stds)],
        [t + s for t, s in zip(top1s, stds)],
        color="#2563EB", alpha=0.15,
        label="Top-1 ± 1 std (folds × rotations)"
    )
    ax.plot(caps, top5s, "s-", color="#DC2626", lw=2.5, ms=8,
            label="Top-5 Accuracy")
    ax.plot(caps, f1s,   "^-", color="#16A34A", lw=2.5, ms=8,
            label="Macro F1")

    for x, y in zip(caps, top1s):
        ax.annotate(f"{y:.1f}%", (x, y),
                    textcoords="offset points", xytext=(0, 12),
                    ha="center", fontsize=9, color="#2563EB", fontweight="bold")

    ax.set_xlabel("Cap (videos per word per user)", fontsize=12)
    ax.set_ylabel("Accuracy / F1 (%)", fontsize=12)
    ax.set_title(
        "RQ2 — Effect of Training Data Size on LOSO Recognition\n"
        "(Cap-averaged: mean over 3 rotation rounds × 15 LOSO folds)",
        fontsize=13,
    )
    ax.set_xticks(caps)
    ax.set_ylim(0, 110)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=130)
    plt.close()
    print(f"  Main curve saved : {save_path}")


def plot_per_rotation_curves(all_rotation_summaries, save_path):
    """
    One line per rotation round (0,1,2), showing how each window
    of videos performs across caps. Helps diagnose if one rotation
    consistently under/over-performs.
    """
    rotations = sorted(set(s["rotation_round"] for s in all_rotation_summaries))
    caps_all  = sorted(set(s["cap"] for s in all_rotation_summaries), reverse=True)
    colors    = ["#2563EB", "#DC2626", "#16A34A"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for rot_idx, rot in enumerate(rotations):
        rot_data = [s for s in all_rotation_summaries if s["rotation_round"] == rot]
        rot_data.sort(key=lambda x: x["cap"], reverse=True)
        caps  = [s["cap"]       for s in rot_data]
        top1s = [s["mean_top1"] for s in rot_data]
        f1s   = [s["mean_macro_f1"] for s in rot_data]
        c = colors[rot_idx % len(colors)]

        axes[0].plot(caps, top1s, "o-", color=c, lw=2, ms=7,
                     label=f"Rotation {rot}")
        axes[1].plot(caps, f1s,   "o-", color=c, lw=2, ms=7,
                     label=f"Rotation {rot}")

    for ax, metric in zip(axes, ["Top-1 Accuracy (%)", "Macro F1 (%)"]):
        ax.set_xlabel("Cap (videos per word per user)", fontsize=11)
        ax.set_ylabel(metric, fontsize=11)
        ax.set_xticks(caps_all)
        ax.set_ylim(0, 110)
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)

    axes[0].set_title("Top-1 per Rotation Round", fontsize=12)
    axes[1].set_title("Macro F1 per Rotation Round", fontsize=12)
    fig.suptitle(
        "RQ2 — Per-Rotation Performance\n"
        "(Each rotation uses a different non-overlapping video window)",
        fontsize=13
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=130)
    plt.close()
    print(f"  Per-rotation curve saved : {save_path}")


def plot_boxplot(all_fold_results, save_path):
    """
    Box plot of Top-1 accuracy distribution per cap.
    Each box contains 3 rotations × 15 folds = 45 data points per cap.
    Shows spread and outliers clearly.
    """
    cap_values = sorted(set(r["cap"] for r in all_fold_results), reverse=True)
    data_per_cap = []
    for cap in cap_values:
        vals = [r["top1_acc"] for r in all_fold_results if r["cap"] == cap]
        data_per_cap.append(vals)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Top-1 boxplot
    bp = axes[0].boxplot(
        data_per_cap,
        labels=[str(c) for c in cap_values],
        patch_artist=True,
        medianprops=dict(color="black", linewidth=2),
        whiskerprops=dict(linewidth=1.5),
        capprops=dict(linewidth=1.5),
    )
    colors_grad = plt.cm.Blues(np.linspace(0.4, 0.85, len(cap_values)))
    for patch, color in zip(bp["boxes"], colors_grad):
        patch.set_facecolor(color)
    axes[0].set_xlabel("Cap (videos per word per user)", fontsize=11)
    axes[0].set_ylabel("Top-1 Accuracy (%)", fontsize=11)
    axes[0].set_title(
        "Top-1 Distribution per Cap\n(3 rotations × 15 folds = 45 points each)",
        fontsize=12
    )
    axes[0].grid(axis="y", alpha=0.3)
    axes[0].set_ylim(0, 105)

    # Macro F1 boxplot
    data_f1 = []
    for cap in cap_values:
        vals = [r["macro_f1"] for r in all_fold_results if r["cap"] == cap]
        data_f1.append(vals)

    bp2 = axes[1].boxplot(
        data_f1,
        labels=[str(c) for c in cap_values],
        patch_artist=True,
        medianprops=dict(color="black", linewidth=2),
        whiskerprops=dict(linewidth=1.5),
        capprops=dict(linewidth=1.5),
    )
    colors_grad2 = plt.cm.Greens(np.linspace(0.4, 0.85, len(cap_values)))
    for patch, color in zip(bp2["boxes"], colors_grad2):
        patch.set_facecolor(color)
    axes[1].set_xlabel("Cap (videos per word per user)", fontsize=11)
    axes[1].set_ylabel("Macro F1 (%)", fontsize=11)
    axes[1].set_title(
        "Macro F1 Distribution per Cap\n(3 rotations × 15 folds = 45 points each)",
        fontsize=12
    )
    axes[1].grid(axis="y", alpha=0.3)
    axes[1].set_ylim(0, 105)

    fig.suptitle("RQ2 — Score Distributions Across Folds & Rotations", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=130)
    plt.close()
    print(f"  Boxplot saved : {save_path}")


def plot_heatmap(all_rotation_summaries, save_path):
    """
    Heatmap: rows = caps, columns = rotations.
    Cell value = mean Top-1 across 15 folds for that (cap, rotation).
    Instantly reveals if any (cap, rotation) combination is an outlier.
    """
    caps      = sorted(set(s["cap"] for s in all_rotation_summaries), reverse=True)
    rotations = sorted(set(s["rotation_round"] for s in all_rotation_summaries))

    matrix = np.zeros((len(caps), len(rotations)))
    for i, cap in enumerate(caps):
        for j, rot in enumerate(rotations):
            match = [s for s in all_rotation_summaries
                     if s["cap"] == cap and s["rotation_round"] == rot]
            if match:
                matrix[i, j] = match[0]["mean_top1"]

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(matrix, cmap="YlGnBu", aspect="auto",
                   vmin=max(0, matrix.min() - 5), vmax=min(100, matrix.max() + 5))
    plt.colorbar(im, ax=ax, label="Mean Top-1 Accuracy (%)")

    ax.set_xticks(range(len(rotations)))
    ax.set_xticklabels([f"Rotation {r}" for r in rotations], fontsize=11)
    ax.set_yticks(range(len(caps)))
    ax.set_yticklabels([f"Cap={c}" for c in caps], fontsize=11)

    for i in range(len(caps)):
        for j in range(len(rotations)):
            ax.text(j, i, f"{matrix[i,j]:.1f}%",
                    ha="center", va="center", fontsize=11,
                    color="black" if matrix[i, j] > matrix.min() + (matrix.max() - matrix.min()) * 0.5
                    else "white")

    ax.set_title(
        "RQ2 — Top-1 Heatmap: Cap × Rotation\n"
        "(Each cell = mean over 15 LOSO folds)",
        fontsize=13
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=130)
    plt.close()
    print(f"  Heatmap saved : {save_path}")


def plot_fold_variance(all_fold_results, save_path):
    """
    Per-fold Top-1 accuracy across all caps (averaged over rotations).
    Shows which signers are consistently hard/easy to recognize regardless
    of training size — useful for diagnosing dataset imbalance.
    """
    folds     = sorted(set(r["fold_idx"] for r in all_fold_results))
    cap_values = sorted(set(r["cap"] for r in all_fold_results), reverse=True)
    colors    = plt.cm.tab20(np.linspace(0, 1, len(folds)))

    fig, ax = plt.subplots(figsize=(14, 6))

    for fi, fold in enumerate(folds):
        fold_data = [r for r in all_fold_results if r["fold_idx"] == fold]
        cap_means = []
        for cap in cap_values:
            vals = [r["top1_acc"] for r in fold_data if r["cap"] == cap]
            cap_means.append(np.mean(vals) if vals else float("nan"))

        test_user = next(r["test_user"] for r in fold_data)
        label = f"Fold {fold} ({test_user.split('_')[-1] if '_' in test_user else test_user[:12]})"
        ax.plot(cap_values, cap_means, "o-", color=colors[fi], lw=1.5,
                ms=5, label=label, alpha=0.8)

    ax.set_xlabel("Cap (videos per word per user)", fontsize=12)
    ax.set_ylabel("Top-1 Accuracy (%, avg over rotations)", fontsize=12)
    ax.set_title(
        "RQ2 — Per-Fold Accuracy Across Caps\n"
        "(Reveals which signers are consistently hard to recognize)",
        fontsize=13
    )
    ax.set_xticks(cap_values)
    ax.set_ylim(0, 110)
    ax.legend(fontsize=7, ncol=3, loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=130)
    plt.close()
    print(f"  Fold variance plot saved : {save_path}")


def plot_improvement_curve(cap_summaries, save_path):
    """
    Shows marginal gain in Top-1 for each additional cap unit.
    ΔTop-1 = Top1(cap) - Top1(cap-1). Useful for diminishing returns analysis.
    """
    caps  = [s["cap"]       for s in cap_summaries]   # already sorted high→low
    top1s = [s["mean_top1"] for s in cap_summaries]

    # Compute deltas going from low cap to high cap
    sorted_pairs = sorted(zip(caps, top1s))
    sorted_caps, sorted_top1s = zip(*sorted_pairs)
    deltas = [sorted_top1s[i] - sorted_top1s[i-1] for i in range(1, len(sorted_top1s))]
    delta_caps = sorted_caps[1:]

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    axes[0].plot(sorted_caps, sorted_top1s, "o-", color="#2563EB", lw=2.5, ms=8)
    for x, y in zip(sorted_caps, sorted_top1s):
        axes[0].annotate(f"{y:.1f}%", (x, y),
                         textcoords="offset points", xytext=(0, 10),
                         ha="center", fontsize=9, color="#2563EB")
    axes[0].set_xlabel("Cap (videos per word per user)", fontsize=11)
    axes[0].set_ylabel("Mean Top-1 Accuracy (%)", fontsize=11)
    axes[0].set_title("Absolute Performance vs Cap", fontsize=12)
    axes[0].set_xticks(sorted_caps)
    axes[0].set_ylim(0, 110)
    axes[0].grid(alpha=0.3)

    bar_colors = ["#16A34A" if d >= 0 else "#DC2626" for d in deltas]
    axes[1].bar([str(c) for c in delta_caps], deltas, color=bar_colors,
                edgecolor="black", linewidth=0.8)
    for i, (x, y) in enumerate(zip(delta_caps, deltas)):
        axes[1].annotate(f"{y:+.1f}%", (i, y),
                         textcoords="offset points",
                         xytext=(0, 5 if y >= 0 else -15),
                         ha="center", fontsize=10, fontweight="bold")
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_xlabel("Cap (transition to this cap from cap-1)", fontsize=11)
    axes[1].set_ylabel("ΔTop-1 Accuracy (%)", fontsize=11)
    axes[1].set_title("Marginal Gain per Additional Video/Word/User", fontsize=12)
    axes[1].grid(axis="y", alpha=0.3)

    fig.suptitle("RQ2 — Diminishing Returns Analysis", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=130)
    plt.close()
    print(f"  Improvement curve saved : {save_path}")


def plot_wrap_coverage(window_coverage_data, save_path):
    """
    Bar chart showing how many (user, word) pairs needed cyclic wrapping
    per (cap, rotation). High wrap counts = data scarcity concern.
    """
    if not window_coverage_data:
        return

    caps      = sorted(set(d["cap"]             for d in window_coverage_data), reverse=True)
    rotations = sorted(set(d["rotation_round"]  for d in window_coverage_data))

    fig, ax = plt.subplots(figsize=(12, 5))
    width  = 0.25
    colors = ["#2563EB", "#DC2626", "#16A34A"]
    x      = np.arange(len(caps))

    for ri, rot in enumerate(rotations):
        counts = []
        for cap in caps:
            match = [d for d in window_coverage_data
                     if d["cap"] == cap and d["rotation_round"] == rot]
            counts.append(match[0]["total_wrap_events"] if match else 0)
        ax.bar(x + ri * width, counts, width, label=f"Rotation {rot}",
               color=colors[ri], edgecolor="black", linewidth=0.7)

    ax.set_xticks(x + width)
    ax.set_xticklabels([f"Cap={c}" for c in caps], fontsize=11)
    ax.set_xlabel("Cap", fontsize=11)
    ax.set_ylabel("Total Cyclic-Wrap Events\n(summed over all folds)", fontsize=11)
    ax.set_title(
        "RQ2 — Window Wrap Events per (Cap, Rotation)\n"
        "(Wrapping occurs when a user has fewer videos than the window requires)",
        fontsize=12
    )
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=130)
    plt.close()
    print(f"  Wrap coverage plot saved : {save_path}")


# ============================================================
# MAIN
# ============================================================

def main(args):
    set_seed(config["seed"])
    os.makedirs(args.results_dir, exist_ok=True)

    # ── Save config ───────────────────────────────────────────
    with open(os.path.join(args.results_dir, "config.json"), "w") as f:
        json.dump({
            **config,
            "CAP_START"  : args.cap_start,
            "CAP_END"    : args.cap_end,
            "ROTATIONS"  : args.rotations,
            "DATA_ROOT"  : args.data_root,
            "RESULTS_DIR": args.results_dir,
        }, f, indent=4)

    # ── Logging ───────────────────────────────────────────────
    ts    = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_f = open(os.path.join(args.results_dir, f"rq2_log_{ts}.txt"), "w")
    sys.stdout = Tee(sys.stdout, log_f)
    sys.stderr = Tee(sys.stderr, log_f)

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    total_runs = (args.cap_start - args.cap_end + 1) * args.rotations * 15
    print("=" * 70)
    print(f"  RQ2 v2 — DataEfficiency-LOSO with Rotating Windows  |  {ts}")
    print(f"  Data root    : {args.data_root}")
    print(f"  Results dir  : {args.results_dir}")
    print(f"  Cap sweep    : {args.cap_start} → {args.cap_end}")
    print(f"  Rotations    : {args.rotations} per cap")
    print(f"  Total runs   : {total_runs}  "
          f"({args.cap_start - args.cap_end + 1} caps × "
          f"{args.rotations} rotations × 15 folds)")
    print(f"  Device       : {device}")
    if device.type == "cuda":
        print(f"  GPU          : {torch.cuda.get_device_name(device.index)}")
    print("=" * 70)

    # ── Discover and group files ──────────────────────────────
    all_paths = find_h5_files(args.data_root)
    if not all_paths:
        raise RuntimeError(f"No .h5 files found under: {args.data_root}")
    print(f"\n  Total .h5 files : {len(all_paths)}")

    user_word_paths = group_by_user_and_word(all_paths)
    users           = sorted(user_word_paths.keys())
    num_folds       = len(users)
    print(f"  Users ({num_folds})       : {users}")

    print(f"\n  Per-user recording stats:")
    print(f"  {'User':<25} {'Words':>6} {'MinReps':>8} {'MaxReps':>8} {'MeanReps':>9}")
    print(f"  {'-'*60}")
    for u in users:
        counts = [len(v) for v in user_word_paths[u].values()]
        print(f"  {u:<25} {len(counts):>6} {min(counts):>8} "
              f"{max(counts):>8} {np.mean(counts):>9.1f}")

    # ── Label encoder ─────────────────────────────────────────
    all_labels  = [label_from_filename(p) for p in all_paths]
    all_classes = sorted(set(all_labels))
    num_classes = len(all_classes)

    label_encoder = LabelEncoder()
    label_encoder.fit(all_classes)
    np.save(
        os.path.join(args.results_dir, "class_names.npy"),
        np.array(label_encoder.classes_),
    )
    print(f"\n  Word classes : {num_classes}")

    # ── LOSO fold assignments ─────────────────────────────────
    fold_assignments = []
    for i, test_user in enumerate(users):
        fold_assignments.append({
            "fold_idx"   : i,
            "test_user"  : test_user,
            "train_users": [u for u in users if u != test_user],
        })

    # ── Fix test paths (never subsampled) ─────────────────────
    test_paths_per_fold    = {}
    missing_words_per_fold = {}
    for fa in fold_assignments:
        fi = fa["fold_idx"]
        test_paths_per_fold[fi] = [
            p
            for word_paths in user_word_paths[fa["test_user"]].values()
            for p in word_paths
        ]
        test_labels = set(label_from_filename(p) for p in test_paths_per_fold[fi])
        missing_words_per_fold[fi] = sorted(set(all_classes) - test_labels)

    # Save test coverage CSV once
    coverage_csv = os.path.join(args.results_dir, "test_coverage.csv")
    with open(coverage_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "fold", "test_user", "test_files",
            "n_missing_words", "missing_words",
        ])
        for fa in fold_assignments:
            fi = fa["fold_idx"]
            writer.writerow([
                fi, fa["test_user"],
                len(test_paths_per_fold[fi]),
                len(missing_words_per_fold[fi]),
                "|".join(missing_words_per_fold[fi]),
            ])
    print(f"  Test coverage saved: {coverage_csv}")

    # ── Global summary CSV header ─────────────────────────────
    global_csv = os.path.join(args.results_dir, "rq2_summary.csv")
    with open(global_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "cap", "rotation_round",
            "nominal_train_per_word",
            "actual_mean_per_word",
            "total_wrap_events",
            "mean_top1(%)", "std_top1(%)",
            "mean_top5(%)",
            "mean_macro_f1(%)", "std_macro_f1(%)",
            "mean_weighted_f1(%)",
        ])

    cap_avg_csv = os.path.join(args.results_dir, "rq2_cap_averaged_summary.csv")
    with open(cap_avg_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "cap",
            "nominal_train_per_word",
            "actual_mean_per_word",
            "mean_top1(%)", "std_top1(%)",
            "mean_top5(%)",
            "mean_macro_f1(%)", "std_macro_f1(%)",
            "mean_weighted_f1(%)",
            "mean_wrap_events",
        ])

    # ── Window coverage log ───────────────────────────────────
    window_coverage_csv = os.path.join(args.results_dir, "rq2_window_coverage.csv")
    with open(window_coverage_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "cap", "rotation_round", "fold_idx", "test_user",
            "n_wrap_events", "user", "word",
            "n_available", "window_start", "window_end", "indices_used",
        ])

    # ============================================================
    # MAIN SWEEP: cap × rotation × fold
    # ============================================================

    cap_values           = list(range(args.cap_start, args.cap_end - 1, -1))
    all_rotation_summaries = []   # one entry per (cap, rotation)
    all_fold_results       = []   # one entry per (cap, rotation, fold)
    window_coverage_data   = []   # for the wrap plot

    run_counter = 0

    for cap in cap_values:
        nominal_per_word = (num_folds - 1) * cap
        cap_dir          = os.path.join(args.results_dir, f"cap_{cap}")
        os.makedirs(cap_dir, exist_ok=True)
        cap_rotation_results = []  # all fold results for all rotations of this cap

        print(f"\n\n{'#'*70}")
        print(f"  CAP = {cap}  |  nominal train/word = {nominal_per_word}  |  "
              f"{args.rotations} rotations × 15 folds = "
              f"{args.rotations * 15} runs")
        print(f"{'#'*70}")

        for rotation_round in range(args.rotations):
            rot_dir = os.path.join(cap_dir, f"rotation_{rotation_round}")
            os.makedirs(rot_dir, exist_ok=True)
            rot_fold_results = []
            total_wrap_this_rot = 0

            print(f"\n  ── Cap={cap}, Rotation={rotation_round} "
                  f"(window offset = {rotation_round * cap}) ──")

            for fa in fold_assignments:
                fold_idx    = fa["fold_idx"]
                test_user   = fa["test_user"]
                train_users = fa["train_users"]
                fold_log_dir = os.path.join(rot_dir, f"fold_{fold_idx}")
                run_counter += 1

                print(f"\n  [{run_counter}/{total_runs}] "
                      f"Cap={cap} | Rot={rotation_round} | "
                      f"Fold={fold_idx} | test={test_user}")

                # ── Window-based sampling ──────────────────────
                train_paths, sample_stats, wrap_events = sample_train_paths_windowed(
                    train_users     = train_users,
                    user_word_paths = user_word_paths,
                    cap             = cap,
                    rotation_round  = rotation_round,
                    fold_idx        = fold_idx,
                    base_seed       = config["seed"],
                )
                test_paths = test_paths_per_fold[fold_idx]
                total_wrap_this_rot += sample_stats["n_wrap_events"]

                print(f"    train_files={sample_stats['total_files']} | "
                      f"test_files={len(test_paths)} | "
                      f"mean_per_word={sample_stats['mean_per_word']:.1f} | "
                      f"wrap_events={sample_stats['n_wrap_events']}")

                # Log wrap events to CSV
                if wrap_events:
                    with open(window_coverage_csv, "a", newline="") as f:
                        writer = csv.writer(f)
                        for we in wrap_events:
                            writer.writerow([
                                cap, rotation_round, fold_idx, test_user,
                                sample_stats["n_wrap_events"],
                                we["user"], we["word"],
                                we["n_available"], we["window_start"],
                                we["window_end"],
                                str(we["indices_used"]),
                            ])

                # ── Train ─────────────────────────────────────
                result = train_one_fold(
                    fold_idx      = fold_idx,
                    train_paths   = train_paths,
                    test_paths    = test_paths,
                    label_encoder = label_encoder,
                    class_names   = list(label_encoder.classes_),
                    fold_log_dir  = fold_log_dir,
                    device        = device,
                    config        = config,
                    tb_root       = os.path.join(
                        args.results_dir, "runs",
                        f"cap_{cap}", f"rotation_{rotation_round}"
                    ),
                )

                # ── Evaluate ──────────────────────────────────
                eval_metrics = evaluate_fold(
                    fold_idx      = fold_idx,
                    fold_log_dir  = fold_log_dir,
                    test_paths    = test_paths,
                    label_encoder = label_encoder,
                    class_names   = list(label_encoder.classes_),
                    device        = device,
                    config        = config,
                )

                fold_result = {
                    "cap"                   : cap,
                    "rotation_round"        : rotation_round,
                    "fold_idx"              : fold_idx,
                    "test_user"             : test_user,
                    "train_files"           : sample_stats["total_files"],
                    "actual_mean_per_word"  : sample_stats["mean_per_word"],
                    "n_wrap_events"         : sample_stats["n_wrap_events"],
                    "top1_acc"              : eval_metrics["top1_acc"],
                    "top5_acc"              : eval_metrics["top5_acc"],
                    "macro_f1"              : eval_metrics["macro_f1"],
                    "weighted_f1"           : eval_metrics["weighted_f1"],
                    "test_loss"             : eval_metrics["test_loss"],
                }
                rot_fold_results.append(fold_result)
                cap_rotation_results.append(fold_result)
                all_fold_results.append(fold_result)

            # ── Rotation-level aggregation ─────────────────────
            top1s   = [r["top1_acc"]          for r in rot_fold_results]
            top5s   = [r["top5_acc"]          for r in rot_fold_results]
            mac_f1s = [r["macro_f1"]          for r in rot_fold_results]
            wtd_f1s = [r["weighted_f1"]       for r in rot_fold_results]
            act_mean = np.mean([r["actual_mean_per_word"] for r in rot_fold_results])

            rot_summary = {
                "cap"               : cap,
                "rotation_round"    : rotation_round,
                "nominal_per_word"  : nominal_per_word,
                "actual_mean_per_word": float(act_mean),
                "total_wrap_events" : total_wrap_this_rot,
                "mean_top1"         : float(np.mean(top1s)),
                "std_top1"          : float(np.std(top1s)),
                "mean_top5"         : float(np.mean(top5s)),
                "mean_macro_f1"     : float(np.mean(mac_f1s)),
                "std_macro_f1"      : float(np.std(mac_f1s)),
                "mean_weighted_f1"  : float(np.mean(wtd_f1s)),
            }
            all_rotation_summaries.append(rot_summary)
            window_coverage_data.append({
                "cap"               : cap,
                "rotation_round"    : rotation_round,
                "total_wrap_events" : total_wrap_this_rot,
            })

            print(f"\n  ── Rotation {rotation_round} Summary (cap={cap}) ──")
            print(f"  Mean Top-1    : {rot_summary['mean_top1']:.2f}% ± {rot_summary['std_top1']:.2f}%")
            print(f"  Mean Top-5    : {rot_summary['mean_top5']:.2f}%")
            print(f"  Mean Macro F1 : {rot_summary['mean_macro_f1']:.2f}%")
            print(f"  Wrap events   : {total_wrap_this_rot}")

            print(f"\n  {'Fold':<6} {'Test User':<25} "
                  f"{'Top-1':>8} {'Top-5':>8} {'MacroF1':>9}")
            print(f"  {'-'*60}")
            for r in rot_fold_results:
                print(f"  {r['fold_idx']:<6} {r['test_user']:<25} "
                      f"{r['top1_acc']:7.2f}%  "
                      f"{r['top5_acc']:7.2f}%  "
                      f"{r['macro_f1']:8.2f}%")

            # Save per-rotation CSV
            rot_csv = os.path.join(rot_dir, "rotation_summary.csv")
            with open(rot_csv, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "fold", "test_user",
                    "top1(%)", "top5(%)", "macro_f1(%)", "weighted_f1(%)",
                    "train_files", "actual_mean_per_word", "n_wrap_events",
                ])
                for r in rot_fold_results:
                    writer.writerow([
                        r["fold_idx"], r["test_user"],
                        f"{r['top1_acc']:.2f}", f"{r['top5_acc']:.2f}",
                        f"{r['macro_f1']:.2f}", f"{r['weighted_f1']:.2f}",
                        r["train_files"],
                        f"{r['actual_mean_per_word']:.1f}",
                        r["n_wrap_events"],
                    ])
                writer.writerow([])
                writer.writerow([
                    "MEAN", "—",
                    f"{rot_summary['mean_top1']:.2f}",
                    f"{rot_summary['mean_top5']:.2f}",
                    f"{rot_summary['mean_macro_f1']:.2f}",
                    f"{rot_summary['mean_weighted_f1']:.2f}",
                    "—", f"{act_mean:.1f}", total_wrap_this_rot,
                ])
            print(f"  Rotation CSV saved: {rot_csv}")

            # Append to global summary
            with open(global_csv, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    cap, rotation_round, nominal_per_word,
                    f"{act_mean:.1f}", total_wrap_this_rot,
                    f"{rot_summary['mean_top1']:.2f}",
                    f"{rot_summary['std_top1']:.2f}",
                    f"{rot_summary['mean_top5']:.2f}",
                    f"{rot_summary['mean_macro_f1']:.2f}",
                    f"{rot_summary['std_macro_f1']:.2f}",
                    f"{rot_summary['mean_weighted_f1']:.2f}",
                ])

        # ── Cap-level aggregation (over all rotations) ─────────
        all_top1s   = [r["top1_acc"]    for r in cap_rotation_results]
        all_top5s   = [r["top5_acc"]    for r in cap_rotation_results]
        all_mac_f1s = [r["macro_f1"]    for r in cap_rotation_results]
        all_wtd_f1s = [r["weighted_f1"] for r in cap_rotation_results]
        all_acts    = [r["actual_mean_per_word"] for r in cap_rotation_results]
        all_wraps   = sum(r["n_wrap_events"] for r in cap_rotation_results)

        cap_mean_top1    = float(np.mean(all_top1s))
        cap_std_top1     = float(np.std(all_top1s))
        cap_mean_top5    = float(np.mean(all_top5s))
        cap_mean_mac_f1  = float(np.mean(all_mac_f1s))
        cap_std_mac_f1   = float(np.std(all_mac_f1s))
        cap_mean_wtd_f1  = float(np.mean(all_wtd_f1s))
        cap_mean_act     = float(np.mean(all_acts))

        cap_summary_entry = {
            "cap"               : cap,
            "nominal_per_word"  : nominal_per_word,
            "actual_mean_per_word": cap_mean_act,
            "mean_top1"         : cap_mean_top1,
            "std_top1"          : cap_std_top1,
            "mean_top5"         : cap_mean_top5,
            "mean_macro_f1"     : cap_mean_mac_f1,
            "std_macro_f1"      : cap_std_mac_f1,
            "mean_weighted_f1"  : cap_mean_wtd_f1,
            "mean_wrap_events"  : all_wraps / (args.rotations * num_folds),
        }

        print(f"\n  ══ Cap={cap} OVERALL Summary (all {args.rotations} rotations) ══")
        print(f"  Mean Top-1    : {cap_mean_top1:.2f}% ± {cap_std_top1:.2f}%")
        print(f"  Mean Top-5    : {cap_mean_top5:.2f}%")
        print(f"  Mean Macro F1 : {cap_mean_mac_f1:.2f}%")
        print(f"  Total wraps   : {all_wraps}")

        # Save per-cap summary CSV
        cap_csv = os.path.join(cap_dir, "cap_summary.csv")
        with open(cap_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "rotation_round", "fold", "test_user",
                "top1(%)", "top5(%)", "macro_f1(%)", "weighted_f1(%)",
                "train_files", "actual_mean_per_word", "n_wrap_events",
            ])
            for r in cap_rotation_results:
                writer.writerow([
                    r["rotation_round"], r["fold_idx"], r["test_user"],
                    f"{r['top1_acc']:.2f}", f"{r['top5_acc']:.2f}",
                    f"{r['macro_f1']:.2f}", f"{r['weighted_f1']:.2f}",
                    r["train_files"],
                    f"{r['actual_mean_per_word']:.1f}",
                    r["n_wrap_events"],
                ])
            writer.writerow([])
            writer.writerow([
                "MEAN(all rotations)", "—", "—",
                f"{cap_mean_top1:.2f}", f"{cap_mean_top5:.2f}",
                f"{cap_mean_mac_f1:.2f}", f"{cap_mean_wtd_f1:.2f}",
                "—", f"{cap_mean_act:.1f}", f"{all_wraps}",
            ])
        print(f"  Cap summary CSV saved: {cap_csv}")

        # Append to cap-averaged CSV
        with open(cap_avg_csv, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                cap, nominal_per_word,
                f"{cap_mean_act:.1f}",
                f"{cap_mean_top1:.2f}", f"{cap_std_top1:.2f}",
                f"{cap_mean_top5:.2f}",
                f"{cap_mean_mac_f1:.2f}", f"{cap_std_mac_f1:.2f}",
                f"{cap_mean_wtd_f1:.2f}",
                f"{cap_summary_entry['mean_wrap_events']:.2f}",
            ])

    # ============================================================
    # FINAL SUMMARY
    # ============================================================

    print("\n\n" + "=" * 70)
    print("  RQ2 v2 COMPLETE — FINAL SUMMARY (cap-averaged over rotations)")
    print("=" * 70)

    # Recompute cap summaries from all_fold_results for plotting
    cap_summaries = []
    for cap in cap_values:
        cap_results = [r for r in all_fold_results if r["cap"] == cap]
        top1s   = [r["top1_acc"]  for r in cap_results]
        top5s   = [r["top5_acc"]  for r in cap_results]
        mac_f1s = [r["macro_f1"]  for r in cap_results]
        wtd_f1s = [r["weighted_f1"] for r in cap_results]
        cap_summaries.append({
            "cap"               : cap,
            "nominal_per_word"  : (num_folds - 1) * cap,
            "actual_mean_per_word": np.mean([r["actual_mean_per_word"] for r in cap_results]),
            "mean_top1"         : float(np.mean(top1s)),
            "std_top1"          : float(np.std(top1s)),
            "mean_top5"         : float(np.mean(top5s)),
            "mean_macro_f1"     : float(np.mean(mac_f1s)),
            "std_macro_f1"      : float(np.std(mac_f1s)),
            "mean_weighted_f1"  : float(np.mean(wtd_f1s)),
        })

    print(f"\n  {'Cap':<5} {'Nominal/word':<14} {'Top-1':>9} "
          f"{'±std':>7} {'Top-5':>9} {'MacroF1':>9}")
    print(f"  {'-'*58}")
    for s in cap_summaries:
        print(f"  {s['cap']:<5} {s['nominal_per_word']:<14} "
              f"{s['mean_top1']:7.2f}%  ±{s['std_top1']:.2f}%  "
              f"{s['mean_top5']:7.2f}%  {s['mean_macro_f1']:7.2f}%")

    # ── Generate all plots ─────────────────────────────────────
    print("\n  Generating plots...")

    plot_main_curve(
        cap_summaries,
        os.path.join(args.results_dir, "rq2_accuracy_curve.png"),
    )
    plot_per_rotation_curves(
        all_rotation_summaries,
        os.path.join(args.results_dir, "rq2_per_rotation_curves.png"),
    )
    plot_boxplot(
        all_fold_results,
        os.path.join(args.results_dir, "rq2_boxplot.png"),
    )
    plot_heatmap(
        all_rotation_summaries,
        os.path.join(args.results_dir, "rq2_heatmap.png"),
    )
    plot_fold_variance(
        all_fold_results,
        os.path.join(args.results_dir, "rq2_fold_variance.png"),
    )
    plot_improvement_curve(
        cap_summaries,
        os.path.join(args.results_dir, "rq2_improvement_curve.png"),
    )
    plot_wrap_coverage(
        window_coverage_data,
        os.path.join(args.results_dir, "rq2_wrap_coverage.png"),
    )

    print(f"\n  Global summary CSV      : {global_csv}")
    print(f"  Cap-averaged CSV        : {cap_avg_csv}")
    print(f"  Window coverage CSV     : {window_coverage_csv}")
    print(f"  All results             : {args.results_dir}/")
    print(f"  TensorBoard             : "
          f"tensorboard --logdir={args.results_dir}/runs")
    print("=" * 70)


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RQ2 v2: Progressive training data reduction with rotating windows"
    )
    parser.add_argument(
        "--data_root",   type=str, default=DATA_ROOT,
        help="Root directory of H5 files",
    )
    parser.add_argument(
        "--results_dir", type=str, default=RESULTS_DIR,
        help="Output directory",
    )
    parser.add_argument(
        "--cap_start",   type=int, default=CAP_START,
        help="Starting cap (max reps per word per user, default=5)",
    )
    parser.add_argument(
        "--cap_end",     type=int, default=CAP_END,
        help="Ending cap (min reps per word per user, default=1)",
    )
    parser.add_argument(
        "--rotations",   type=int, default=ROTATIONS,
        help="Number of rotation rounds per cap (default=3)",
    )
    args = parser.parse_args()
    main(args)
