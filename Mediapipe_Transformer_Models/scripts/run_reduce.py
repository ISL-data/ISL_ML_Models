
"""
rq2_runner.py  —  RQ2: Progressive Training Data Reduction (DataEfficiency-LOSO)
==================================================================================
Analyzes the relationship between training data size and signer-independent
recognition performance under the LOSO protocol.

Cap-based subsampling strategy:
    For each round, a cap is applied to the maximum number of recordings
    retained per word per training user. Words with more recordings than
    the cap are randomly subsampled down to the cap. Words with fewer
    recordings than the cap contribute all available recordings.

    Round 1  : cap=10  →  min(actual, 10) reps per word per user
    Round 2  : cap=9   →  min(actual, 9)
    ...
    Round 10 : cap=1   →  min(actual, 1)

    Nominal train videos per word = 14 users × cap
    Actual  train videos per word = sum of min(user_word_count, cap)
                                    averaged across all words and folds
    The actual mean is used as the X-axis in the result curve.

Reproducibility:
    Subsampling seed = base_seed + round_idx * 1000 + fold_idx
    Same (round, fold) always selects identical videos across runs.

Test set:
    Fixed across all rounds — held-out user's complete recordings.
    Never subsampled.

Outputs (saved to RESULTS_DIR):
    cap_10/  cap_9/  ...  cap_1/
        fold_0/ ... fold_14/
            final_model.pt
            norm_mean.npy, norm_std.npy
            eval/
                confusion_matrix.png
                top_confused_pairs.png/.csv
                per_class_accuracy.csv/_bar.png
                classification_report.txt
                metrics.txt
        round_summary.csv
    rq2_summary.csv
    rq2_accuracy_curve.png
    test_coverage.csv
    class_names.npy
    config.json
    rq2_log_TIMESTAMP.txt

Run:
    python rq2_runner.py
    python rq2_runner.py --cap_start 10 --cap_end 1 --results_dir results_rq2
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

DATA_ROOT   = "/mnt/9a528fe4-4fe8-4dff-9a0c-8b1a3cf3d7ba/popsign/ISL_Goa_Data_h5_30fps"
RESULTS_DIR = "results_rq2"

CAP_START = 10   # maximum reps per word per user (full data)
CAP_END   = 1    # minimum reps per word per user

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
    """
    uwp = defaultdict(lambda: defaultdict(list))
    for p in paths:
        uwp[user_from_path(p)][label_from_filename(p)].append(p)
    return uwp


# ============================================================
# CAP-BASED SUBSAMPLING
# ============================================================

def sample_train_paths(train_users, user_word_paths, cap,
                       round_idx, fold_idx, base_seed):
    """
    For each training user and each word, retain at most `cap`
    recordings. Words with fewer recordings than `cap` contribute
    all their recordings unchanged.

    Subsampling is deterministic:
        seed = base_seed + round_idx * 1000 + fold_idx

    Parameters
    ----------
    train_users     : list[str]   — user IDs in training split
    user_word_paths : dict        — user → word → list[path]
    cap             : int         — max recordings per word per user
    round_idx       : int         — current round (0-based)
    fold_idx        : int         — current fold (0-based)
    base_seed       : int         — base seed from config

    Returns
    -------
    selected : list[str]  — flat list of selected h5 paths
    stats    : dict       — actual counts for logging
    """
    seed = base_seed + round_idx * 1000 + fold_idx
    rng  = np.random.default_rng(seed=seed)

    selected       = []
    per_word_counts = defaultdict(int)   # word → total selected across users

    for user in sorted(train_users):
        for word in sorted(user_word_paths[user].keys()):
            paths = user_word_paths[user][word]
            n     = len(paths)

            if n <= cap:
                # Fewer than cap — keep all
                chosen = paths
            else:
                # More than cap — randomly subsample down to cap
                idx    = rng.choice(n, size=cap, replace=False)
                chosen = [paths[i] for i in idx]

            selected.extend(chosen)
            per_word_counts[word] += len(chosen)

    counts = list(per_word_counts.values())
    stats  = {
        "total_files"   : len(selected),
        "mean_per_word" : float(np.mean(counts)),
        "min_per_word"  : int(np.min(counts)),
        "max_per_word"  : int(np.max(counts)),
        "words_at_cap"  : int(sum(
            1 for c in counts
            if c == cap * len(train_users)
        )),
        "words_below_cap": int(sum(
            1 for c in counts
            if c < cap * len(train_users)
        )),
    }
    return selected, stats


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

    # Confusion matrix — always square
    cm = confusion_matrix(
        y_true, y_pred,
        labels=list(range(num_classes))
    )
    save_confusion_matrix(
        cm, class_names,
        os.path.join(eval_dir, "confusion_matrix.png"),
        title=f"Confusion Matrix — Fold {fold_idx}",
    )
    save_top_confused_pairs(
        cm, class_names,
        os.path.join(eval_dir, "top_confused_pairs.png"),
        n=20,
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
# PLOTTING
# ============================================================

def plot_rq2_curve(round_summaries, save_path):
    """
    Accuracy vs actual mean training videos per word.
    X-axis : actual mean training videos per word (high → low)
    Y-axis : accuracy / F1 (%)
    Shaded band = ± 1 std across folds for Top-1.
    """
    vids  = [s["actual_mean_per_word"] for s in round_summaries]
    top1s = [s["mean_top1"]            for s in round_summaries]
    top5s = [s["mean_top5"]            for s in round_summaries]
    f1s   = [s["mean_macro_f1"]        for s in round_summaries]
    stds  = [s["std_top1"]             for s in round_summaries]

    fig, ax = plt.subplots(figsize=(13, 6))

    ax.plot(vids, top1s, marker="o", color="steelblue",
            linewidth=2.5, markersize=8, label="Top-1 Accuracy")
    ax.fill_between(
        vids,
        [t - s for t, s in zip(top1s, stds)],
        [t + s for t, s in zip(top1s, stds)],
        color="steelblue", alpha=0.15, label="Top-1 ± 1 std (across folds)"
    )
    ax.plot(vids, top5s, marker="s", color="coral",
            linewidth=2.5, markersize=8, label="Top-5 Accuracy")
    ax.plot(vids, f1s,   marker="^", color="mediumseagreen",
            linewidth=2.5, markersize=8, label="Macro F1")

    for x, y in zip(vids, top1s):
        ax.annotate(
            f"{y:.1f}%", (x, y),
            textcoords="offset points", xytext=(0, 12),
            ha="center", fontsize=9,
            color="steelblue", fontweight="bold",
        )

    ax.set_xlabel(
        "Actual Mean Training Videos per Word  "
        "(averaged across folds and words)",
        fontsize=12,
    )
    ax.set_ylabel("Accuracy / F1  (%)", fontsize=12)
    ax.set_title(
        "RQ2 — Effect of Training Data Size on LOSO Recognition Performance\n"
        "(Test set fixed: held-out signer, all available recordings)",
        fontsize=13,
    )
    ax.set_xticks(vids)
    ax.set_xticklabels([f"{v:.0f}" for v in vids], fontsize=10)
    ax.set_ylim(0, 110)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f" RQ2 curve saved: {save_path}")


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
            "DATA_ROOT"  : args.data_root,
            "RESULTS_DIR": args.results_dir,
        }, f, indent=4)

    # ── Logging ───────────────────────────────────────────────
    ts    = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_f = open(
        os.path.join(args.results_dir, f"rq2_log_{ts}.txt"), "w"
    )
    sys.stdout = Tee(sys.stdout, log_f)
    sys.stderr = Tee(sys.stderr, log_f)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("=" * 65)
    print(f"  RQ2 — DataEfficiency-LOSO  |  {ts}")
    print(f"  Data root   : {args.data_root}")
    print(f"  Results dir : {args.results_dir}")
    print(f"  Cap sweep   : {args.cap_start} → {args.cap_end}")
    print(f"  Device      : {device}")
    if device.type == "cuda":
        print(f"  GPU         : {torch.cuda.get_device_name(0)}")
    print("=" * 65)

    # ── Discover and group files ──────────────────────────────
    all_paths = find_h5_files(args.data_root)
    if not all_paths:
        raise RuntimeError(f"No .h5 files found under: {args.data_root}")

    print(f"\n  Total .h5 files : {len(all_paths)}")

    user_word_paths = group_by_user_and_word(all_paths)
    users           = sorted(user_word_paths.keys())
    num_folds       = len(users)

    print(f"  Users ({num_folds})       : {users}")

    # Print per-user word count stats
    print(f"\n  Per-user recording stats:")
    print(f"  {'User':<25} {'Words':>6} {'MinReps':>8} "
          f"{'MaxReps':>8} {'MeanReps':>9}")
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

    print(f"\n  {'Fold':<6} {'Test User':<25} {'Test Files':>10}")
    print(f"  {'-'*45}")

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
        test_labels = set(
            label_from_filename(p)
            for p in test_paths_per_fold[fi]
        )
        missing_words_per_fold[fi] = sorted(
            set(all_classes) - test_labels
        )
        print(f"  {fi:<6} {fa['test_user']:<25} "
              f"{len(test_paths_per_fold[fi]):>10}  "
              f"missing_words={len(missing_words_per_fold[fi])}")

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
    print(f"\n  Test coverage saved: {coverage_csv}")

    # ── Global summary CSV header ─────────────────────────────
    global_csv = os.path.join(args.results_dir, "rq2_summary.csv")
    with open(global_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "round", "cap",
            "nominal_train_per_word",
            "actual_mean_per_word",
            "actual_min_per_word",
            "actual_max_per_word",
            "mean_top1(%)", "std_top1(%)",
            "mean_top5(%)",
            "mean_macro_f1(%)", "std_macro_f1(%)",
            "mean_weighted_f1(%)",
        ])

    # ============================================================
    # CAP SWEEP
    # ============================================================

    cap_values      = list(range(args.cap_start, args.cap_end - 1, -1))
    round_summaries = []

    for round_idx, cap in enumerate(cap_values):

        nominal_per_word = (num_folds - 1) * cap  # e.g. 14 × 10 = 140
        round_dir        = os.path.join(args.results_dir, f"cap_{cap}")
        os.makedirs(round_dir, exist_ok=True)

        print(f"\n\n{'#'*65}")
        print(f"  ROUND {round_idx + 1}/{len(cap_values)}")
        print(f"  Cap per word per user : {cap}")
        print(f"  Nominal train/word    : {nominal_per_word}  "
              f"(14 users × {cap})")
        print(f"  Output folder         : {round_dir}")
        print(f"{'#'*65}")

        fold_results = []

        for fa in fold_assignments:
            fold_idx    = fa["fold_idx"]
            test_user   = fa["test_user"]
            train_users = fa["train_users"]

            fold_log_dir = os.path.join(round_dir, f"fold_{fold_idx}")

            # ── Sample training data with cap ─────────────────
            train_paths, sample_stats = sample_train_paths(
                train_users     = train_users,
                user_word_paths = user_word_paths,
                cap             = cap,
                round_idx       = round_idx,
                fold_idx        = fold_idx,
                base_seed       = config["seed"],
            )
            test_paths = test_paths_per_fold[fold_idx]

            print(f"\n  Fold {fold_idx} | test={test_user} | "
                  f"cap={cap} | "
                  f"train_files={sample_stats['total_files']} | "
                  f"test_files={len(test_paths)}")
            print(f"    Train vids/word — "
                  f"nominal={nominal_per_word} "
                  f"actual: mean={sample_stats['mean_per_word']:.1f} "
                  f"min={sample_stats['min_per_word']} "
                  f"max={sample_stats['max_per_word']} "
                  f"words_at_cap={sample_stats['words_at_cap']} "
                  f"words_below_cap={sample_stats['words_below_cap']}")

            # ── Train ─────────────────────────────────────────
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
                    args.results_dir, "runs", f"cap_{cap}"
                ),
            )

            # ── Evaluate ──────────────────────────────────────
            print(f"\n  Evaluating fold {fold_idx}...")
            eval_metrics = evaluate_fold(
                fold_idx      = fold_idx,
                fold_log_dir  = fold_log_dir,
                test_paths    = test_paths,
                label_encoder = label_encoder,
                class_names   = list(label_encoder.classes_),
                device        = device,
                config        = config,
            )

            fold_results.append({
                "fold"                  : fold_idx,
                "test_user"             : test_user,
                "train_users"           : train_users,
                "cap"                   : cap,
                "train_files"           : sample_stats["total_files"],
                "actual_mean_per_word"  : sample_stats["mean_per_word"],
                "actual_min_per_word"   : sample_stats["min_per_word"],
                "actual_max_per_word"   : sample_stats["max_per_word"],
                "words_at_cap"          : sample_stats["words_at_cap"],
                "words_below_cap"       : sample_stats["words_below_cap"],
                "missing_words_in_test" : len(missing_words_per_fold[fold_idx]),
                "top1_acc"              : eval_metrics["top1_acc"],
                "top5_acc"              : eval_metrics["top5_acc"],
                "macro_f1"              : eval_metrics["macro_f1"],
                "weighted_f1"           : eval_metrics["weighted_f1"],
                "test_loss"             : eval_metrics["test_loss"],
                "y_true"                : eval_metrics["y_true"],
                "y_pred"                : eval_metrics["y_pred"],
            })

        # ── Round aggregation ─────────────────────────────────
        top1s    = [r["top1_acc"]   for r in fold_results]
        top5s    = [r["top5_acc"]   for r in fold_results]
        mac_f1s  = [r["macro_f1"]   for r in fold_results]
        wtd_f1s  = [r["weighted_f1"] for r in fold_results]
        act_mean = [r["actual_mean_per_word"] for r in fold_results]
        act_min  = [r["actual_min_per_word"]  for r in fold_results]
        act_max  = [r["actual_max_per_word"]  for r in fold_results]

        mean_top1    = float(np.mean(top1s))
        std_top1     = float(np.std(top1s))
        mean_top5    = float(np.mean(top5s))
        mean_mac_f1  = float(np.mean(mac_f1s))
        std_mac_f1   = float(np.std(mac_f1s))
        mean_wtd_f1  = float(np.mean(wtd_f1s))
        global_mean  = float(np.mean(act_mean))
        global_min   = int(np.min(act_min))
        global_max   = int(np.max(act_max))

        print(f"\n  ── Round {round_idx+1} Summary (cap={cap}) ──")
        print(f"  Mean Top-1    : {mean_top1:.2f}% ± {std_top1:.2f}%")
        print(f"  Mean Top-5    : {mean_top5:.2f}%")
        print(f"  Mean Macro F1 : {mean_mac_f1:.2f}%")
        print(f"  Actual mean train vids/word : {global_mean:.1f}")

        print(f"\n  {'Fold':<6} {'Test User':<25} "
              f"{'Top-1':>8} {'Top-5':>8} {'MacroF1':>9}")
        print(f"  {'-'*60}")
        for r in fold_results:
            print(f"  {r['fold']:<6} {r['test_user']:<25} "
                  f"{r['top1_acc']:7.2f}%  "
                  f"{r['top5_acc']:7.2f}%  "
                  f"{r['macro_f1']:8.2f}%")

        # Per-round CSV
        round_csv = os.path.join(round_dir, "round_summary.csv")
        with open(round_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "fold", "test_user",
                "top1(%)", "top5(%)",
                "macro_f1(%)", "weighted_f1(%)",
                "train_files",
                "actual_mean_per_word",
                "actual_min_per_word",
                "actual_max_per_word",
                "words_at_cap",
                "words_below_cap",
                "missing_words_in_test",
            ])
            for r in fold_results:
                writer.writerow([
                    r["fold"], r["test_user"],
                    f"{r['top1_acc']:.2f}",
                    f"{r['top5_acc']:.2f}",
                    f"{r['macro_f1']:.2f}",
                    f"{r['weighted_f1']:.2f}",
                    r["train_files"],
                    f"{r['actual_mean_per_word']:.1f}",
                    r["actual_min_per_word"],
                    r["actual_max_per_word"],
                    r["words_at_cap"],
                    r["words_below_cap"],
                    r["missing_words_in_test"],
                ])
            writer.writerow([])
            writer.writerow([
                "MEAN", "—",
                f"{mean_top1:.2f}", f"{mean_top5:.2f}",
                f"{mean_mac_f1:.2f}", f"{mean_wtd_f1:.2f}",
                "—", f"{global_mean:.1f}", "—", "—", "—", "—", "—",
            ])
            writer.writerow([
                "STD", "—",
                f"{std_top1:.2f}", "—",
                f"{std_mac_f1:.2f}", "—",
                "—", "—", "—", "—", "—", "—", "—",
            ])
        print(f"  Round CSV saved: {round_csv}")

        # Append to global summary
        with open(global_csv, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                round_idx + 1, cap,
                nominal_per_word,
                f"{global_mean:.1f}",
                global_min,
                global_max,
                f"{mean_top1:.2f}", f"{std_top1:.2f}",
                f"{mean_top5:.2f}",
                f"{mean_mac_f1:.2f}", f"{std_mac_f1:.2f}",
                f"{mean_wtd_f1:.2f}",
            ])

        round_summaries.append({
            "round"               : round_idx + 1,
            "cap"                 : cap,
            "nominal_per_word"    : nominal_per_word,
            "actual_mean_per_word": global_mean,
            "mean_top1"           : mean_top1,
            "std_top1"            : std_top1,
            "mean_top5"           : mean_top5,
            "mean_macro_f1"       : mean_mac_f1,
            "mean_weighted_f1"    : mean_wtd_f1,
        })

    # ============================================================
    # FINAL SUMMARY + CURVE
    # ============================================================

    print("\n\n" + "=" * 65)
    print("  RQ2 COMPLETE — FINAL SUMMARY")
    print("=" * 65)
    print(f"\n  {'Cap':<5} {'Nominal/word':<14} {'Actual/word':<13} "
          f"{'Top-1':>9} {'±std':>7} {'Top-5':>9} {'MacroF1':>9}")
    print(f"  {'-'*70}")
    for s in round_summaries:
        print(
            f"  {s['cap']:<5} {s['nominal_per_word']:<14} "
            f"{s['actual_mean_per_word']:<13.1f} "
            f"{s['mean_top1']:7.2f}%  "
            f"±{s['std_top1']:.2f}%  "
            f"{s['mean_top5']:7.2f}%  "
            f"{s['mean_macro_f1']:7.2f}%"
        )

    plot_rq2_curve(
        round_summaries,
        save_path=os.path.join(args.results_dir, "rq2_accuracy_curve.png"),
    )

    print(f"\n  Global summary CSV : {global_csv}")
    print(f"  All results        : {args.results_dir}/")
    print(f"  TensorBoard        : "
          f"tensorboard --logdir={args.results_dir}/runs")
    print("=" * 65)


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RQ2: Progressive training data reduction under LOSO"
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
        help="Starting cap (max reps per word per user, default=10)",
    )
    parser.add_argument(
        "--cap_end",     type=int, default=CAP_END,
        help="Ending cap (min reps per word per user, default=1)",
    )
    args = parser.parse_args()
    main(args)
