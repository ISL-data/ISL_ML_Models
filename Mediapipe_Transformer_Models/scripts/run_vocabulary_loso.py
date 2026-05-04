"""
vocab_partition_runner.py  —  Vocabulary-Partitioned LOSO Evaluation
======================================================================
Experiment:
    The full vocabulary (489 classes) is randomly partitioned into
    NUM_BUCKETS disjoint subsets. For each bucket, a complete 15-fold
    LOSO evaluation is performed using only the words in that bucket.
    This is repeated NUM_ROUNDS times with different random seeds to
    estimate performance variability due to vocabulary composition.

Protocol:
    - NUM_ROUNDS  : 5  (different random partitions)
    - NUM_BUCKETS : 5  (disjoint subsets per round)
    - LOSO folds  : 15 (one per user)
    - ~98 words per bucket (last bucket gets remainder)

Outputs (all saved to RESULTS_DIR):
    config.json                      — full config for reproducibility
    round_X_buckets.csv              — exact word-to-bucket assignment per round
    global_summary.csv               — per-fold metrics for every run
    round_summary.csv                — per-round aggregated metrics
    difficult_words.csv              — per-word accuracy across all rounds
    cm_rX_bY.png                     — aggregated confusion matrix per bucket
    loss_rX_bY.png                   — mean loss curve per bucket
    top1_rX_bY.png                   — mean top-1 curve per bucket
    top5_rX_bY.png                   — mean top-5 curve per bucket

Run:
    python vocab_partition_runner.py
"""

import os
import sys
import csv
import json
import random
import numpy as np
import torch
from datetime import datetime
from collections import defaultdict

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.dataset_class import find_h5_files, label_from_filename, user_from_path
from src.train import train_one_fold, Tee, save_confusion_matrix

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# CONFIG  (matches main runner.py exactly)
# ============================================================

DATA_ROOT   = "/mnt/9a528fe4-4fe8-4dff-9a0c-8b1a3cf3d7ba/popsign/ISL_Goa_Data_h5_30fps"
RESULTS_DIR = "results_vocab_partition"

NUM_BUCKETS = 5
NUM_ROUNDS  = 5

config = {
    "max_frames"      : 150,
    "batch_size"      : 32,
    "epochs"          : 50,
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
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ============================================================
# DEVICE
# ============================================================

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# ============================================================
# VOCABULARY PARTITIONING
# ============================================================

def create_buckets(classes, k, seed):
    """
    Randomly partition `classes` into k disjoint subsets.

    Uses numpy's default_rng for reproducible, seed-controlled
    shuffling independent of global random state.

    Parameters
    ----------
    classes : list  — full sorted class list
    k       : int   — number of buckets
    seed    : int   — random seed for this partition

    Returns
    -------
    list of k lists, each containing approximately len(classes)//k words.
    The last bucket absorbs the remainder.
    """
    rng = np.random.default_rng(seed)
    classes = list(classes)
    rng.shuffle(classes)

    n = len(classes)
    base = n // k          # 97
    rem  = n % k           # 4

    sizes = [base + 1 if i < rem else base for i in range(k)]
    # → [98, 98, 98, 98, 97]

    buckets = []
    start = 0

    for size in sizes:
        end = start + size
        buckets.append(classes[start:end])
        start = end

    return buckets



# ============================================================
# CURVE HELPER
# ============================================================

def pad_to(lst, length):
    """Pad list to length by repeating last value."""
    return lst + [lst[-1]] * (length - len(lst))


def save_mean_curve(fold_results, key_train, key_test,
                    ylabel, title, save_path):
    """
    Plot mean train/test curve across all folds with individual
    fold traces shown faintly in the background.
    """
    max_ep     = max(len(r[key_train]) for r in fold_results)
    mean_train = np.mean([pad_to(r[key_train], max_ep) for r in fold_results], axis=0)
    mean_test  = np.mean([pad_to(r[key_test],  max_ep) for r in fold_results], axis=0)

    plt.figure(figsize=(10, 5))
    for r in fold_results:
        ep = list(range(len(r[key_train])))
        plt.plot(ep, r[key_train], alpha=0.15, color="steelblue")
        plt.plot(ep, r[key_test],  alpha=0.15, color="coral")

    plt.plot(mean_train, color="steelblue", linewidth=2, label="Mean Train")
    plt.plot(mean_test,  color="coral",     linewidth=2, label="Mean Test")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close()


# ============================================================
# MAIN
# ============================================================

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ── Save config for full reproducibility ──────────────────
    with open(os.path.join(RESULTS_DIR, "config.json"), "w") as f:
        json.dump({**config,
                   "NUM_BUCKETS": NUM_BUCKETS,
                   "NUM_ROUNDS" : NUM_ROUNDS,
                   "DATA_ROOT"  : DATA_ROOT},
                  f, indent=4)

    # ── Logging ───────────────────────────────────────────────
    ts    = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_f = open(os.path.join(RESULTS_DIR, f"log_{ts}.txt"), "w")
    sys.stdout = Tee(sys.stdout, log_f)
    sys.stderr = Tee(sys.stderr, log_f)

    print("=" * 70)
    print("  VOCABULARY-PARTITIONED LOSO EVALUATION")
    print(f"  Rounds      : {NUM_ROUNDS}")
    print(f"  Buckets     : {NUM_BUCKETS}")
    print(f"  Device      : {device}")
    print(f"  Data root   : {DATA_ROOT}")
    print(f"  Results dir : {RESULTS_DIR}")
    print("=" * 70)

    # ── Data discovery ────────────────────────────────────────
    all_paths = find_h5_files(DATA_ROOT)
    if not all_paths:
        raise RuntimeError(f"No .h5 files found under: {DATA_ROOT}")

    user_to_paths = defaultdict(list)
    for p in all_paths:
        user_to_paths[user_from_path(p)].append(p)

    users       = sorted(user_to_paths.keys())
    all_labels  = [label_from_filename(p) for p in all_paths]
    all_classes = sorted(set(all_labels))

    print(f"\n  Total files   : {len(all_paths)}")
    print(f"  Total classes : {len(all_classes)}")
    print(f"  Total users   : {len(users)}  →  {users}")

    # ── CSV headers ───────────────────────────────────────────
    global_csv = os.path.join(RESULTS_DIR, "global_summary.csv")
    with open(global_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "round", "bucket", "fold", "test_user",
            "train_top1(%)", "train_top5(%)",
            "test_top1(%)", "test_top5(%)",
            "macro_f1(%)", "weighted_f1(%)", "test_loss",
        ])

    round_csv = os.path.join(RESULTS_DIR, "round_summary.csv")
    with open(round_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "round",
            "mean_top1(%)", "mean_top5(%)", "mean_macro_f1(%)",
            "std_top1(%)",  "std_top5(%)",  "std_macro_f1(%)",
        ])

    # ── Per-word tracking across all rounds ───────────────────
    word_metrics = defaultdict(list)  # word → list of 0/1 correct predictions

    round_results = []   # (mean_top1, mean_top5, mean_f1) per round

    # ============================================================
    # ROUND LOOP
    # ============================================================

    for r_idx in range(NUM_ROUNDS):
        round_seed = config["seed"] + r_idx
        set_seed(round_seed)

        print(f"\n\n{'='*70}")
        print(f"  ROUND {r_idx + 1} / {NUM_ROUNDS}  |  seed={round_seed}")
        print(f"{'='*70}")

        # ── Partition vocabulary ──────────────────────────────
        buckets = create_buckets(all_classes, NUM_BUCKETS, round_seed)

        print(f"\n  Bucket sizes: {[len(b) for b in buckets]}")

        # Save bucket assignment for reproducibility
        bucket_csv = os.path.join(RESULTS_DIR, f"round_{r_idx}_buckets.csv")
        with open(bucket_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["bucket_id", "num_words"] +
                            [f"word_{i}" for i in range(max(len(b) for b in buckets))])
            for i, b in enumerate(buckets):
                writer.writerow([f"bucket_{i}", len(b)] + b)

        bucket_top1s = []
        bucket_top5s = []
        bucket_f1s   = []

        # ============================================================
        # BUCKET LOOP
        # ============================================================

        for b_idx, bucket in enumerate(buckets):
            bucket_set = set(bucket)

            print(f"\n  --- Round {r_idx}  Bucket {b_idx}  "
                  f"({len(bucket)} words) ---")

            # Filter paths to this bucket's vocabulary
            filtered = [
                p for p in all_paths
                if label_from_filename(p) in bucket_set
            ]

            if not filtered:
                print(f"   No files found for bucket {b_idx}. Skipping.")
                continue

            b_user_map = defaultdict(list)
            for p in filtered:
                b_user_map[user_from_path(p)].append(p)

            # Label encoder fitted on this bucket's classes only
            le = LabelEncoder()
            le.fit(sorted(bucket))

            fold_results = []

            print(f"\n  {'Fold':<5} {'User':<20} {'TrTop1':>7} {'TrTop5':>7} "
                  f"{'TeTop1':>7} {'TeTop5':>7} {'MacroF1':>8} {'WtdF1':>7}")
            print(f"  {'-'*72}")

            # ========================================================
            # FOLD LOOP  (LOSO)
            # ========================================================

            for f_idx, test_user in enumerate(users):
                train_users = [u for u in users if u != test_user]
                train_paths = [p for u in train_users for p in b_user_map[u]]
                test_paths  = b_user_map[test_user]

                if not train_paths or not test_paths:
                    print(f"  {f_idx:<5} {test_user:<20} SKIPPED "
                          f"(train={len(train_paths)} test={len(test_paths)})")
                    continue

                fold_log_dir = os.path.join(
                    RESULTS_DIR, f"r{r_idx}_b{b_idx}_f{f_idx}"
                )

                result = train_one_fold(
                    fold_idx      = f_idx,
                    train_paths   = train_paths,
                    test_paths    = test_paths,
                    label_encoder = le,
                    class_names   = list(le.classes_),
                    fold_log_dir  = fold_log_dir,
                    device        = device,
                    config        = config,
                    tb_root       = os.path.join(RESULTS_DIR, "runs"),
                )

                result["test_user"]   = test_user
                result["train_users"] = train_users
                fold_results.append(result)

                print(
                    f"  {f_idx:<5} {test_user:<20} "
                    f"{result['train_top1_acc']:6.2f}%  "
                    f"{result['train_top5_acc']:6.2f}%  "
                    f"{result['top1_acc']:6.2f}%  "
                    f"{result['top5_acc']:6.2f}%  "
                    f"{result['macro_f1']:7.2f}%  "
                    f"{result['weighted_f1']:6.2f}%"
                )

                # Append to global CSV
                with open(global_csv, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        r_idx, b_idx, f_idx, test_user,
                        f"{result['train_top1_acc']:.2f}",
                        f"{result['train_top5_acc']:.2f}",
                        f"{result['top1_acc']:.2f}",
                        f"{result['top5_acc']:.2f}",
                        f"{result['macro_f1']:.2f}",
                        f"{result['weighted_f1']:.2f}",
                        f"{result['test_loss']:.4f}",
                    ])

                # Per-word accuracy tracking
                for y_t, y_p in zip(result["y_true"], result["y_pred"]):
                    word = le.inverse_transform([y_t])[0]
                    word_metrics[word].append(int(y_t == y_p))

            if not fold_results:
                print(f"   No fold results for bucket {b_idx}. Skipping plots.")
                continue

            # ── Aggregated confusion matrix for this bucket ───
            y_true_all = np.concatenate([r["y_true"] for r in fold_results])
            y_pred_all = np.concatenate([r["y_pred"] for r in fold_results])

            cm = confusion_matrix(
                y_true_all, y_pred_all,
                labels=list(range(len(le.classes_)))   # always square
            )
            save_confusion_matrix(
                cm,
                list(le.classes_),
                os.path.join(RESULTS_DIR, f"cm_r{r_idx}_b{b_idx}.png"),
                title=f"Confusion Matrix — Round {r_idx}  Bucket {b_idx}",
            )

            # ── Mean curves for this bucket ───────────────────
            save_mean_curve(
                fold_results, "train_losses", "test_losses",
                ylabel="Loss",
                title=f"Mean Loss — Round {r_idx}  Bucket {b_idx}",
                save_path=os.path.join(RESULTS_DIR, f"loss_r{r_idx}_b{b_idx}.png"),
            )
            save_mean_curve(
                fold_results, "train_top1s", "test_top1s",
                ylabel="Top-1 Accuracy (%)",
                title=f"Mean Top-1 — Round {r_idx}  Bucket {b_idx}",
                save_path=os.path.join(RESULTS_DIR, f"top1_r{r_idx}_b{b_idx}.png"),
            )
            save_mean_curve(
                fold_results, "train_top5s", "test_top5s",
                ylabel="Top-5 Accuracy (%)",
                title=f"Mean Top-5 — Round {r_idx}  Bucket {b_idx}",
                save_path=os.path.join(RESULTS_DIR, f"top5_r{r_idx}_b{b_idx}.png"),
            )

            # ── Bucket summary ────────────────────────────────
            b_top1 = np.mean([r["top1_acc"]  for r in fold_results])
            b_top5 = np.mean([r["top5_acc"]  for r in fold_results])
            b_f1   = np.mean([r["macro_f1"]  for r in fold_results])

            bucket_top1s.append(b_top1)
            bucket_top5s.append(b_top5)
            bucket_f1s.append(b_f1)

            print(f"\n  Bucket {b_idx} summary → "
                  f"Top1={b_top1:.2f}%  Top5={b_top5:.2f}%  F1={b_f1:.2f}%")

        # ── Round summary ─────────────────────────────────────
        if not bucket_top1s:
            print(f"\n   Round {r_idx} had no valid buckets.")
            continue

        r_top1 = np.mean(bucket_top1s)
        r_top5 = np.mean(bucket_top5s)
        r_f1   = np.mean(bucket_f1s)

        r_std_top1 = np.std(bucket_top1s)
        r_std_top5 = np.std(bucket_top5s)
        r_std_f1   = np.std(bucket_f1s)

        round_results.append((r_top1, r_top5, r_f1))

        with open(round_csv, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                r_idx,
                f"{r_top1:.2f}", f"{r_top5:.2f}", f"{r_f1:.2f}",
                f"{r_std_top1:.2f}", f"{r_std_top5:.2f}", f"{r_std_f1:.2f}",
            ])

        print(f"\n  Round {r_idx} summary → "
              f"Top1={r_top1:.2f}±{r_std_top1:.2f}%  "
              f"Top5={r_top5:.2f}±{r_std_top5:.2f}%  "
              f"F1={r_f1:.2f}±{r_std_f1:.2f}%")

    # ============================================================
    # FINAL AGGREGATED RESULTS
    # ============================================================

    if not round_results:
        print("\n   No round results to aggregate.")
        return

    final_top1     = np.mean([r[0] for r in round_results])
    final_top5     = np.mean([r[1] for r in round_results])
    final_f1       = np.mean([r[2] for r in round_results])
    final_std_top1 = np.std([r[0] for r in round_results])
    final_std_top5 = np.std([r[1] for r in round_results])
    final_std_f1   = np.std([r[2] for r in round_results])

    print("\n\n" + "=" * 70)
    print("  FINAL RESULTS  (aggregated across all rounds)")
    print("=" * 70)
    print(f"  Top-1 Accuracy : {final_top1:.2f}% ± {final_std_top1:.2f}%")
    print(f"  Top-5 Accuracy : {final_top5:.2f}% ± {final_std_top5:.2f}%")
    print(f"  Macro F1       : {final_f1:.2f}% ± {final_std_f1:.2f}%")

    # Save final summary
    with open(os.path.join(RESULTS_DIR, "final_summary.txt"), "w") as f:
        f.write(f"Vocabulary-Partitioned LOSO — Final Results\n")
        f.write(f"Rounds  : {NUM_ROUNDS}\n")
        f.write(f"Buckets : {NUM_BUCKETS}\n")
        f.write(f"Folds   : {len(users)}\n\n")
        f.write(f"Top-1 Accuracy : {final_top1:.2f}% ± {final_std_top1:.2f}%\n")
        f.write(f"Top-5 Accuracy : {final_top5:.2f}% ± {final_std_top5:.2f}%\n")
        f.write(f"Macro F1       : {final_f1:.2f}% ± {final_std_f1:.2f}%\n")

    # ============================================================
    # DIFFICULT WORDS  (sorted by ascending accuracy)
    # ============================================================

    difficult_csv = os.path.join(RESULTS_DIR, "difficult_words.csv")
    with open(difficult_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["word", "accuracy(%)", "num_evaluations"])

        for word, vals in sorted(
            word_metrics.items(),
            key=lambda x: np.mean(x[1])   # ascending — hardest first
        ):
            writer.writerow([
                word,
                f"{np.mean(vals) * 100:.2f}",
                len(vals),
            ])

    print(f"\n   Difficult words saved: {difficult_csv}")
    print(f"   All results saved to : {RESULTS_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
