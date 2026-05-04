"""
analyze_grouped_kfold.py  —  Post-hoc Analysis for Grouped 5-Fold Results
==========================================================================
Reads results_grouped/ and produces the same style of outputs as the
vocab-partitioned LOSO analysis.

CSVs (saved to results_grouped/analysis/):
    per_fold_accuracy.csv     — Top1/Top5/F1/Loss per fold (train + test)
    per_word_accuracy.csv     — per-word accuracy sorted hardest first
    per_user_accuracy.csv     — per-user accuracy averaged across folds
    global_accuracy.csv       — final aggregated numbers

Plots:
    01_fold_comparison.png    — bar chart: train vs test per fold
    02_global_summary.png     — final summary bar chart with std
    03_per_user_heatmap.png   — user × fold accuracy heatmap
    04_difficult_words.png    — bottom-30 hardest words
    05_word_accuracy_dist.png — histogram of word accuracy distribution
    06_fold_boxplot.png       — per-user accuracy box plot per fold
    07_train_vs_test.png      — train vs test top1 line chart across folds

Run (from results_grouped/ parent folder):
    python analyze_grouped_kfold.py
"""

import os
import csv
import glob
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from collections import defaultdict

# ── Paths ──────────────────────────────────────────────────────────────────
RESULTS_DIR  = "results_grouped"
ANALYSIS_DIR = os.path.join(RESULTS_DIR, "analysis")
os.makedirs(ANALYSIS_DIR, exist_ok=True)

KFOLD_CSV = os.path.join(RESULTS_DIR, "kfold_summary.csv")

# ── Plot style ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family"      : "DejaVu Sans",
    "axes.spines.top"  : False,
    "axes.spines.right": False,
    "axes.grid"        : True,
    "grid.alpha"       : 0.3,
    "figure.dpi"       : 120,
})
PALETTE = ["#2563EB", "#10B981", "#F59E0B", "#EF4444", "#8B5CF6"]

# ═══════════════════════════════════════════════════════════════════════════
# LOAD KFOLD SUMMARY
# ═══════════════════════════════════════════════════════════════════════════
print("Loading kfold_summary.csv ...")
kfold_df = pd.read_csv(KFOLD_CSV)
print(kfold_df.to_string(index=False))

NUM_FOLDS = len(kfold_df)

# ═══════════════════════════════════════════════════════════════════════════
# LOAD PER-WORD ACCURACY FROM ALL FOLD DIRS
# ═══════════════════════════════════════════════════════════════════════════
print("\nLoading per_class_accuracy.csv from each fold ...")

word_metrics = defaultdict(list)   # word → list of accuracy values per fold

for fold_idx in range(NUM_FOLDS):
    path = os.path.join(RESULTS_DIR, f"fold_{fold_idx}", "per_class_accuracy.csv")
    if not os.path.exists(path):
        print(f"  WARNING: {path} not found, skipping.")
        continue
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            word_metrics[row["class"]].append(float(row["accuracy(%)"]))

print(f"  Found {len(word_metrics)} unique words across folds.")

# ═══════════════════════════════════════════════════════════════════════════
# LOAD PER-USER ACCURACY FROM ALL FOLD DIRS
# ═══════════════════════════════════════════════════════════════════════════
print("\nLoading per_user_accuracy.csv from each fold ...")

user_fold_acc = defaultdict(dict)   # user → fold → accuracy

for fold_idx in range(NUM_FOLDS):
    path = os.path.join(RESULTS_DIR, f"fold_{fold_idx}", "per_user_accuracy.csv")
    if not os.path.exists(path):
        print(f"  WARNING: {path} not found, skipping.")
        continue
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            user_fold_acc[row["user"]][fold_idx] = float(row["top1_accuracy(%)"])

# ═══════════════════════════════════════════════════════════════════════════
# 1. PER-FOLD ACCURACY CSV
# ═══════════════════════════════════════════════════════════════════════════
print("\nGenerating per_fold_accuracy.csv ...")

per_fold = []
for _, row in kfold_df.iterrows():
    per_fold.append({
        "fold"        : int(row["fold"]),
        "train_top1"  : float(row["train_top1"]),
        "train_top5"  : float(row["train_top5"]),
        "test_top1"   : float(row["test_top1"]),
        "test_top5"   : float(row["test_top5"]),
        "macro_f1"    : float(row["macro_f1"]),
        "weighted_f1" : float(row["weighted_f1"]),
        "test_loss"   : float(row["test_loss"]),
        "gap_top1"    : float(row["train_top1"]) - float(row["test_top1"]),
    })

per_fold_df = pd.DataFrame(per_fold)
per_fold_df.to_csv(
    os.path.join(ANALYSIS_DIR, "per_fold_accuracy.csv"),
    index=False, float_format="%.4f"
)
print(f"  Saved: per_fold_accuracy.csv")

# ═══════════════════════════════════════════════════════════════════════════
# 2. PER-WORD ACCURACY CSV
# ═══════════════════════════════════════════════════════════════════════════
print("Generating per_word_accuracy.csv ...")

word_rows = []
for word, accs in sorted(word_metrics.items(), key=lambda x: np.mean(x[1])):
    word_rows.append({
        "word"           : word,
        "accuracy_pct"   : round(np.mean(accs), 2),
        "std_pct"        : round(np.std(accs), 2),
        "num_folds"      : len(accs),
    })

word_df = pd.DataFrame(word_rows)
word_df["rank_hardest"] = range(1, len(word_df) + 1)
word_df.to_csv(
    os.path.join(ANALYSIS_DIR, "per_word_accuracy.csv"),
    index=False, float_format="%.2f"
)
print(f"  Saved: per_word_accuracy.csv  ({len(word_df)} words)")

# ═══════════════════════════════════════════════════════════════════════════
# 3. PER-USER ACCURACY CSV
# ═══════════════════════════════════════════════════════════════════════════
print("Generating per_user_accuracy.csv ...")

user_rows = []
for user in sorted(user_fold_acc.keys()):
    accs = list(user_fold_acc[user].values())
    user_rows.append({
        "user"        : user,
        "mean_top1"   : round(np.mean(accs), 2),
        "std_top1"    : round(np.std(accs), 2),
        "min_top1"    : round(np.min(accs), 2),
        "max_top1"    : round(np.max(accs), 2),
        "num_folds"   : len(accs),
    })

user_df = pd.DataFrame(user_rows).sort_values("mean_top1")
user_df.to_csv(
    os.path.join(ANALYSIS_DIR, "per_user_accuracy.csv"),
    index=False, float_format="%.2f"
)
print(f"  Saved: per_user_accuracy.csv  ({len(user_df)} users)")

# ═══════════════════════════════════════════════════════════════════════════
# 4. GLOBAL ACCURACY CSV
# ═══════════════════════════════════════════════════════════════════════════
print("Generating global_accuracy.csv ...")

global_acc = {
    "num_folds"              : NUM_FOLDS,
    "num_words"              : len(word_df),
    "num_users"              : len(user_df),

    "mean_test_top1"         : round(kfold_df["test_top1"].mean(), 4),
    "std_test_top1"          : round(kfold_df["test_top1"].std(), 4),
    "min_test_top1"          : round(kfold_df["test_top1"].min(), 4),
    "max_test_top1"          : round(kfold_df["test_top1"].max(), 4),

    "mean_test_top5"         : round(kfold_df["test_top5"].mean(), 4),
    "std_test_top5"          : round(kfold_df["test_top5"].std(), 4),

    "mean_macro_f1"          : round(kfold_df["macro_f1"].mean(), 4),
    "std_macro_f1"           : round(kfold_df["macro_f1"].std(), 4),

    "mean_weighted_f1"       : round(kfold_df["weighted_f1"].mean(), 4),
    "mean_test_loss"         : round(kfold_df["test_loss"].mean(), 4),

    "mean_train_test_gap"    : round((kfold_df["train_top1"] - kfold_df["test_top1"]).mean(), 4),

    "word_mean_accuracy"     : round(word_df["accuracy_pct"].mean(), 4),
    "word_std_accuracy"      : round(word_df["accuracy_pct"].std(), 4),
    "pct_words_above_90"     : round((word_df["accuracy_pct"] >= 90).mean() * 100, 2),
    "pct_words_below_70"     : round((word_df["accuracy_pct"] < 70).mean() * 100, 2),

    "hardest_user"           : user_df.iloc[0]["user"],
    "hardest_user_mean_top1" : user_df.iloc[0]["mean_top1"],
    "easiest_user"           : user_df.iloc[-1]["user"],
    "easiest_user_mean_top1" : user_df.iloc[-1]["mean_top1"],
}

with open(os.path.join(ANALYSIS_DIR, "global_accuracy.csv"), "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["metric", "value"])
    for k, v in global_acc.items():
        writer.writerow([k, v])
print(f"  Saved: global_accuracy.csv")

# ═══════════════════════════════════════════════════════════════════════════
# VISUALIZATIONS
# ═══════════════════════════════════════════════════════════════════════════

folds      = kfold_df["fold"].tolist()
test_top1  = kfold_df["test_top1"].tolist()
test_top5  = kfold_df["test_top5"].tolist()
macro_f1   = kfold_df["macro_f1"].tolist()
train_top1 = kfold_df["train_top1"].tolist()

# ── 01: Fold Comparison Bar Chart ─────────────────────────────────────────
print("\nPlot 01: Fold comparison bar chart...")

x     = np.arange(NUM_FOLDS)
width = 0.22
metrics_plot = [
    (train_top1, "#93C5FD", "Train Top-1 (%)"),
    (test_top1,  "#2563EB", "Test Top-1 (%)"),
    (macro_f1,   "#10B981", "Macro F1 (%)"),
    (test_top5,  "#F59E0B", "Test Top-5 (%)"),
]

fig, ax = plt.subplots(figsize=(13, 6))
for i, (vals, color, label) in enumerate(metrics_plot):
    bars = ax.bar(x + i * width, vals, width, label=label,
                  color=color, alpha=0.88)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.1,
                f"{v:.1f}", ha="center", va="bottom", fontsize=7.5)

ax.set_xticks(x + width * 1.5)
ax.set_xticklabels([f"Fold {f}" for f in folds])
ax.set_ylabel("(%)")
ax.set_title("Per-Fold Performance: Train Top-1, Test Top-1, Macro F1, Test Top-5",
             fontsize=12, fontweight="bold")
ax.legend(loc="lower right", fontsize=9)
ax.set_ylim(88, 101)
plt.tight_layout()
plt.savefig(os.path.join(ANALYSIS_DIR, "01_fold_comparison.png"), dpi=130)
plt.close()
print("  Saved: 01_fold_comparison.png")

# ── 02: Global Summary Bar Chart ──────────────────────────────────────────
print("Plot 02: Global summary bar chart...")

g_labels = ["Test Top-1", "Test Top-5", "Macro F1", "Weighted F1"]
g_means  = [
    global_acc["mean_test_top1"],
    global_acc["mean_test_top5"],
    global_acc["mean_macro_f1"],
    global_acc["mean_weighted_f1"],
]
g_stds = [
    global_acc["std_test_top1"],
    global_acc["std_test_top5"],
    global_acc["std_macro_f1"],
    0,
]
g_colors = ["#2563EB", "#F59E0B", "#10B981", "#8B5CF6"]

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.bar(g_labels, g_means, color=g_colors, alpha=0.88,
              yerr=g_stds, capsize=6,
              error_kw={"elinewidth": 2, "ecolor": "black"})
for bar, v, s in zip(bars, g_means, g_stds):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + s + 0.1,
            f"{v:.2f}%", ha="center", va="bottom",
            fontsize=11, fontweight="bold")

ax.set_ylabel("(%)")
ax.set_title("Global Performance Summary\n(Mean ± Std across 5 folds)",
             fontsize=13, fontweight="bold")
ax.set_ylim(88, 102)
plt.tight_layout()
plt.savefig(os.path.join(ANALYSIS_DIR, "02_global_summary.png"), dpi=130)
plt.close()
print("  Saved: 02_global_summary.png")

# ── 03: Per-User Heatmap ──────────────────────────────────────────────────
print("Plot 03: Per-user heatmap...")

# Build pivot: user × fold
all_users = sorted(user_fold_acc.keys())
pivot_data = {}
for user in all_users:
    pivot_data[user] = {f: user_fold_acc[user].get(f, np.nan)
                        for f in range(NUM_FOLDS)}

pivot_df = pd.DataFrame(pivot_data).T
pivot_df.columns = [f"Fold {f}" for f in range(NUM_FOLDS)]
pivot_df["Mean"] = pivot_df.mean(axis=1)
pivot_df = pivot_df.sort_values("Mean", ascending=False)

fig, ax = plt.subplots(figsize=(10, max(6, len(all_users) * 0.5)))
sns.heatmap(
    pivot_df, annot=True, fmt=".1f", cmap="RdYlGn",
    vmin=80, vmax=100, linewidths=0.5, linecolor="white",
    ax=ax, annot_kws={"size": 8},
    cbar_kws={"label": "Top-1 Accuracy (%)"},
)
ax.set_title("Per-User Top-1 Accuracy Across Folds\n(sorted by mean accuracy)",
             fontsize=12, fontweight="bold")
ax.set_xlabel("Fold")
ax.set_ylabel("User")
plt.tight_layout()
plt.savefig(os.path.join(ANALYSIS_DIR, "03_per_user_heatmap.png"), dpi=130)
plt.close()
print("  Saved: 03_per_user_heatmap.png")

# ── 04: Difficult Words (bottom 30) ──────────────────────────────────────
print("Plot 04: Difficult words bar chart...")

bottom30 = word_df.head(30)
bar_colors = ["#EF4444" if v < 70 else "#F59E0B" if v < 85 else "#10B981"
              for v in bottom30["accuracy_pct"]]

fig, ax = plt.subplots(figsize=(12, 7))
bars = ax.barh(bottom30["word"], bottom30["accuracy_pct"],
               color=bar_colors, alpha=0.85,
               xerr=bottom30["std_pct"], capsize=3,
               error_kw={"elinewidth": 1.2, "ecolor": "gray"})

ax.axvline(70, color="#EF4444", linestyle="--", linewidth=1.2, label="70% threshold")
ax.axvline(85, color="#F59E0B", linestyle="--", linewidth=1.2, label="85% threshold")

for bar, v in zip(bars, bottom30["accuracy_pct"]):
    ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
            f"{v:.1f}%", va="center", fontsize=8)

ax.set_xlabel("Accuracy (%)")
ax.set_title("30 Most Difficult Words\n(error bars = std across folds; red<70%, orange<85%, green≥85%)",
             fontsize=12, fontweight="bold")
ax.legend(loc="lower right")
ax.set_xlim(0, 110)
plt.tight_layout()
plt.savefig(os.path.join(ANALYSIS_DIR, "04_difficult_words.png"), dpi=130)
plt.close()
print("  Saved: 04_difficult_words.png")

# ── 05: Word Accuracy Distribution ───────────────────────────────────────
print("Plot 05: Word accuracy distribution...")

fig, ax = plt.subplots(figsize=(10, 5))
n, bins, patches = ax.hist(word_df["accuracy_pct"], bins=20,
                           edgecolor="white", alpha=0.85)
for patch, left in zip(patches, bins[:-1]):
    patch.set_facecolor(
        "#EF4444" if left < 70 else "#F59E0B" if left < 85 else "#10B981"
    )

ax.axvline(word_df["accuracy_pct"].mean(), color="black",
           linestyle="--", linewidth=2,
           label=f"Mean = {word_df['accuracy_pct'].mean():.1f}%")
ax.set_xlabel("Word-Level Accuracy (%)")
ax.set_ylabel("Number of Words")
ax.set_title("Distribution of Per-Word Accuracy\n(red<70%, orange<85%, green≥85%)",
             fontsize=12, fontweight="bold")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(ANALYSIS_DIR, "05_word_accuracy_dist.png"), dpi=130)
plt.close()
print("  Saved: 05_word_accuracy_dist.png")

# ── 06: Per-User Accuracy Box Plot per Fold ───────────────────────────────
print("Plot 06: Per-user accuracy box plot per fold...")

fold_user_data = []
for f in range(NUM_FOLDS):
    accs = [user_fold_acc[u][f] for u in all_users if f in user_fold_acc[u]]
    fold_user_data.append(accs)

fig, ax = plt.subplots(figsize=(9, 5))
bp = ax.boxplot(fold_user_data, patch_artist=True, notch=False,
                medianprops={"color": "black", "linewidth": 2})
for patch, color in zip(bp["boxes"], PALETTE):
    patch.set_facecolor(color)
    patch.set_alpha(0.75)

ax.set_xticklabels([f"Fold {f}" for f in range(NUM_FOLDS)])
ax.set_ylabel("Per-User Top-1 Accuracy (%)")
ax.set_title("Per-User Accuracy Distribution Across Folds\n(each point = one user)",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(ANALYSIS_DIR, "06_fold_boxplot.png"), dpi=130)
plt.close()
print("  Saved: 06_fold_boxplot.png")

# ── 07: Train vs Test Top-1 Line Chart ───────────────────────────────────
print("Plot 07: Train vs test top-1 line chart...")

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(folds, train_top1, marker="o", color="#93C5FD",
        linewidth=2.5, markersize=8, label="Train Top-1")
ax.plot(folds, test_top1,  marker="s", color="#2563EB",
        linewidth=2.5, markersize=8, label="Test Top-1")
ax.fill_between(folds, train_top1, test_top1,
                alpha=0.12, color="#2563EB", label="Train-Test Gap")

for x, tr, te in zip(folds, train_top1, test_top1):
    ax.annotate(f"{tr:.2f}%", (x, tr), textcoords="offset points",
                xytext=(0, 8), ha="center", fontsize=8, color="#1D4ED8")
    ax.annotate(f"{te:.2f}%", (x, te), textcoords="offset points",
                xytext=(0, -14), ha="center", fontsize=8, color="#1D4ED8")

ax.set_xticks(folds)
ax.set_xticklabels([f"Fold {f}" for f in folds])
ax.set_ylabel("Top-1 Accuracy (%)")
ax.set_title(f"Train vs Test Top-1 per Fold\n"
             f"Mean gap = {global_acc['mean_train_test_gap']:.2f}%  "
             f"(Train − Test)",
             fontsize=12, fontweight="bold")
ax.legend()
ax.set_ylim(93, 97)
plt.tight_layout()
plt.savefig(os.path.join(ANALYSIS_DIR, "07_train_vs_test.png"), dpi=130)
plt.close()
print("  Saved: 07_train_vs_test.png")

# ═══════════════════════════════════════════════════════════════════════════
# FINAL PRINT
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 65)
print("  ANALYSIS COMPLETE")
print("=" * 65)
print(f"\n  Test Top-1   : {global_acc['mean_test_top1']:.2f}% ± {global_acc['std_test_top1']:.2f}%")
print(f"  Test Top-5   : {global_acc['mean_test_top5']:.2f}% ± {global_acc['std_test_top5']:.2f}%")
print(f"  Macro F1     : {global_acc['mean_macro_f1']:.2f}% ± {global_acc['std_macro_f1']:.2f}%")
print(f"  Train-Test Gap : {global_acc['mean_train_test_gap']:.2f}%")
print(f"\n  Words > 90%  : {global_acc['pct_words_above_90']:.1f}%")
print(f"  Words < 70%  : {global_acc['pct_words_below_70']:.1f}%")
print(f"\n  Hardest user : {global_acc['hardest_user']}  ({global_acc['hardest_user_mean_top1']:.2f}%)")
print(f"  Easiest user : {global_acc['easiest_user']}  ({global_acc['easiest_user_mean_top1']:.2f}%)")
print(f"\n  All outputs  : {ANALYSIS_DIR}/")
print("=" * 65)
