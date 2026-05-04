"""
analyze_results.py  —  Post-hoc Analysis & Report Generation
=============================================================
Reads CSVs from results_vocab_partition/ and produces:

CSVs:
    analysis/per_bucket_accuracy.csv   — Top1/Top5/F1 per round×bucket
    analysis/per_word_accuracy.csv     — per-word accuracy (from difficult_words.csv)
    analysis/global_accuracy.csv       — final aggregated numbers

Visualizations:
    analysis/01_round_comparison.png   — bar chart per round with error bars
    analysis/02_global_summary.png     — final summary bar chart
    analysis/03_per_user_heatmap.png   — user × round accuracy heatmap
    analysis/04_difficult_words.png    — bottom-30 hardest words
    analysis/05_word_accuracy_dist.png — histogram of word accuracy
    analysis/06_fold_boxplot.png       — top1 distribution per round (box plot)
    analysis/07_bucket_heatmap.png     — round × bucket accuracy heatmap

Run:
    python analyze_results.py
"""

import os
import csv
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from collections import defaultdict

# ── Paths ─────────────────────────────────────────────────────────────────────
RESULTS_DIR  = "."
ANALYSIS_DIR = os.path.join(RESULTS_DIR, "analysis")
os.makedirs(ANALYSIS_DIR, exist_ok=True)

GLOBAL_CSV   = os.path.join(RESULTS_DIR, "global_summary.csv")
ROUND_CSV    = os.path.join(RESULTS_DIR, "round_summary.csv")
DIFFICULT_CSV = os.path.join(RESULTS_DIR, "difficult_words.csv")

# ── Plot style ─────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family"     : "DejaVu Sans",
    "axes.spines.top" : False,
    "axes.spines.right": False,
    "axes.grid"       : True,
    "grid.alpha"      : 0.3,
    "figure.dpi"      : 120,
})

PALETTE = ["#2563EB", "#10B981", "#F59E0B", "#EF4444", "#8B5CF6"]

# ═══════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ═══════════════════════════════════════════════════════════════════════════════

print("Loading CSVs...")

global_df    = pd.read_csv(GLOBAL_CSV)
round_df     = pd.read_csv(ROUND_CSV)
difficult_df = pd.read_csv(DIFFICULT_CSV)

# Rename columns for convenience
global_df.columns = [c.strip().replace("(%)", "").replace(" ", "_").lower()
                     for c in global_df.columns]

print(f"  global_summary   : {len(global_df)} rows")
print(f"  round_summary    : {len(round_df)} rows")
print(f"  difficult_words  : {len(difficult_df)} rows")

# ═══════════════════════════════════════════════════════════════════════════════
# 1. PER-BUCKET ACCURACY CSV
# ═══════════════════════════════════════════════════════════════════════════════

print("\nGenerating per_bucket_accuracy.csv ...")

bucket_grp = global_df.groupby(["round", "bucket"]).agg(
    num_folds     = ("fold", "count"),
    mean_top1     = ("test_top1", "mean"),
    std_top1      = ("test_top1", "std"),
    mean_top5     = ("test_top5", "mean"),
    std_top5      = ("test_top5", "std"),
    mean_macro_f1 = ("macro_f1", "mean"),
    std_macro_f1  = ("macro_f1", "std"),
    mean_test_loss= ("test_loss", "mean"),
).reset_index()

bucket_grp.to_csv(
    os.path.join(ANALYSIS_DIR, "per_bucket_accuracy.csv"),
    index=False, float_format="%.4f"
)
print(f"  Saved: per_bucket_accuracy.csv  ({len(bucket_grp)} rows)")

# ═══════════════════════════════════════════════════════════════════════════════
# 2. PER-WORD ACCURACY CSV  (rename + sort)
# ═══════════════════════════════════════════════════════════════════════════════

print("\nGenerating per_word_accuracy.csv ...")

word_df = difficult_df.copy()
word_df.columns = ["word", "accuracy_pct", "num_evaluations"]
word_df = word_df.sort_values("accuracy_pct", ascending=True).reset_index(drop=True)
word_df["rank_hardest"] = word_df.index + 1

word_df.to_csv(
    os.path.join(ANALYSIS_DIR, "per_word_accuracy.csv"),
    index=False, float_format="%.2f"
)
print(f"  Saved: per_word_accuracy.csv  ({len(word_df)} words)")

# ═══════════════════════════════════════════════════════════════════════════════
# 3. GLOBAL ACCURACY CSV
# ═══════════════════════════════════════════════════════════════════════════════

print("\nGenerating global_accuracy.csv ...")

# From round_summary
rd = round_df.copy()
rd.columns = [c.strip().replace("(%)", "").replace(" ", "_").lower()
              for c in rd.columns]

global_acc = {
    "total_rounds"        : len(rd),
    "total_folds"         : len(global_df),

    "final_top1_mean"     : rd["mean_top1"].mean(),
    "final_top1_std"      : rd["mean_top1"].std(),
    "final_top1_min"      : rd["mean_top1"].min(),
    "final_top1_max"      : rd["mean_top1"].max(),

    "final_top5_mean"     : rd["mean_top5"].mean(),
    "final_top5_std"      : rd["mean_top5"].std(),
    "final_top5_min"      : rd["mean_top5"].min(),
    "final_top5_max"      : rd["mean_top5"].max(),

    "final_macro_f1_mean" : rd["mean_macro_f1"].mean(),
    "final_macro_f1_std"  : rd["mean_macro_f1"].std(),
    "final_macro_f1_min"  : rd["mean_macro_f1"].min(),
    "final_macro_f1_max"  : rd["mean_macro_f1"].max(),

    "word_acc_overall_mean": word_df["accuracy_pct"].mean(),
    "word_acc_overall_std" : word_df["accuracy_pct"].std(),
    "pct_words_above_90"  : (word_df["accuracy_pct"] >= 90).mean() * 100,
    "pct_words_below_70"  : (word_df["accuracy_pct"] < 70).mean() * 100,
}

with open(os.path.join(ANALYSIS_DIR, "global_accuracy.csv"), "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["metric", "value"])
    for k, v in global_acc.items():
        writer.writerow([k, f"{v:.4f}" if isinstance(v, float) else v])

print(f"  Saved: global_accuracy.csv")

# ═══════════════════════════════════════════════════════════════════════════════
# VISUALIZATIONS
# ═══════════════════════════════════════════════════════════════════════════════

# ── 01: Round Comparison Bar Chart ───────────────────────────────────────────
print("\nPlot 01: Round comparison bar chart...")

metrics   = ["mean_top1", "mean_top5", "mean_macro_f1"]
std_cols  = ["std_top1",  "std_top5",  "std_macro_f1"]
labels    = ["Top-1 (%)", "Top-5 (%)", "Macro F1 (%)"]
colors    = ["#2563EB", "#10B981", "#F59E0B"]

rounds = rd["round"].tolist()
x      = np.arange(len(rounds))
width  = 0.25

fig, ax = plt.subplots(figsize=(12, 6))
for i, (m, s, lbl, c) in enumerate(zip(metrics, std_cols, labels, colors)):
    vals = rd[m].values
    stds = rd[s].values
    bars = ax.bar(x + i * width, vals, width, label=lbl,
                  color=c, alpha=0.85, yerr=stds, capsize=4,
                  error_kw={"elinewidth": 1.5, "ecolor": "black"})
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                f"{v:.1f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

ax.set_xticks(x + width)
ax.set_xticklabels([f"Round {r}" for r in rounds])
ax.set_ylabel("Accuracy / F1 (%)")
ax.set_title("Per-Round Performance: Top-1, Top-5, Macro F1\n(error bars = std across buckets)",
             fontsize=13, fontweight="bold")
ax.legend(loc="lower right")
ax.set_ylim(90, 100)
ax.yaxis.set_minor_locator(mticker.MultipleLocator(0.5))
plt.tight_layout()
plt.savefig(os.path.join(ANALYSIS_DIR, "01_round_comparison.png"), dpi=130)
plt.close()
print("  Saved: 01_round_comparison.png")

# ── 02: Global Summary Bar Chart ─────────────────────────────────────────────
print("Plot 02: Global summary bar chart...")

g_means = [global_acc["final_top1_mean"],
           global_acc["final_top5_mean"],
           global_acc["final_macro_f1_mean"]]
g_stds  = [global_acc["final_top1_std"],
           global_acc["final_top5_std"],
           global_acc["final_macro_f1_std"]]
g_labels = ["Top-1 Accuracy", "Top-5 Accuracy", "Macro F1"]

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(g_labels, g_means, color=colors, alpha=0.88,
              yerr=g_stds, capsize=6,
              error_kw={"elinewidth": 2, "ecolor": "black"})

for bar, v, s in zip(bars, g_means, g_stds):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + s + 0.15,
            f"{v:.2f}%", ha="center", va="bottom",
            fontsize=11, fontweight="bold")

ax.set_ylabel("(%)")
ax.set_title("Global Performance Summary\n(Mean ± Std across 5 rounds × 5 buckets)",
             fontsize=13, fontweight="bold")
ax.set_ylim(88, 101)
plt.tight_layout()
plt.savefig(os.path.join(ANALYSIS_DIR, "02_global_summary.png"), dpi=130)
plt.close()
print("  Saved: 02_global_summary.png")

# ── 03: Per-User Heatmap ─────────────────────────────────────────────────────
print("Plot 03: Per-user heatmap...")

user_round = global_df.groupby(["test_user", "round"])["test_top1"].mean().reset_index()
pivot = user_round.pivot(index="test_user", columns="round", values="test_top1")
pivot.columns = [f"R{c}" for c in pivot.columns]
pivot["Mean"] = pivot.mean(axis=1)
pivot = pivot.sort_values("Mean", ascending=False)

fig, ax = plt.subplots(figsize=(10, max(6, len(pivot) * 0.45)))
sns.heatmap(
    pivot, annot=True, fmt=".1f", cmap="RdYlGn",
    vmin=80, vmax=100, linewidths=0.5,
    linecolor="white", ax=ax,
    annot_kws={"size": 8},
    cbar_kws={"label": "Top-1 Accuracy (%)"},
)
ax.set_title("Per-User Mean Top-1 Accuracy Across Rounds\n(sorted by mean accuracy)",
             fontsize=12, fontweight="bold")
ax.set_xlabel("Round")
ax.set_ylabel("User")
plt.tight_layout()
plt.savefig(os.path.join(ANALYSIS_DIR, "03_per_user_heatmap.png"), dpi=130)
plt.close()
print("  Saved: 03_per_user_heatmap.png")

# ── 04: Difficult Words (bottom 30) ─────────────────────────────────────────
print("Plot 04: Difficult words bar chart...")

bottom30 = word_df.head(30)

fig, ax = plt.subplots(figsize=(12, 7))
bar_colors = ["#EF4444" if v < 70 else "#F59E0B" if v < 85 else "#10B981"
              for v in bottom30["accuracy_pct"]]
bars = ax.barh(bottom30["word"], bottom30["accuracy_pct"],
               color=bar_colors, alpha=0.85)

ax.axvline(70, color="#EF4444", linestyle="--", linewidth=1.2, label="70% threshold")
ax.axvline(85, color="#F59E0B", linestyle="--", linewidth=1.2, label="85% threshold")

for bar, v in zip(bars, bottom30["accuracy_pct"]):
    ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
            f"{v:.1f}%", va="center", fontsize=8)

ax.set_xlabel("Accuracy (%)")
ax.set_title("30 Most Difficult Words\n(sorted by accuracy, lowest first)",
             fontsize=12, fontweight="bold")
ax.legend(loc="lower right")
ax.set_xlim(0, 105)
plt.tight_layout()
plt.savefig(os.path.join(ANALYSIS_DIR, "04_difficult_words.png"), dpi=130)
plt.close()
print("  Saved: 04_difficult_words.png")

# ── 05: Word Accuracy Distribution Histogram ─────────────────────────────────
print("Plot 05: Word accuracy distribution...")

fig, ax = plt.subplots(figsize=(10, 5))
n, bins, patches = ax.hist(word_df["accuracy_pct"], bins=20,
                           color="#2563EB", alpha=0.8, edgecolor="white")

# Color bins by performance zone
for patch, left in zip(patches, bins[:-1]):
    if left < 70:
        patch.set_facecolor("#EF4444")
    elif left < 85:
        patch.set_facecolor("#F59E0B")
    else:
        patch.set_facecolor("#10B981")

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

# ── 06: Box Plot — Top-1 per Round ───────────────────────────────────────────
print("Plot 06: Top-1 box plot per round...")

fold_data = [global_df[global_df["round"] == r]["test_top1"].values
             for r in sorted(global_df["round"].unique())]

fig, ax = plt.subplots(figsize=(9, 5))
bp = ax.boxplot(fold_data, patch_artist=True, notch=False,
                medianprops={"color": "black", "linewidth": 2})
for patch, color in zip(bp["boxes"], PALETTE):
    patch.set_facecolor(color)
    patch.set_alpha(0.75)

ax.set_xticklabels([f"Round {r}" for r in sorted(global_df["round"].unique())])
ax.set_ylabel("Top-1 Accuracy per Fold (%)")
ax.set_title("Top-1 Accuracy Distribution per Round\n(each point = one fold)",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(ANALYSIS_DIR, "06_fold_boxplot.png"), dpi=130)
plt.close()
print("  Saved: 06_fold_boxplot.png")

# ── 07: Round × Bucket Heatmap ───────────────────────────────────────────────
print("Plot 07: Round × bucket heatmap...")

pivot2 = bucket_grp.pivot(index="round", columns="bucket", values="mean_top1")
pivot2.index   = [f"Round {r}" for r in pivot2.index]
pivot2.columns = [f"Bucket {b}" for b in pivot2.columns]

fig, ax = plt.subplots(figsize=(9, 5))
sns.heatmap(
    pivot2, annot=True, fmt=".2f", cmap="YlOrRd",
    vmin=92, vmax=98,
    linewidths=0.5, linecolor="white", ax=ax,
    annot_kws={"size": 10, "fontweight": "bold"},
    cbar_kws={"label": "Mean Top-1 Accuracy (%)"},
)
ax.set_title("Mean Top-1 Accuracy: Round × Bucket\n(reveals vocabulary-difficulty patterns)",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(ANALYSIS_DIR, "07_bucket_heatmap.png"), dpi=130)
plt.close()
print("  Saved: 07_bucket_heatmap.png")

# ═══════════════════════════════════════════════════════════════════════════════
# PRINT FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 65)
print("  ANALYSIS COMPLETE")
print("=" * 65)
print(f"\n  Final Top-1  : {global_acc['final_top1_mean']:.2f}% ± {global_acc['final_top1_std']:.2f}%")
print(f"  Final Top-5  : {global_acc['final_top5_mean']:.2f}% ± {global_acc['final_top5_std']:.2f}%")
print(f"  Macro F1     : {global_acc['final_macro_f1_mean']:.2f}% ± {global_acc['final_macro_f1_std']:.2f}%")
print(f"\n  Words > 90%  : {global_acc['pct_words_above_90']:.1f}%")
print(f"  Words < 70%  : {global_acc['pct_words_below_70']:.1f}%")
print(f"\n  All outputs  : {ANALYSIS_DIR}/")
print("=" * 65)