"""
analyze_loso.py  —  Post-hoc Analysis for Standard LOSO Results
================================================================
Reads results/ and produces:

CSVs (results/analysis/):
    per_fold_accuracy.csv     — per fold metrics + train-test gap
    per_word_accuracy.csv     — per-word accuracy sorted hardest first
    global_accuracy.csv       — final aggregated numbers

Plots:
    01_fold_comparison.png    — test top1/top5/F1 per fold with user labels
    02_global_summary.png     — final summary bar chart with std
    03_per_user_bar.png       — per-user test top1 sorted bar chart
    04_difficult_words.png    — bottom-30 hardest words
    05_word_accuracy_dist.png — histogram of word accuracy
    06_train_vs_test.png      — train vs test top1 per fold with gap
    07_loss_bar.png           — test loss per fold bar chart

Run (from results/ parent folder):
    python analyze_loso.py
"""

import os
import csv
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# ── Paths ──────────────────────────────────────────────────────────────────
RESULTS_DIR  = "results"
ANALYSIS_DIR = os.path.join(RESULTS_DIR, "analysis")
os.makedirs(ANALYSIS_DIR, exist_ok=True)

# ── Plot style ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family"      : "DejaVu Sans",
    "axes.spines.top"  : False,
    "axes.spines.right": False,
    "axes.grid"        : True,
    "grid.alpha"       : 0.3,
    "figure.dpi"       : 120,
})

# ═══════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ═══════════════════════════════════════════════════════════════════════════
print("Loading kfold_summary.csv ...")

raw_df   = pd.read_csv(os.path.join(RESULTS_DIR, "kfold_summary.csv"))

# Drop MEAN/STD rows
kfold_df = raw_df[~raw_df["fold"].astype(str).isin(["MEAN", "STD"])].copy()
kfold_df.columns = [c.replace("(%)", "").strip() for c in kfold_df.columns]
for col in ["train_top1","train_top5","test_top1","test_top5","macro_f1","weighted_f1","test_loss"]:
    kfold_df[col] = pd.to_numeric(kfold_df[col], errors="coerce")
# Rename columns cleanly
kfold_df.columns = [c.replace("(%)", "").strip() for c in kfold_df.columns]

NUM_FOLDS = len(kfold_df)
print(f"  {NUM_FOLDS} folds loaded.")

# Short user labels  e.g. ISL_DATA_USER001 → U01
kfold_df["user_short"] = kfold_df["test_user"].str.extract(r"USER(\d+)").apply(
    lambda x: f"U{int(x[0]):02d}", axis=1
)

# ── Load per-word accuracy ─────────────────────────────────────────────────
print("Loading per_class_accuracy.csv from each fold ...")
word_metrics = defaultdict(list)

for fold_idx in range(NUM_FOLDS):
    path = os.path.join(RESULTS_DIR, f"fold_{fold_idx}", "per_class_accuracy.csv")
    if not os.path.exists(path):
        print(f"  WARNING: {path} not found.")
        continue
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            word_metrics[row["class"]].append(float(row["accuracy(%)"]))

print(f"  {len(word_metrics)} unique words found.")

# ═══════════════════════════════════════════════════════════════════════════
# 1. PER-FOLD CSV
# ═══════════════════════════════════════════════════════════════════════════
print("\nGenerating per_fold_accuracy.csv ...")

per_fold_df = kfold_df[[
    "fold", "test_user", "train_top1", "train_top5",
    "test_top1", "test_top5", "macro_f1", "weighted_f1", "test_loss"
]].copy()
per_fold_df["train_test_gap"] = per_fold_df["train_top1"] - per_fold_df["test_top1"]

per_fold_df.to_csv(
    os.path.join(ANALYSIS_DIR, "per_fold_accuracy.csv"),
    index=False, float_format="%.4f"
)
print("  Saved: per_fold_accuracy.csv")

# ═══════════════════════════════════════════════════════════════════════════
# 2. PER-WORD CSV
# ═══════════════════════════════════════════════════════════════════════════
print("Generating per_word_accuracy.csv ...")

word_rows = sorted(
    [{"word": w, "accuracy_pct": round(np.mean(a), 2),
      "std_pct": round(np.std(a), 2), "num_folds": len(a)}
     for w, a in word_metrics.items()],
    key=lambda x: x["accuracy_pct"]
)
word_df = pd.DataFrame(word_rows)
word_df["rank_hardest"] = range(1, len(word_df) + 1)
word_df.to_csv(
    os.path.join(ANALYSIS_DIR, "per_word_accuracy.csv"),
    index=False, float_format="%.2f"
)
print(f"  Saved: per_word_accuracy.csv  ({len(word_df)} words)")

# ═══════════════════════════════════════════════════════════════════════════
# 3. GLOBAL CSV
# ═══════════════════════════════════════════════════════════════════════════
print("Generating global_accuracy.csv ...")

global_acc = {
    "num_folds"           : NUM_FOLDS,
    "num_words"           : len(word_df),
    "mean_test_top1"      : round(kfold_df["test_top1"].mean(), 4),
    "std_test_top1"       : round(kfold_df["test_top1"].std(), 4),
    "min_test_top1"       : round(kfold_df["test_top1"].min(), 4),
    "max_test_top1"       : round(kfold_df["test_top1"].max(), 4),
    "mean_test_top5"      : round(kfold_df["test_top5"].mean(), 4),
    "std_test_top5"       : round(kfold_df["test_top5"].std(), 4),
    "mean_macro_f1"       : round(kfold_df["macro_f1"].mean(), 4),
    "std_macro_f1"        : round(kfold_df["macro_f1"].std(), 4),
    "mean_test_loss"      : round(kfold_df["test_loss"].mean(), 4),
    "mean_train_test_gap" : round((kfold_df["train_top1"] - kfold_df["test_top1"]).mean(), 4),
    "pct_words_above_90"  : round((word_df["accuracy_pct"] >= 90).mean() * 100, 2),
    "pct_words_below_70"  : round((word_df["accuracy_pct"] < 70).mean() * 100, 2),
    "hardest_user"        : kfold_df.loc[kfold_df["test_top1"].idxmin(), "test_user"],
    "hardest_user_top1"   : round(kfold_df["test_top1"].min(), 2),
    "easiest_user"        : kfold_df.loc[kfold_df["test_top1"].idxmax(), "test_user"],
    "easiest_user_top1"   : round(kfold_df["test_top1"].max(), 2),
}

with open(os.path.join(ANALYSIS_DIR, "global_accuracy.csv"), "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["metric", "value"])
    for k, v in global_acc.items():
        writer.writerow([k, v])
print("  Saved: global_accuracy.csv")

# ═══════════════════════════════════════════════════════════════════════════
# PLOTS
# ═══════════════════════════════════════════════════════════════════════════

users  = kfold_df["user_short"].tolist()
folds  = kfold_df["fold"].tolist()
t1     = kfold_df["test_top1"].tolist()
t5     = kfold_df["test_top5"].tolist()
f1     = kfold_df["macro_f1"].tolist()
tr1    = kfold_df["train_top1"].tolist()
loss   = kfold_df["test_loss"].tolist()

# ── 01: Fold Comparison ────────────────────────────────────────────────────
print("\nPlot 01: Fold comparison...")

x     = np.arange(NUM_FOLDS)
width = 0.26

fig, ax = plt.subplots(figsize=(16, 6))
for i, (vals, color, label) in enumerate([
    (t1, "#2563EB", "Test Top-1 (%)"),
    (f1, "#10B981", "Macro F1 (%)"),
    (t5, "#F59E0B", "Test Top-5 (%)"),
]):
    bars = ax.bar(x + i * width, vals, width, label=label, color=color, alpha=0.85)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.2,
                f"{v:.1f}", ha="center", va="bottom", fontsize=7)

ax.axhline(np.mean(t1), color="#2563EB", linestyle="--",
           linewidth=1.5, alpha=0.6, label=f"Mean Top-1 = {np.mean(t1):.2f}%")

ax.set_xticks(x + width)
ax.set_xticklabels(users, fontsize=9)
ax.set_ylabel("(%)")
ax.set_title("Per-User LOSO Performance: Test Top-1, Macro F1, Test Top-5",
             fontsize=12, fontweight="bold")
ax.legend(loc="lower left", fontsize=9)
ax.set_ylim(75, 105)
plt.tight_layout()
plt.savefig(os.path.join(ANALYSIS_DIR, "01_fold_comparison.png"), dpi=130)
plt.close()
print("  Saved: 01_fold_comparison.png")

# ── 02: Global Summary ─────────────────────────────────────────────────────
print("Plot 02: Global summary...")

g_labels = ["Test Top-1", "Test Top-5", "Macro F1"]
g_means  = [global_acc["mean_test_top1"], global_acc["mean_test_top5"], global_acc["mean_macro_f1"]]
g_stds   = [global_acc["std_test_top1"],  global_acc["std_test_top5"],  global_acc["std_macro_f1"]]
g_colors = ["#2563EB", "#F59E0B", "#10B981"]

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(g_labels, g_means, color=g_colors, alpha=0.88,
              yerr=g_stds, capsize=8,
              error_kw={"elinewidth": 2, "ecolor": "black"})
for bar, v, s in zip(bars, g_means, g_stds):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + s + 0.3,
            f"{v:.2f}%", ha="center", va="bottom",
            fontsize=12, fontweight="bold")

ax.set_ylabel("(%)")
ax.set_title("Global LOSO Performance\n(Mean ± Std across 15 folds)",
             fontsize=13, fontweight="bold")
ax.set_ylim(75, 105)
plt.tight_layout()
plt.savefig(os.path.join(ANALYSIS_DIR, "02_global_summary.png"), dpi=130)
plt.close()
print("  Saved: 02_global_summary.png")

# ── 03: Per-User Bar (sorted) ──────────────────────────────────────────────
print("Plot 03: Per-user bar chart...")

sorted_idx = np.argsort(t1)
sorted_users = [users[i] for i in sorted_idx]
sorted_t1    = [t1[i]    for i in sorted_idx]
bar_colors   = ["#EF4444" if v < 85 else "#F59E0B" if v < 90 else "#10B981"
                for v in sorted_t1]

fig, ax = plt.subplots(figsize=(12, 5))
bars = ax.bar(sorted_users, sorted_t1, color=bar_colors, alpha=0.85)
ax.axhline(np.mean(t1), color="black", linestyle="--",
           linewidth=1.5, label=f"Mean = {np.mean(t1):.2f}%")
for bar, v in zip(bars, sorted_t1):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f"{v:.1f}%", ha="center", va="bottom", fontsize=8)

ax.set_ylabel("Test Top-1 Accuracy (%)")
ax.set_title("Per-User Test Top-1 Accuracy\n(sorted ascending; red<85%, orange<90%, green≥90%)",
             fontsize=12, fontweight="bold")
ax.legend()
ax.set_ylim(75, 100)
plt.tight_layout()
plt.savefig(os.path.join(ANALYSIS_DIR, "03_per_user_bar.png"), dpi=130)
plt.close()
print("  Saved: 03_per_user_bar.png")

# ── 04: Difficult Words ────────────────────────────────────────────────────
print("Plot 04: Difficult words...")

bottom30   = word_df.head(30)
bar_colors = ["#EF4444" if v < 70 else "#F59E0B" if v < 85 else "#10B981"
              for v in bottom30["accuracy_pct"]]

fig, ax = plt.subplots(figsize=(12, 7))
ax.barh(bottom30["word"], bottom30["accuracy_pct"],
        color=bar_colors, alpha=0.85,
        xerr=bottom30["std_pct"], capsize=3,
        error_kw={"elinewidth": 1.2, "ecolor": "gray"})
ax.axvline(70, color="#EF4444", linestyle="--", linewidth=1.2, label="70%")
ax.axvline(85, color="#F59E0B", linestyle="--", linewidth=1.2, label="85%")
for i, (_, row) in enumerate(bottom30.iterrows()):
    ax.text(row["accuracy_pct"] + 0.5, i,
            f"{row['accuracy_pct']:.1f}%", va="center", fontsize=8)
ax.set_xlabel("Accuracy (%)")
ax.set_title("30 Most Difficult Words\n(error bars = std across folds)",
             fontsize=12, fontweight="bold")
ax.legend()
ax.set_xlim(0, 110)
plt.tight_layout()
plt.savefig(os.path.join(ANALYSIS_DIR, "04_difficult_words.png"), dpi=130)
plt.close()
print("  Saved: 04_difficult_words.png")

# ── 05: Word Accuracy Distribution ────────────────────────────────────────
print("Plot 05: Word accuracy distribution...")

fig, ax = plt.subplots(figsize=(10, 5))
n, bins, patches = ax.hist(word_df["accuracy_pct"], bins=20, edgecolor="white", alpha=0.85)
for patch, left in zip(patches, bins[:-1]):
    patch.set_facecolor(
        "#EF4444" if left < 70 else "#F59E0B" if left < 85 else "#10B981"
    )
ax.axvline(word_df["accuracy_pct"].mean(), color="black", linestyle="--",
           linewidth=2, label=f"Mean = {word_df['accuracy_pct'].mean():.1f}%")
ax.set_xlabel("Word-Level Accuracy (%)")
ax.set_ylabel("Number of Words")
ax.set_title("Distribution of Per-Word Accuracy Across All Folds\n(red<70%, orange<85%, green≥85%)",
             fontsize=12, fontweight="bold")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(ANALYSIS_DIR, "05_word_accuracy_dist.png"), dpi=130)
plt.close()
print("  Saved: 05_word_accuracy_dist.png")

# ── 06: Train vs Test Top-1 ────────────────────────────────────────────────
print("Plot 06: Train vs test...")

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(users, tr1, marker="o", color="#93C5FD", linewidth=2.5,
        markersize=7, label="Train Top-1")
ax.plot(users, t1,  marker="s", color="#2563EB", linewidth=2.5,
        markersize=7, label="Test Top-1")
ax.fill_between(range(NUM_FOLDS), tr1, t1, alpha=0.1, color="#2563EB")

for i, (tr, te, u) in enumerate(zip(tr1, t1, users)):
    ax.annotate(f"{te:.1f}%", (i, te),
                textcoords="offset points", xytext=(0, -14),
                ha="center", fontsize=7.5, color="#1D4ED8")

ax.set_xticks(range(NUM_FOLDS))
ax.set_xticklabels(users, fontsize=9)
ax.set_ylabel("Top-1 Accuracy (%)")
ax.set_title(f"Train vs Test Top-1 per User\nMean gap = {global_acc['mean_train_test_gap']:.2f}% (Train − Test)",
             fontsize=12, fontweight="bold")
ax.legend()
ax.set_ylim(78, 100)
plt.tight_layout()
plt.savefig(os.path.join(ANALYSIS_DIR, "06_train_vs_test.png"), dpi=130)
plt.close()
print("  Saved: 06_train_vs_test.png")

# ── 07: Test Loss per Fold ─────────────────────────────────────────────────
print("Plot 07: Loss per fold...")

loss_colors = ["#EF4444" if v > 0.5 else "#F59E0B" if v > 0.3 else "#10B981"
               for v in loss]

fig, ax = plt.subplots(figsize=(12, 4))
bars = ax.bar(users, loss, color=loss_colors, alpha=0.85)
ax.axhline(np.mean(loss), color="black", linestyle="--",
           linewidth=1.5, label=f"Mean = {np.mean(loss):.4f}")
for bar, v in zip(bars, loss):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{v:.3f}", ha="center", va="bottom", fontsize=8)
ax.set_ylabel("Test Loss")
ax.set_title("Test Loss per User Fold\n(red>0.5, orange>0.3, green≤0.3)",
             fontsize=12, fontweight="bold")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(ANALYSIS_DIR, "07_loss_bar.png"), dpi=130)
plt.close()
print("  Saved: 07_loss_bar.png")

# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  ANALYSIS COMPLETE")
print("=" * 65)
print(f"\n  Test Top-1   : {global_acc['mean_test_top1']:.2f}% ± {global_acc['std_test_top1']:.2f}%")
print(f"  Test Top-5   : {global_acc['mean_test_top5']:.2f}% ± {global_acc['std_test_top5']:.2f}%")
print(f"  Macro F1     : {global_acc['mean_macro_f1']:.2f}% ± {global_acc['std_macro_f1']:.2f}%")
print(f"  Train-Test Gap : {global_acc['mean_train_test_gap']:.2f}%")
print(f"\n  Hardest user : {global_acc['hardest_user']}  ({global_acc['hardest_user_top1']:.2f}%)")
print(f"  Easiest user : {global_acc['easiest_user']}  ({global_acc['easiest_user_top1']:.2f}%)")
print(f"\n  Words > 90%  : {global_acc['pct_words_above_90']:.1f}%")
print(f"  Words < 70%  : {global_acc['pct_words_below_70']:.1f}%")
print(f"\n  All outputs  : {ANALYSIS_DIR}/")
print("=" * 65)
