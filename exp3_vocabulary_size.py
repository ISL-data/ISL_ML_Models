"""
exp3_vocabulary_size.py  —  Experiment 3: Impact of Vocabulary Size
====================================================================
Evaluates recognition performance as vocabulary size is reduced from
489 words to ~100 words, under the LOSO setting for both pipelines,
sequentially.

Paper reference : Section 7.3.3
Protocol        : Partition the 489-word vocabulary into 5 disjoint subsets
                  of ~100 words each. For each subset, run a full LOSO
                  evaluation across all 15 signers. Repeat the entire process
                  5 times with different random partitions. Reports mean ± std
                  across all partitions and rounds.

Expected results (from paper):
    PopSign  —  Top-1: 94.97% ± 0.12%  (vs 90.53% on full 489 words)
    MASA     —  Top-1: 92.07% ± 0.16%  (vs 86.63% on full 489 words)

    Smaller vocabulary = easier task = higher accuracy.
    Performance is stable across different random partitions,
    confirming the result is not partition-dependent.

Run:
    python exp3_vocabulary_size.py

    PopSign runs first and releases GPU memory before MASA starts.
    No GPU conflict between the two pipelines.

Outputs:
    results/exp3_popsign/   — per-round, per-subset results for PopSign
    results/exp3_masa/      — per-round, per-subset results for MASA
"""

import os
import sys
import subprocess

# ============================================================
# DATA ROOTS — match HuggingFace folder structure exactly
# Download from: https://huggingface.co/datasets/ISL500/ISL-DATA/tree/main
# ============================================================

POPSIGN_DATA_ROOT = "ISL-DATA/Landmarks/MediaPipe"   # .h5 files
MASA_DATA_ROOT    = "ISL-DATA/Landmarks/Pose"         # .npy files

POPSIGN_RESULTS   = "results/exp3_popsign"
MASA_RESULTS      = "results/exp3_masa"

# ============================================================

def run_pipeline(label, script_path, data_root, results_dir):
    print("\n" + "=" * 70)
    print(f"  Running {label}")
    print("=" * 70)

    env = os.environ.copy()
    env["EXP_DATA_ROOT"]   = data_root
    env["EXP_RESULTS_DIR"] = results_dir

    result = subprocess.run(
        [sys.executable, script_path],
        env=env
    )

    if result.returncode != 0:
        print(f"\n[ERROR] {label} failed with return code {result.returncode}")
        sys.exit(1)

    print(f"\n[DONE] {label} — results saved to: {results_dir}")


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)

    # PopSign runs first — GPU memory is fully released before MASA starts
    run_pipeline(
        label       = "PopSign Vocabulary Size (MediaPipe)",
        script_path = "Mediapipe_Transformer_Models/scripts/run_vocabulary_loso.py",
        data_root   = POPSIGN_DATA_ROOT,
        results_dir = POPSIGN_RESULTS,
    )

    # MASA runs second — no GPU conflict
    run_pipeline(
        label       = "MASA Vocabulary Size (MMPose)",
        script_path = "MMpose_AutoEncoder/scripts/run_masa_round.py",
        data_root   = MASA_DATA_ROOT,
        results_dir = MASA_RESULTS,
    )

    print("\n" + "=" * 70)
    print("  EXPERIMENT 3 COMPLETE")
    print(f"  PopSign results : {POPSIGN_RESULTS}")
    print(f"  MASA results    : {MASA_RESULTS}")
    print("=" * 70)
