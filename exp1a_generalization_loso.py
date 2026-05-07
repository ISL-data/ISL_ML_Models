"""
exp1a_generalization_loso.py  —  Experiment 1a: Signer-Independent Recognition (LOSO)
========================================================================================
Runs the full 15-fold Leave-One-Signer-Out (LOSO) cross-validation for BOTH
the PopSign (MediaPipe) and MASA (MMPose) pipelines, sequentially.

Paper reference : Table 3, Section 7.3.1
Protocol        : Train on 14 signers, test on 1. Repeat for all 15 signers.
                  Results averaged across all 15 folds.

Expected results (from paper):
    PopSign  —  Top-1: 90.53% ± 3.36%  |  Top-5: 98.10% ± 1.31%
                Macro F1: 89.66%        |  Weighted F1: 89.80%

    MASA     —  Top-1: 86.63% ± 3.84%  |  Top-5: 97.16% ± 1.74%
                Macro F1: 85.73%        |  Weighted F1: 85.88%

Run:
    python exp1a_generalization_loso.py

    PopSign runs first and releases GPU memory before MASA starts.
    No GPU conflict between the two pipelines.

Outputs:
    results/exp1a_popsign/   — PopSign fold results, confusion matrix, curves
    results/exp1a_masa/      — MASA fold results, confusion matrix, curves
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

POPSIGN_RESULTS   = "results/exp1a_popsign"
MASA_RESULTS      = "results/exp1a_masa"

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
        label       = "PopSign LOSO (MediaPipe, 30fps)",
        script_path = "Mediapipe_Transformer_Models/scripts/run_loso.py",
        data_root   = POPSIGN_DATA_ROOT,
        results_dir = POPSIGN_RESULTS,
    )

    # MASA runs second — no GPU conflict
    run_pipeline(
        label       = "MASA LOSO (MMPose, 30fps)",
        script_path = "MMpose_AutoEncoder/scripts/run_masa_loso.py",
        data_root   = MASA_DATA_ROOT,
        results_dir = MASA_RESULTS,
    )

    print("\n" + "=" * 70)
    print("  EXPERIMENT 1a COMPLETE")
    print(f"  PopSign results : {POPSIGN_RESULTS}")
    print(f"  MASA results    : {MASA_RESULTS}")
    print("=" * 70)
