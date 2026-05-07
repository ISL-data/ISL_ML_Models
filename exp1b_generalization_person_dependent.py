"""
exp1b_generalization_person_dependent.py  —  Experiment 1b: Signer-Dependent Recognition
==========================================================================================
Runs the 5-fold person-dependent (80-20) cross-validation for BOTH
the PopSign (MediaPipe) and MASA (MMPose) pipelines, sequentially.

Paper reference : Table 3, Section 7.3.1
Protocol        : 5-fold cross-validation with 80/20 train/test split grouped
                  by signer. Serves as an upper-bound reference to quantify
                  the generalization gap compared to Experiment 1a (LOSO).

Expected results (from paper):
    PopSign  —  Top-1: 95.09%  |  Top-5: 99.50%
                Macro F1: 94.96%  |  Weighted F1: 95.03%

    MASA     —  Top-1: 93.16%  |  Top-5: 99.16%
                Macro F1: 93.09%  |  Weighted F1: 93.16%

Run:
    python exp1b_generalization_person_dependent.py

    PopSign runs first and releases GPU memory before MASA starts.
    No GPU conflict between the two pipelines.

Outputs:
    results/exp1b_popsign/   — PopSign fold results, confusion matrix, curves
    results/exp1b_masa/      — MASA fold results, confusion matrix, curves
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

POPSIGN_RESULTS   = "results/exp1b_popsign"
MASA_RESULTS      = "results/exp1b_masa"

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
        label       = "PopSign Person-Dependent 80-20 (MediaPipe, 30fps)",
        script_path = "Mediapipe_Transformer_Models/scripts/run_80_20.py",
        data_root   = POPSIGN_DATA_ROOT,
        results_dir = POPSIGN_RESULTS,
    )

    # MASA runs second — no GPU conflict
    run_pipeline(
        label       = "MASA Person-Dependent 80-20 (MMPose, 30fps)",
        script_path = "MMpose_AutoEncoder/scripts/run_masa_80_20.py",
        data_root   = MASA_DATA_ROOT,
        results_dir = MASA_RESULTS,
    )

    print("\n" + "=" * 70)
    print("  EXPERIMENT 1b COMPLETE")
    print(f"  PopSign results : {POPSIGN_RESULTS}")
    print(f"  MASA results    : {MASA_RESULTS}")
    print("=" * 70)
