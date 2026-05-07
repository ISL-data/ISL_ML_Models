"""
exp2_framerate_reduction.py  —  Experiment 2: Impact of Frame Rate
===================================================================
Runs the LOSO cross-validation at 5 fps for BOTH pipelines, sequentially,
and compares against the 30 fps baseline from Experiment 1a.

Paper reference : Section 7.3.4
Protocol        : Same 15-fold LOSO as Experiment 1a, but landmarks are
                  sub-sampled to 5 fps (stride=6 from 30 fps source).
                  Quantifies performance degradation under mobile deployment
                  constraints where high frame rate processing is not feasible.

Expected results (from paper):
    PopSign  —  Top-1: 86.97%  |  Top-5: 97.07%
                Macro F1: 85.80%  |  Weighted F1: 86.02%
                (drop of ~3.5% Top-1 vs 30fps)

    MASA     —  Top-1: 70.31%  |  Top-5: 90.58%
                Macro F1: 68.77%  |  Weighted F1: 68.94%
                (drop of ~16% Top-1 vs 30fps — MASA is much more sensitive
                 to frame rate than PopSign)

Run:
    python exp2_framerate_reduction.py

    PopSign runs first and releases GPU memory before MASA starts.
    No GPU conflict between the two pipelines.

Outputs:
    results/exp2_popsign/   — PopSign LOSO results at 5fps
    results/exp2_masa/      — MASA LOSO results at 5fps
"""

import os
import sys
import subprocess

# ============================================================
# DATA ROOTS — match HuggingFace folder structure exactly
# Download from: https://huggingface.co/datasets/ISL500/ISL-DATA/tree/main
# ============================================================

POPSIGN_DATA_ROOT = "ISL-DATA/Landmarks/MediaPipe"   # .h5 files (30fps source, subsampled to 5fps internally)
MASA_DATA_ROOT    = "ISL-DATA/Landmarks/Pose"         # .npy files (30fps source, subsampled to 5fps internally)

POPSIGN_RESULTS   = "results/exp2_popsign"
MASA_RESULTS      = "results/exp2_masa"

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
        label       = "PopSign LOSO at 5fps (MediaPipe)",
        script_path = "Mediapipe_Transformer_Models/scripts/run_loso_5fps.py",
        data_root   = POPSIGN_DATA_ROOT,
        results_dir = POPSIGN_RESULTS,
    )

    # MASA runs second — no GPU conflict
    run_pipeline(
        label       = "MASA LOSO at 5fps (MMPose)",
        script_path = "MMpose_AutoEncoder/scripts/run_masa_loso_5fps.py",
        data_root   = MASA_DATA_ROOT,
        results_dir = MASA_RESULTS,
    )

    print("\n" + "=" * 70)
    print("  EXPERIMENT 2 COMPLETE")
    print(f"  PopSign results : {POPSIGN_RESULTS}")
    print(f"  MASA results    : {MASA_RESULTS}")
    print("=" * 70)
