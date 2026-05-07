"""
exp4_training_data_reduction.py  —  Experiment 4: Impact of Training Data Availability
========================================================================================
Progressively reduces the number of training videos per word and measures
the effect on LOSO recognition performance for both pipelines, sequentially.

Paper reference : Section 7.3.2, Figure 6
Protocol        : Under LOSO, cap the maximum number of training recordings
                  per word per signer. Cap sweeps from n=10 down to n=1.
                  Words with more recordings are randomly subsampled to the cap.
                  Words with fewer recordings contribute all available recordings.
                  Output is a performance curve: Top-1 and Top-5 vs. number
                  of training videos per word.

Expected results (from paper):
    Cap n=10 (~134 videos/word): PopSign 89.85%  |  MASA 85.57%
    Cap n=7  (~98  videos/word): PopSign 88.95%  |  MASA 81.73%
    Cap n=5  (~70  videos/word): PopSign 88.16%  |  MASA 77.45%
    Cap n=2  (~28  videos/word): PopSign 82.84%  |  MASA  —
    Cap n=1  (~14  videos/word): PopSign 74.35%  |  MASA  —

    Performance elbow is at n=3 (~42 videos/word) for both pipelines.
    Below this point, performance drops sharply.

Run:
    python exp4_training_data_reduction.py

    PopSign runs first and releases GPU memory before MASA starts.
    No GPU conflict between the two pipelines.

Outputs:
    results/exp4_popsign/   — per-cap results and accuracy curve for PopSign
    results/exp4_masa/      — per-cap results and accuracy curve for MASA
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

POPSIGN_RESULTS   = "results/exp4_popsign"
MASA_RESULTS      = "results/exp4_masa"

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
        label       = "PopSign Training Data Reduction (MediaPipe)",
        script_path = "Mediapipe_Transformer_Models/scripts/run_reduce.py",
        data_root   = POPSIGN_DATA_ROOT,
        results_dir = POPSIGN_RESULTS,
    )

    # MASA runs second — no GPU conflict
    run_pipeline(
        label       = "MASA Training Data Reduction (MMPose)",
        script_path = "MMpose_AutoEncoder/scripts/run_masa_reduce.py",
        data_root   = MASA_DATA_ROOT,
        results_dir = MASA_RESULTS,
    )

    print("\n" + "=" * 70)
    print("  EXPERIMENT 4 COMPLETE")
    print(f"  PopSign results : {POPSIGN_RESULTS}")
    print(f"  MASA results    : {MASA_RESULTS}")
    print("=" * 70)
