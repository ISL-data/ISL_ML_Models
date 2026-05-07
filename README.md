# ISL_ML_Models

Benchmark code for the paper:

> **ISL-Connect: A Large-Scale Real-World Indian Sign Language Dataset Collected from Native Signers**  
> Submitted to IMWUT 2026

This repository contains the training and evaluation code for all experiments reported in the paper.  
The dataset (landmarks + videos) is hosted on HuggingFace: https://huggingface.co/datasets/ISL500/ISL-DATA/tree/main

---

## Dataset

ISL-Connect contains **86,978 videos** of **500 ISL words** collected from **15 Deaf native signers** using tablets in free-living environments. Experiments are run on **489 classes** after merging synonyms and homonyms.

Landmarks are provided in two formats:
- **MediaPipe** — `.h5` files, shape `(T, 2, 21, 3)` — used by the PopSign pipeline
- **MMPose** — `.npy` files, shape `(T, 133, 3)` — used by the MASA pipeline

Download the landmark files from HuggingFace and set `DATA_ROOT` in each script to point to your local copy.

---

## Repository Structure

```
ISL_ML_Models/
├── exp1a_generalization_loso.py              # Master script — Experiment 1a
├── exp1b_generalization_person_dependent.py  # Master script — Experiment 1b
├── exp2_framerate_reduction.py               # Master script — Experiment 2
├── exp3_vocabulary_size.py                   # Master script — Experiment 3
├── exp4_training_data_reduction.py           # Master script — Experiment 4
│
├── splits/                                   # Exact splits used in the paper
│   ├── fold_splits.json                      # File-level train/test splits for Exp 1b
│   ├── round_0_buckets.csv                   # Vocabulary partitions for Exp 3 — round 0
│   ├── round_1_buckets.csv                   # round 1
│   ├── round_2_buckets.csv                   # round 2
│   ├── round_3_buckets.csv                   # round 3
│   └── round_4_buckets.csv                   # round 4

├── Mediapipe_Transformer_Models/   # PopSign ASL v1.0 pipeline (MediaPipe landmarks)
│   ├── src/                        # Shared model, dataset, training code
│   │   ├── model.py
│   │   ├── dataset_class.py
│   │   ├── train.py
│   │   └── mediapipe_extract.py
│   └── scripts/                    # One script per experiment
│       ├── run_loso.py
│       ├── run_loso_5fps.py
│       ├── run_80_20.py
│       ├── run_reduce.py
│       ├── run_reduce_5cap.py
│       └── run_vocabulary_loso.py
│
├── MMpose_AutoEncoder/             # MASA pipeline (MMPose landmarks)
│   ├── src/                        # Shared model, dataset, training code
│   │   ├── masa_model.py
│   │   ├── masa_dataset.py
│   │   ├── masa_dataset_5fps.py
│   │   ├── masa_train.py
│   │   ├── masa_train_5fps.py
│   │   └── MMPose_extract.py
│   └── scripts/
│       ├── run_masa_loso.py
│       ├── run_masa_loso_5fps.py
│       ├── run_masa_80_20.py
│       ├── run_masa_reduce.py
│       ├── run_masa_5cap.py
│       └── run_masa_round.py
│
└── popsign/                        # PopSign baseline (single-hand, right hand only)
```

---

## How to Run

### 1. Download the landmarks

```bash
# Download from HuggingFace:
# https://huggingface.co/datasets/ISL500/ISL-DATA/tree/main
# You need: Landmarks/MediaPipe/ for PopSign, Landmarks/Pose/ for MASA
```

The expected local folder structure after download:

```
ISL-DATA/
└── Landmarks/
    ├── MediaPipe/    ← POPSIGN_DATA_ROOT (.h5 files)
    │   └── User001/
    └── Pose/         ← MASA_DATA_ROOT (.npy files)
        └── User001/
```

### 2. Set DATA_ROOT

At the top of each master script, update the two path variables to point to your local download:

```python
POPSIGN_DATA_ROOT = "ISL-DATA/Landmarks/MediaPipe"   # .h5 files
MASA_DATA_ROOT    = "ISL-DATA/Landmarks/Pose"         # .npy files
```

These are already set to match the HuggingFace folder structure exactly, so if you download to the same directory as the repo, no changes are needed.

### 3. Install dependencies

```bash
pip install torch mediapipe scikit-learn matplotlib tensorboard h5py numpy
```

### 4. Run any experiment

Each master script runs **PopSign first, then MASA sequentially**. PopSign completes and releases all GPU memory before MASA starts — there is no GPU conflict between the two pipelines.

```bash
python exp1a_generalization_loso.py
python exp1b_generalization_person_dependent.py
python exp2_framerate_reduction.py
python exp3_vocabulary_size.py
python exp4_training_data_reduction.py
```

Results are saved to a `results/` folder including:
- Per-fold metrics (CSV)
- Aggregated confusion matrix (PNG)
- Mean loss and accuracy curves (PNG)
- TensorBoard logs (`tensorboard --logdir=results/runs`)

---

## Experiments and Reported Results

### Experiment 1 — Generalization Across Signers
**Paper reference**: Table 3, Section 7.3.1

This experiment evaluates how well models generalize across different signers, under two settings:

#### 1a — Signer-Independent Recognition (LOSO, 30 fps)
**Protocol**: 15-fold Leave-One-Signer-Out cross-validation. Each fold trains on 14 signers and tests on the held-out signer. Results are averaged across all 15 folds.

| Model | Top-1 Acc | Top-5 Acc | Macro F1 | Weighted F1 |
|---|---|---|---|---|
| PopSign (MediaPipe) | **90.53% ± 3.36%** | 98.10% ± 1.31% | 89.66% ± 3.76% | 89.80% ± 3.66% |
| MASA (MMPose) | **86.63% ± 3.84%** | 97.16% ± 1.74% | 85.73% ± 4.23% | 85.88% ± 4.14% |

```bash
python exp1a_generalization_loso.py
```

#### 1b — Signer-Dependent Recognition (Person-Dependent, 80-20 split)
**Protocol**: Stratified 5-fold cross-validation with 80/20 train/test split grouped by signer. Reported as an upper-bound reference to quantify the signer-independent generalization gap.

**Split details**: This uses a **GroupKFold** approach where recording sessions are the groups. All 15 signers (USER001–USER015) appear in both train and test in every fold, but no individual session appears in more than one fold's test set — confirmed zero overlap across all fold pairs. The model sees each signer during training but is always tested on held-out sessions it has never trained on. The exact file-level splits used in the paper are provided in `splits/fold_splits.json`:

| Fold | Train files | Test files |
|---|---|---|
| fold_0 | 66,431 | 20,329 |
| fold_1 | 67,782 | 18,978 |
| fold_2 | 69,504 | 17,256 |
| fold_3 | 70,938 | 15,822 |
| fold_4 | 72,385 | 14,375 |

| Model | Top-1 Acc | Top-5 Acc | Macro F1 | Weighted F1 |
|---|---|---|---|---|
| PopSign (MediaPipe) | **95.09%** | 99.50% | 94.96% | 95.03% |
| MASA (MMPose) | **93.16%** | 99.16% | 93.09% | 93.16% |

```bash
python exp1b_generalization_person_dependent.py
```

---

### Experiment 2 — Impact of Frame Rate (LOSO, 5 fps)
**Paper reference**: Section 7.3.4  
**Protocol**: Same LOSO setup as Experiment 1a, but landmarks are sub-sampled to 5 fps. Quantifies performance degradation under mobile deployment constraints where high frame rate processing is not feasible.

| Model | Top-1 Acc | Top-5 Acc | Macro F1 | Weighted F1 |
|---|---|---|---|---|
| PopSign (MediaPipe) | **86.97%** | 97.07% | 85.80% | 86.02% |
| MASA (MMPose) | **70.31%** | 90.58% | 68.77% | 68.94% |

```bash
python exp2_framerate_reduction.py
```

---

### Experiment 3 — Impact of Vocabulary Size
**Paper reference**: Section 7.3.3  
**Protocol**: Under LOSO, partition the 489-word vocabulary into 5 disjoint subsets of ~100 words each. Repeated 5 times with different random partitions. Reports mean ± std across partitions and rounds.

**Split details**: The exact word assignments used in the paper are provided in `splits/round_0_buckets.csv` through `splits/round_4_buckets.csv`. Each file contains 5 buckets (bucket_0–bucket_4) of 97–98 words. Bucket_4 always has 97 words (489 is not evenly divisible by 5); all others have 98. To reproduce the exact reported numbers, load these CSVs in the experiment scripts rather than re-randomizing the partitions.

| Model | Top-1 Acc (100 words) | vs. Full 489 words |
|---|---|---|
| PopSign (MediaPipe) | **94.97% ± 0.12%** | +4.44% over full vocab |
| MASA (MMPose) | **92.07% ± 0.16%** | +5.44% over full vocab |

```bash
python exp3_vocabulary_size.py
```

---

### Experiment 4 — Impact of Training Data Availability
**Paper reference**: Section 7.3.2, Figure 6  
**Protocol**: Under LOSO, progressively cap the number of training videos per word per signer (n = 10 down to n = 1). Output is a performance curve (Top-1 and Top-5 vs. number of training videos).

| Training videos/word (approx) | PopSign Top-1 | MASA Top-1 |
|---|---|---|
| ~134 (n=10) | 89.85% | 85.57% |
| ~98 (n=7) | 88.95% | 81.73% |
| ~70 (n=5) | 88.16% | 77.45% |
| ~28 (n=2) | 82.84% | — |
| ~14 (n=1) | 74.35% | — |

Performance elbow is at n=3 (~42 videos/word) for both pipelines. Below this point, performance drops sharply.

```bash
python exp4_training_data_reduction.py
```

---

## Note on run_reduce_5cap.py / run_masa_5cap.py

These scripts (inside `scripts/`) are internal robustness checks that repeat the data reduction across 3 independent sampling rounds to confirm Experiment 4 results are not due to a lucky random subsample. They are not reported as standalone results in the paper and do not have a corresponding master script at the root level.

---

## Re-extracting Landmarks (Optional)

If you have the raw videos and want to re-extract landmarks from scratch:

```bash
# MediaPipe
python Mediapipe_Transformer_Models/src/mediapipe_extract.py

# MMPose
python MMpose_AutoEncoder/src/MMPose_extract.py
```

---

## Citation

```
@article{islconnect2026,
  title={ISL-Connect: A Large-Scale Real-World Indian Sign Language Dataset Collected from Native Signers},
  author={Anonymous},
  journal={Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies},
  year={2026}
}
```
