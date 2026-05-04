# MMPose AutoEncoder Experiments

This folder contains MASA-based experiments for isolated sign language recognition using MMPose pose/keypoint features and Transformer/autoencoder-style modeling.

## Folder Structure

| Folder | Description |
|---|---|
| `Masa_loso` | 15-fold Leave-One-Subject-Out (LOSO) experiment for MASA |
| `masa_loso_5fps` | Same LOSO experiment performed using 5 FPS input |
| `masa_80-20` | 80/20 grouped K-fold experiment for MASA |
| `masa_round_experiment` | 5 rounds of 15-fold LOSO; each round consists of 5 buckets |
| `masa_reduce_videos` | Reduces the number of training videos in a fixed pattern |
| `masa_5cap_reduce` | Reduced-video strategy with 3 rounds for each stage, starting from 70 videos per word down to 14 videos per word |

## Experiment Overview

The experiments evaluate the performance of MASA-style pose-based recognition under different evaluation protocols and training-data conditions.

The main goals are:

- To test signer-independent recognition using LOSO evaluation
- To analyze model performance at lower frame rates
- To compare grouped 80/20 splits with LOSO evaluation
- To study stability across repeated LOSO rounds
- To evaluate data efficiency by reducing the number of training videos per word

## 1. MASA LOSO

`Masa_loso` performs a 15-fold Leave-One-Subject-Out experiment.

In each fold:

- One user is held out for testing
- Remaining users are used for training
- Results are averaged across all 15 users

This setup measures how well the model generalizes to unseen signers.

## 2. MASA LOSO at 5 FPS

`masa_loso_5fps` repeats the LOSO experiment using 5 FPS input.

This experiment evaluates whether the model can maintain recognition performance when temporal information is reduced.

## 3. MASA 80/20 Grouped K-Fold

`masa_80-20` performs an 80/20 grouped K-fold experiment.

This setup is useful for comparison with LOSO, but LOSO is generally stricter because the test signer is completely unseen during training.

## 4. MASA Round Experiment

`masa_round_experiment` contains 5 rounds of 15-fold LOSO.

Each round consists of 5 buckets. This helps evaluate the stability of results across repeated fold/bucket arrangements.

## 5. MASA Reduced Videos

`masa_reduce_videos` reduces the number of training videos using a fixed pattern.

This experiment studies how model performance changes as the amount of available training data decreases.

Typical cap settings include:

```text
cap = 10, 8, 6, 4, 2, 1

The goal is to identify how much training data is needed for stable signer-independent recognition.

6. MASA 5-Cap Reduced Videos

masa_5cap_reduce follows a similar reduced-video strategy, but uses 3 rounds for each stage.

The experiment starts from approximately:

70 videos per word

and reduces down to:

14 videos per word

This setup evaluates robustness under limited-data conditions and checks the effect of different sample selections.

Metrics

Common evaluation metrics include:

Top-1 Accuracy
Top-5 Accuracy
Macro F1 Score
Weighted F1 Score
Confusion Matrix
Notes
LOSO is the primary protocol for signer-independent evaluation.
Reduced-video experiments are used to study data efficiency.
Lower caps generally reduce signer and motion diversity, which can reduce performance.
Results should be reported using test metrics, especially Top-1 Accuracy and Macro F1 Score.
