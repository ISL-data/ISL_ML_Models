# Mediapipe Transformer Models for Sign Language Recognition

This repository contains Transformer-based models for isolated sign language recognition using pose-based keypoint representations.

The system focuses on signer-independent evaluation and data-efficiency analysis using structured hand-pose features extracted from video sequences.

---
## 1. System Overview

The recognition pipeline consists of the following stages:

1. Pose extraction from video
2. Keypoint preprocessing and normalization
3. Sequence representation
4. Transformer-based classification
5. Evaluation under different experimental settings

The objective is to map a sequence of hand-pose keypoints to a sign label.

---

## 2. Input Representation

Each video is represented as a temporal sequence of keypoint vectors.

For every frame:

- 42 hand joints are extracted
- Each joint has 3 spatial coordinates (x, y, z)
- 2 additional flags indicate left and right hand presence

Total feature dimension per frame:

```text
128 features per frame
