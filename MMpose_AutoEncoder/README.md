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
Total feature dimension per frame:128 features per frame
Sequences are variable in length and are padded to a fixed maximum length during training.
---
## 3. Preprocessing Pipeline

The dataset processing involves:
- Reading pose data from `.h5` files  
- Extracting frame-wise hand keypoints  
- Trimming inactive frames  
- Normalizing keypoints relative to the wrist  
- Separating left-hand and right-hand normalization  
- Appending hand-presence indicators  
- Padding or truncating sequences to a fixed length  
This preprocessing ensures spatial consistency and temporal alignment across samples.
---
## 4. Model Architecture
The system uses a Transformer encoder for sequence classification.
Key components include:
- Linear projection layer to map input features to embedding space  
- Positional encoding to preserve temporal order  
- Multi-head self-attention layers  
- Feed-forward networks within each Transformer block  
- Dropout and stochastic depth for regularization  
- Masked pooling to ignore padded frames  
- Fully connected classification layer  
The model processes the entire sequence and produces a probability distribution over sign classes.
---
## 5. Training Strategy
The training procedure includes:
- Cross-entropy loss with optional label smoothing  
- Data augmentation techniques such as MixUp  
- Mini-batch training with padded sequences  
- Learning rate scheduling  
- Optimization using AdamW  
Masking is applied to ensure that padded frames do not influence the learning process.
---

## 6. Evaluation Protocols
### 6.1 Leave-One-Subject-Out (LOSO)
- Each user is held out once as the test set  
- Remaining users are used for training  
- Results are averaged across all users  
This protocol evaluates generalization to unseen signers.
---
### 6.2 Reduced Training Data Experiments
Training data is limited per class using a cap on the number of videos per user.
Typical configurations include:Cap = 10, 8, 6, 4, 2, 1
This setup evaluates the relationship between training data size and model performance.
---
### 6.3 Multi-Rotation Sampling (5-Cap Experiments)
In reduced-data settings, multiple sampling rotations are used to:
- Vary the subset of training samples  
- Evaluate robustness to sample selection  
- Reduce bias introduced by fixed sampling  
---
### 6.4 Frame Rate Reduction
Experiments are conducted using reduced frame rates to analyze:
- Temporal redundancy  
- Performance under sparse temporal sampling  
---
### 6.5 Train-Test Split (80-20)
A standard random split is used as a baseline for comparison with LOSO.
---
## 7. Metrics
The system is evaluated using:
- Top-1 Accuracy  
- Top-5 Accuracy  
- Macro F1 Score  
- Weighted F1 Score  
Macro F1 is particularly important for handling class imbalance.
---
## 8. Requirements

```bash
pip install torch torchvision torchaudio
pip install numpy h5py scikit-learn matplotlib tensorboard
python runner.py
---
```markdown


