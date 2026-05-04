# ISL Machine Learning ModelsThis repository contains Transformer-based models for Indian Sign Language (ISL) recognition using pose/keypoint features.The experiments focus on signer-independent recognition and data-efficiency analysis using Mediapipe-style hand keypoints.---## 📁 Repository Structure```textMediapipe_Transformer_Models/│├── Loso/├── loso_5fps/├── 80-20/├── reduce_training_videos/├── reduce_5cap/├── vocabulary_loso/└── README.md
 Problem Statement
The goal of this project is:
To recognize isolated sign language words using pose-based Transformer models and evaluate performance under different data conditions.

🧾 Input Representation
Each video is converted into pose/keypoint sequences:

42 hand joints

3 coordinates (x, y, z)

2 hand presence flags

Total features per frame = 128
Each sample is a sequence of frames fed into a Transformer model.

🤖 Model Overview
The model is a Transformer-based sequence classifier:

Linear embedding layer

Positional encoding

Multi-head self-attention encoder

Dropout + stochastic depth

Masked pooling (ignores padded frames)

Final classification head



📊 Experiments

1. LOSO (Leave-One-Subject-Out)
Folder: Loso/

Train on N-1 users

Test on 1 unseen user

Repeat for all users

👉 This evaluates real-world generalization

2. Reduced Training Videos (Data Efficiency)
Folder: reduce_training_videos/
Training data is reduced per class:
CapVideos per word10~1408~1126~844~562~281~14
👉 Purpose:


Study how much data is needed


Identify performance breakdown



3. 5-Cap Reduced Experiment
Folder: reduce_5cap/
Caps:
5, 4, 3, 2, 1
Includes multiple rotations to test:


sampling variation


robustness



4. 5 FPS Experiment
Folder: loso_5fps/


Reduces frame rate


Tests temporal robustness



5. 80-20 Split
Folder: 80-20/


Standard train-test split


Easier than LOSO


Used for comparison



6. Vocabulary LOSO
Folder: vocabulary_loso/


Evaluates performance on vocabulary subsets



📈 Key Observations
From experiments:


High performance with sufficient data


Gradual drop from cap 10 → cap 4


Sharp drop at cap 2 and cap 1


👉 Insight:

The model requires sufficient signer variation to generalize


⚠️ Important Notes


Training uses augmentation (e.g., MixUp)


Train accuracy may not be meaningful


Always report:


Test accuracy


Macro F1





📊 Metrics


Top-1 Accuracy


Top-5 Accuracy


Macro F1 Score


Weighted F1 Score


Confusion Matrix



🚀 How to Run
Clone repo:
git clone https://github.com/ISL-data/ISL_ML_Models.gitcd ISL_ML_Models/Mediapipe_Transformer_Models
Run LOSO:
cd Losopython runner.py
Run reduced data experiment:
cd reduce_training_videospython reduce_runner.py

⚙️ Requirements
pip install torch torchvision torchaudiopip install numpy h5py scikit-learn matplotlib tensorboard

📌 Dataset


Input: .h5 pose files


Extracted from video using keypoint models



🧾 Conclusion
This work demonstrates:


Transformer models can effectively learn sign recognition from pose data


Performance strongly depends on training data diversity


There exists a minimum data threshold for stable LOSO performance


