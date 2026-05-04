import os
import numpy as np
import torch
import h5py
import random
import math
from collections import Counter


# ----- FILE / LABEL UTILS ------

def find_h5_files(root_dir):
    paths = []
    for r, _, files in os.walk(root_dir):
        for f in files:
            if f.lower().endswith(".h5"):
                paths.append(os.path.join(r, f))
    return sorted(paths)


def label_from_filename(path: str) -> str:
    """
    split sign label from before __ (double underscore) file name 
    """
    base = os.path.basename(path)
    stem = os.path.splitext(base)[0]

    if "__" in stem:
        return stem.split("__", 1)[0]

    if "_" in stem:
        return stem.split("_", 1)[0]

    return stem


# ----- NORMALIZATION UTILS ------

def compute_normalization_stats(paths, ref_idx=0, key="intermediate"):
    """
    Computes global mean/std for (T, 21, 3) centered around reference landmark.
    """
    all_centered = []
    for path in paths:
        try:
            with h5py.File(path, "r") as f:
                data = f[key][:].squeeze(1)  # (T, 21, 3)
                if data.shape[0] == 0:
                    continue
                ref = data[:, ref_idx:ref_idx+1, :]
                ref[np.isnan(ref)] = 0.5
                data_centered = data - ref
                all_centered.append(data_centered)
        except Exception as e:
            print(f"Skipping {path}: {e}")
            continue

    if len(all_centered) == 0:
        mean = np.zeros((1, 1, 3), dtype=np.float32)
        std = np.ones((1, 1, 3), dtype=np.float32)
        return {"mean": mean, "std": std}

    all_concat = np.concatenate(all_centered, axis=0)  # (total_frames, 21, 3)
    mean = np.nanmean(all_concat, axis=(0, 1), keepdims=True).astype(np.float32)  # (1,1,3)
    std = (np.nanstd(all_concat, axis=(0, 1), keepdims=True) + 1e-6).astype(np.float32)
    return {"mean": mean, "std": std}


# ------ DATASET CLASS ----------

class H5SignDatasetMobileLastN(torch.utils.data.Dataset):
    """
    Directory-based dataset for .h5 mediapipe landmarks.

    Expected .h5 dataset key: 'intermediate' with shape (T, 1, 21, 3).

    Labels are inferred from filename prefix: WORD_*.h5 -> WORD.

    Returns:
      data_tensor: (n, 63)
      label_tensor: ()
      length_tensor: () original length before pad/trim
      (optional) path if debug=True
    """

    def __init__(
        self,
        split_dir,
        label_encoder=None,      # sklearn LabelEncoder (optional)
        label2id=None,           # dict mapping str->int (alternative to label_encoder)
        n=60,
        normalize_stats=None,
        augment=False,
        debug=False,
        key="intermediate"
    ):
        self.paths = find_h5_files(split_dir)
        if len(self.paths) == 0:
            raise ValueError(f"No .h5 files found under: {split_dir}")

        self.n = n
        self.normalize_stats = normalize_stats
        self.augment = augment
        self.debug = debug
        self.key = key

        # Build labels
        self.labels_str = [label_from_filename(p) for p in self.paths]

        if label2id is None and label_encoder is None:
            # build label2id from this split (fine for train; for val/test you should reuse train mapping)
            uniq = sorted(set(self.labels_str))
            self.label2id = {lab: i for i, lab in enumerate(uniq)}
        elif label2id is not None:
            self.label2id = label2id
        else:
            # label_encoder provided
            self.label2id = None

        if label_encoder is not None:
            self.encoded_labels = label_encoder.transform(self.labels_str)
        else:
            self.encoded_labels = np.array([self.label2id[s] for s in self.labels_str], dtype=np.int64)

        # For your existing augment methods
        self.HAND_TREES = [
            [0, 1, 2, 3, 4],     # Thumb
            [0, 5, 6, 7, 8],     # Index
            [0, 9, 10, 11, 12],  # Middle
            [0, 13, 14, 15, 16], # Ring
            [0, 17, 18, 19, 20], # Pinky
        ]
        self.HAND_INDICES = np.arange(21)

        # Optional quick sanity print
        counts = Counter(self.labels_str)
        self.num_classes = len(counts)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        label = int(self.encoded_labels[idx])

        try:
            with h5py.File(path, "r") as f:
                data = f[self.key][:]   # (T, 1, 21, 3)
                data = data.squeeze(1)  # (T, 21, 3)
        except Exception as e:
            print(f"Error reading {path}: {e}")
            data = np.zeros((0, 21, 3), dtype=np.float32)

        # handle empty sequences
        if data.shape[0] == 0:
            orig_len = 0
            data = np.zeros((self.n, 21, 3), dtype=np.float32)
        else:
            orig_len = min(data.shape[0], self.n)

            if self.augment:
                data = self.apply_augmentations(data)

            if self.normalize_stats:
                data = (data - self.normalize_stats["mean"]) / (self.normalize_stats["std"] + 1e-6)

            # Truncate or pad at end
            T = data.shape[0]
            if T > self.n:
                data = data[:self.n]
            elif T < self.n:
                pad = np.zeros((self.n - T, 21, 3), dtype=data.dtype)
                data = np.concatenate([data, pad], axis=0)

        data_tensor = torch.tensor(data, dtype=torch.float32).view(self.n, -1)  # (n, 63)
        label_tensor = torch.tensor(label, dtype=torch.long)
        length_tensor = torch.tensor(orig_len, dtype=torch.long)

        if self.debug:
            return data_tensor, label_tensor, length_tensor, path
        return data_tensor, label_tensor, length_tensor

    # --------------------------
    # Augmentations 
    # --------------------------

    def apply_augmentations(self, data):
        T, J, C = data.shape

        # 1. Temporal resample
        if random.random() < 0.7 and T > 1:
            factor = random.uniform(0.7, 1.5)
            new_len = max(1, int(T * factor))
            indices = np.linspace(0, T - 1, new_len).astype(int)
            data = data[indices]
            T = new_len

        # 2. Horizontal flip - set to 0 since all hands are flipped to be right handed
        if random.random() < 0:
            data[..., 0] = 1.0 - data[..., 0]

        # 3. Affine transforms (xy only)
        if random.random() < 0.8:
            xy = data[..., :2]
            z = data[..., 2:] if C > 2 else None
            center = np.array([0.5, 0.5])
            xy = xy - center
            scale = random.uniform(0.8, 1.2)
            xy = xy * scale
            shear_x = random.uniform(-0.15, 0.15)
            shear_y = random.uniform(-0.15, 0.15)
            shear_matrix = np.array([[1.0, shear_x], [shear_y, 1.0]])
            xy = xy @ shear_matrix.T
            angle = random.uniform(-30, 30) * math.pi / 180
            c, s = math.cos(angle), math.sin(angle)
            rotation = np.array([[c, -s], [s, c]])
            xy = xy @ rotation.T
            shift = np.random.uniform(-0.05, 0.05, size=(1, 1, 2))
            xy = xy + shift + center
            data = np.concatenate([xy, z], axis=-1) if z is not None else xy

        # 4. Finger tree rotate
        if random.random() < 0.5:
            data = self.finger_tree_rotate(
                data,
                hand_joint_indices=self.HAND_INDICES,
                hand_trees=self.HAND_TREES,
                max_angle_deg=15,
                joint_prob=0.4
            )

        # 5. Random rotation around wrist 
        if random.random() < 0.5:
            data = self.rotate_hand_3d(data, max_angle_deg=15)

        # 6. Random masking
        if random.random() < 0.6 and T > 1:
            start_mask = max(1, min(T, int(0.1 * T)))
            end_mask = min(10, T)
            if start_mask <= end_mask:
                mask_len = random.randint(start_mask, end_mask)
                start = random.randint(0, max(0, T - mask_len))
                data[start:start + mask_len] = 0.0

        # 7. Random joint cutout
        if random.random() < 0.8:
            for _ in range(random.randint(1, 3)):
                j = random.randint(0, J - 1)
                data[:, j] = 0.0

        return data

    @staticmethod
    def finger_tree_rotate(data, hand_joint_indices, hand_trees, max_angle_deg=15, joint_prob=0.4):
        for tree in hand_trees:
            if np.random.rand() < joint_prob:
                root_joint = tree[0]
                children = tree[1:]
                angle = np.deg2rad(np.random.uniform(-max_angle_deg, max_angle_deg))
                cos_a, sin_a = np.cos(angle), np.sin(angle)
                rot_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
                root_pos = data[:, root_joint, :2]
                for j in children:
                    rel_pos = data[:, j, :2] - root_pos
                    rotated = (rot_matrix @ rel_pos.T).T + root_pos
                    data[:, j, :2] = rotated
        return data

    @staticmethod
    def get_rotation_matrix_xyz(angle_x=0.0, angle_y=0.0, angle_z=0.0):
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(angle_x), -np.sin(angle_x)],
            [0, np.sin(angle_x),  np.cos(angle_x)]
        ])
        Ry = np.array([
            [ np.cos(angle_y), 0, np.sin(angle_y)],
            [0, 1, 0],
            [-np.sin(angle_y), 0, np.cos(angle_y)]
        ])
        Rz = np.array([
            [np.cos(angle_z), -np.sin(angle_z), 0],
            [np.sin(angle_z),  np.cos(angle_z), 0],
            [0, 0, 1]
        ])
        return Rz @ Ry @ Rx

    @staticmethod
    def rotate_hand_3d(data, max_angle_deg=15):
        angle_x = np.deg2rad(np.random.uniform(-max_angle_deg, max_angle_deg))
        angle_y = np.deg2rad(np.random.uniform(-max_angle_deg, max_angle_deg))
        angle_z = np.deg2rad(np.random.uniform(-max_angle_deg, max_angle_deg))
        R = H5SignDatasetMobileLastN.get_rotation_matrix_xyz(angle_x, angle_y, angle_z)  # BUGFIX

        origin = data[:, 0:1, :]
        centered = data - origin
        rotated = np.einsum("ij,tkj->tki", R, centered)
        return rotated + origin
