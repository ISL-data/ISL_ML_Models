import os
import cv2
import h5py
import argparse
import mediapipe as mp
import numpy as np
from tqdm import tqdm

mp_holistic = mp.solutions.holistic


def _landmarks_to_coords(hand_landmarks, input_features=21):
    return np.array(
        [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark[:input_features]],
        dtype=np.float32
    )


def _mirror_left_to_right(coords_xyz):
    out = coords_xyz.copy()
    out[:, 0] = 1.0 - out[:, 0]
    return out


def extract_mediapipe_keypoints_holistic_right_only(video_path, input_features=21):
    cap = cv2.VideoCapture(video_path)

    holistic = mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    per_frame = []

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = holistic.process(rgb)

        L = _landmarks_to_coords(res.left_hand_landmarks, input_features) if res.left_hand_landmarks else None
        R = _landmarks_to_coords(res.right_hand_landmarks, input_features) if res.right_hand_landmarks else None

        if (L is None) and (R is None):
            continue

        per_frame.append({"L": L, "R": R})

    cap.release()
    holistic.close()

    if len(per_frame) == 0:
        return np.zeros((0, 1, input_features, 3), dtype=np.float32)

    left_count  = sum(1 for d in per_frame if d["L"] is not None)
    right_count = sum(1 for d in per_frame if d["R"] is not None)
    use_left_as_right = left_count > right_count

    frames_out = []
    for d in per_frame:
        if d["R"] is not None:
            coords = d["R"]
        elif use_left_as_right and d["L"] is not None:
            coords = _mirror_left_to_right(d["L"])
        else:
            continue
        frames_out.append(coords)

    arr = np.stack(frames_out, axis=0)     # (T, 21, 3)
    arr = np.expand_dims(arr, axis=1)      # (T, 1, 21, 3)
    return arr


def list_mp4s(split_dir):
    mp4s = []
    for root, _, files in os.walk(split_dir):
        for f in files:
            if f.lower().endswith(".mp4"):
                mp4s.append(os.path.join(root, f))
    return mp4s


def process_split(split_dir, output_root, overwrite=False):
    split_dir = os.path.abspath(split_dir)
    split_name = os.path.basename(os.path.normpath(split_dir))
    out_split_dir = os.path.join(output_root, split_name)

    mp4_files = list_mp4s(split_dir)
    print(f"\n[{split_name}] Found {len(mp4_files)} .mp4 files")

    for video_path in tqdm(mp4_files, desc=f"Processing {split_name}"):
        rel_path = os.path.relpath(video_path, split_dir)
        save_path = os.path.join(out_split_dir, rel_path)
        save_path = os.path.splitext(save_path)[0] + ".h5"

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        if (not overwrite) and os.path.exists(save_path):
            continue

        try:
            keypoints = extract_mediapipe_keypoints_holistic_right_only(video_path)
            with h5py.File(save_path, "w") as f:
                f.create_dataset("intermediate", data=keypoints, compression="gzip", compression_opts=4)

        except Exception as e:
            print(f"\nERROR: {video_path}\n  -> {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--splits",
        nargs="+",
        required=True,
        help="Split directories, e.g. /data/ISL_GOA_SAMPLE/train_split /data/ISL_GOA_SAMPLE/val_split"
    )
    parser.add_argument(
        "--output_root",
        required=True,
        help="Output root, e.g. /data/ISL_GOA_SAMPLE_h5"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing .h5 files"
    )
    args = parser.parse_args()

    os.makedirs(args.output_root, exist_ok=True)

    for split_dir in args.splits:
        process_split(split_dir, args.output_root, overwrite=args.overwrite)

