import os
import cv2
import h5py
import numpy as np from tqdm
import tqdmimport mediapipe as mp

from multiprocessing import Pool, cpu_count

mp_holistic = mp.solutions.holistic

==========================================

Landmark Helper Functions

==========================================

def landmarks_to_coords(hand_landmarks, input_features=21):"""Convert MediaPipe landmarks → numpy array (21,3)."""return np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark[]],dtype=np.float32)

==========================================

MediaPipe Extraction Core (BOTH HANDS)

==========================================

def extract_both_hands(video_path, input_features=21):"""Extract BOTH hands per frame.

Option A:
  - If a hand is missing → fill with zeros

Output:
  (T, 2, 21, 3)
    hand[0] = Left
    hand[1] = Right
"""

cap = cv2.VideoCapture(video_path)

holistic = mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

frames_out = []

while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = holistic.process(rgb)

    # --------------------------
    # LEFT HAND (or zeros)
    # --------------------------
    if res.left_hand_landmarks:
        L = landmarks_to_coords(res.left_hand_landmarks, input_features)
    else:
        L = np.zeros((input_features, 3), dtype=np.float32)

    # --------------------------
    # RIGHT HAND (or zeros)
    # --------------------------
    if res.right_hand_landmarks:
        R = landmarks_to_coords(res.right_hand_landmarks, input_features)
    else:
        R = np.zeros((input_features, 3), dtype=np.float32)

    # Combine into (2,21,3)
    both = np.stack([L, R], axis=0)

    frames_out.append(both)

cap.release()
holistic.close()

# If no frames extracted
if len(frames_out) == 0:
    return np.zeros((0, 2, input_features, 3), dtype=np.float32)

# Final tensor: (T,2,21,3)
return np.stack(frames_out, axis=0)

==========================================

File Discovery

==========================================

def find_all_mp4s(root_dir):"""Recursively collect ALL mp4 files."""mp4s = []for root, _, files in os.walk(root_dir):for f in files:if f.lower().endswith(".mp4"):mp4s.append(os.path.join(root, f))return sorted(mp4s)

==========================================

Worker Function (Multiprocessing)

==========================================

def process_one_video(args):"""Worker job:MP4 → Extract BOTH hands → Save H5"""video_path, data_root, output_root, overwrite = args

rel_path = os.path.relpath(video_path, data_root)

save_path = os.path.join(output_root, rel_path)
save_path = os.path.splitext(save_path)[0] + ".h5"

os.makedirs(os.path.dirname(save_path), exist_ok=True)

# Skip existing
if (not overwrite) and os.path.exists(save_path):
    return f"SKIP: {rel_path}"

try:
    # BOTH HAND extraction
    keypoints = extract_both_hands(video_path)

    if keypoints.shape[0] == 0:
        return f"EMPTY (no frames): {rel_path}"

    with h5py.File(save_path, "w") as f:
        f.create_dataset(
            "intermediate",
            data=keypoints,
            compression="gzip",
            compression_opts=4
        )

    return f"DONE: {rel_path}"

except Exception as e:
    return f"ERROR: {rel_path} -> {e}"

==========================================

Main Parallel Extraction

==========================================

def extract_dataset_parallel(data_root, output_root, overwrite=False, workers=40):"""Parallel MediaPipe extraction (Both Hands)."""

data_root = os.path.abspath(data_root)
output_root = os.path.abspath(output_root)

print("\n==============================")
print("DATA ROOT   :", data_root)
print("OUTPUT ROOT :", output_root)
print("==============================")

mp4_files = find_all_mp4s(data_root)
print("Total MP4 videos found:", len(mp4_files))

if len(mp4_files) == 0:
    print(" No videos found. Exiting.")
    return

max_workers = cpu_count()
print("CPU cores available:", max_workers)

# Safety margin
workers = min(workers, max_workers - 2)
print("Using workers:", workers)

args_list = [(v, data_root, output_root, overwrite) for v in mp4_files]

print("\nStarting multiprocessing extraction...\n")

with Pool(processes=workers) as pool:
    for msg in tqdm(pool.imap_unordered(process_one_video, args_list),
                    total=len(args_list)):

        # Print only important messages
        if msg.startswith("EMPTY") or msg.startswith("ERROR"):
            print(msg)

print("\n==============================")
print("DONE. All keypoints saved in:", output_root)
print("==============================\n")

==========================================

Entry Point

==========================================

if name == "main":

DATA_ROOT = "/home/nithin/Desktop/ISL_Goa_Data/user_10-15"
OUTPUT_ROOT = "/mnt/9a528fe4-4fe8-4dff-9a0c-8b1a3cf3d7ba/ISL_h5_bothhands_new_users"

extract_dataset_parallel(
    data_root=DATA_ROOT,
    output_root=OUTPUT_ROOT,
    overwrite=False,
    workers=30
)