"""
extract_mediapipe.py  —  MediaPipe Both-Hands Landmark Extraction
==================================================================
Extracts both-hand landmarks from MP4 videos using MediaPipe Holistic
and saves them as compressed H5 files.

Supports temporal downsampling to a target FPS (e.g. 30 → 15 FPS) for
robustness evaluation under reduced frame-rate conditions, simulating
real-world deployment on mobile/tablet devices.

Frame Downsampling Strategy:
    Uniform stride-based subsampling. Given a video recorded at native
    FPS F_n and a target FPS F_t, every k-th frame is retained where:

        k = round(F_n / F_t)

    For F_n=30, F_t=15  →  k=2  (frames 0, 2, 4, 6, ...)
    For F_n=30, F_t=10  →  k=3  (frames 0, 3, 6, 9, ...)
    For F_n=30, F_t=30  →  k=1  (all frames retained, no downsampling)

    This is equivalent to uniform temporal subsampling — a standard
    approach in video understanding literature (e.g. TSN, SlowFast).
    It preserves the temporal order and relative motion speed of the
    sign, while reducing the number of frames by factor k.

Folder Structure Expected:
    data_root/
        ISL_DATA_USER001/
            All_clips_001/
                [subfolders/] *.mp4
        ISL_DATA_USER002/
            All_clips_002/
                [subfolders/] *.mp4
        ...
        ISL_DATA_USER015/
            All_clips_015/
                [subfolders/] *.mp4

    Only ISL_DATA_USER* folders are scanned.
    Inside each, only the matching All_clips_NNN subfolder is entered.
    All other folders at any level are ignored.

Usage:
    # Extract at native FPS (30)
    python extract_mediapipe.py --data_root /path/to/videos --output_root /path/to/h5 --fps 30

    # Extract at 15 FPS (every other frame)
    python extract_mediapipe.py --data_root /path/to/videos --output_root /path/to/h5 --fps 15

    # Extract at 10 FPS
    python extract_mediapipe.py --data_root /path/to/videos --output_root /path/to/h5 --fps 10

Output:
    H5 file per video with dataset key "intermediate", shape (T, 2, 21, 3)
      - T   : number of frames after downsampling
      - 2   : left hand (index 0), right hand (index 1)
      - 21  : MediaPipe hand landmarks per hand
      - 3   : x, y, z coordinates (normalised to image dimensions)

    Missing hand → filled with zeros (shape 21, 3) for that frame.
"""

import os
import cv2
import h5py
import argparse
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

import mediapipe as mp

mp_holistic = mp.solutions.holistic


# ==========================================
# Landmark Helper
# ==========================================

def landmarks_to_coords(hand_landmarks, num_joints=21):
    """
    Convert MediaPipe hand landmark object → numpy array (21, 3).

    Coordinates are in normalised image space [0, 1] as returned
    by MediaPipe. z represents relative depth (not metric depth).
    """
    return np.array(
        [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark[:num_joints]],
        dtype=np.float32,
    )


# ==========================================
# Core Extraction (Both Hands, Single Video)
# ==========================================

def extract_both_hands(video_path, num_joints=21, target_fps=None):
    """
    Extract both-hand landmarks from a single video.

    Parameters
    ----------
    video_path  : str   — path to .mp4 file
    num_joints  : int   — number of hand joints to retain (default 21)
    target_fps  : float — desired output FPS. If None or >= native FPS,
                          all frames are retained. If lower than native,
                          uniform stride subsampling is applied.

    Returns
    -------
    numpy.ndarray of shape (T, 2, 21, 3), dtype float32
        T is the number of retained frames after downsampling.
        Index 0 = left hand, Index 1 = right hand.
        Absent hand → zeros.

    Frame Downsampling:
        stride k = round(native_fps / target_fps)
        Frames at indices 0, k, 2k, 3k, ... are processed.
        All other frames are read from the video buffer but discarded
        immediately (cap.read() is still called to advance the buffer,
        but no MediaPipe inference is run on skipped frames — this
        saves significant CPU time).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return np.zeros((0, 2, num_joints, 3), dtype=np.float32)

    # ── Determine frame stride ─────────────────────────────────
    native_fps = cap.get(cv2.CAP_PROP_FPS)
    if native_fps <= 0:
        native_fps = 30.0   # safe fallback if metadata is missing

    if target_fps is None or target_fps >= native_fps:
        stride = 1          # no downsampling
    else:
        stride = max(1, int(round(native_fps / target_fps)))

    # ── MediaPipe Holistic ─────────────────────────────────────
    holistic = mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    frames_out = []
    frame_idx  = 0

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break

        # Only run MediaPipe on retained frames (stride subsampling).
        # Skipped frames are advanced via cap.read() above but discarded
        # here without any inference — avoids wasted computation.
        if frame_idx % stride == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = holistic.process(rgb)

            # Left hand — zeros if absent
            if res.left_hand_landmarks:
                L = landmarks_to_coords(res.left_hand_landmarks, num_joints)
            else:
                L = np.zeros((num_joints, 3), dtype=np.float32)

            # Right hand — zeros if absent
            if res.right_hand_landmarks:
                R = landmarks_to_coords(res.right_hand_landmarks, num_joints)
            else:
                R = np.zeros((num_joints, 3), dtype=np.float32)

            frames_out.append(np.stack([L, R], axis=0))  # (2, 21, 3)

        frame_idx += 1

    cap.release()
    holistic.close()

    if len(frames_out) == 0:
        return np.zeros((0, 2, num_joints, 3), dtype=np.float32)

    return np.stack(frames_out, axis=0)   # (T, 2, 21, 3)


# ==========================================
# File Discovery  (ISL_DATA_USER aware)
# ==========================================

def find_all_mp4s(root_dir):
    """
    Collect all .mp4 files strictly from:
        root_dir/ISL_DATA_USERxxx/All_clips_xxx/**/*.mp4

    Only ISL_DATA_USER* folders are entered.
    Inside each, only the matching All_clips_NNN folder is entered.
    Everything else (other sibling folders, unrelated files) is ignored.

    Prints a summary of which user folders and clip folders were found
    so you can verify before extraction starts.
    """
    mp4s = []

    all_entries = sorted(os.listdir(root_dir))
    user_folders = [e for e in all_entries if e.startswith("ISL_DATA_USER")]

    if not user_folders:
        print("   No ISL_DATA_USER* folders found in data_root!")
        return mp4s

    print(f"\n  Found {len(user_folders)} user folder(s):")

    for user_folder in user_folders:
        user_path = os.path.join(root_dir, user_folder)
        if not os.path.isdir(user_path):
            continue

        # Extract zero-padded number e.g. "001" from "ISL_DATA_USER001"
        user_num = user_folder.replace("ISL_DATA_USER", "")
        clips_folder_name = f"All_clips_{user_num}"
        clips_path = os.path.join(user_path, clips_folder_name)

        if not os.path.isdir(clips_path):
            print(f"    {user_folder}/ →  '{clips_folder_name}' not found, skipping")
            continue

        # Recursively find all .mp4 inside All_clips_xxx only
        count_before = len(mp4s)
        for root, _, files in os.walk(clips_path):
            for f in files:
                if f.lower().endswith(".mp4"):
                    mp4s.append(os.path.join(root, f))
        count_added = len(mp4s) - count_before

        print(f"    {user_folder}/{clips_folder_name}/ → {count_added} .mp4 file(s)")

    print(f"\n  Total .mp4 files found : {len(mp4s)}")
    return sorted(mp4s)


# ==========================================
# Worker Function (Multiprocessing)
# ==========================================

def process_one_video(args):
    """
    Multiprocessing worker:
        MP4 → MediaPipe extraction → H5 save

    Parameters (packed as tuple for Pool.imap):
        video_path  : str
        data_root   : str
        output_root : str
        target_fps  : float or None
        overwrite   : bool
    """
    video_path, data_root, output_root, target_fps, overwrite = args

    rel_path  = os.path.relpath(video_path, data_root)
    save_path = os.path.join(output_root, rel_path)
    save_path = os.path.splitext(save_path)[0] + ".h5"

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if (not overwrite) and os.path.exists(save_path):
        return f"SKIP: {rel_path}"

    try:
        keypoints = extract_both_hands(video_path, target_fps=target_fps)

        if keypoints.shape[0] == 0:
            return f"EMPTY (no frames extracted): {rel_path}"

        with h5py.File(save_path, "w") as f:
            f.create_dataset(
                "intermediate",
                data=keypoints,
                compression="gzip",
                compression_opts=4,
            )
            # Store extraction metadata in H5 attributes for traceability
            f.attrs["native_fps"]  = 30.0
            f.attrs["target_fps"]  = target_fps if target_fps is not None else 30.0
            f.attrs["num_frames"]  = keypoints.shape[0]
            f.attrs["shape"]       = str(keypoints.shape)

        return f"DONE: {rel_path}  [{keypoints.shape[0]} frames]"

    except Exception as e:
        return f"ERROR: {rel_path} → {e}"


# ==========================================
# Main Parallel Extraction
# ==========================================

def extract_dataset_parallel(
    data_root,
    output_root,
    target_fps=None,
    overwrite=False,
    workers=40,
):
    """
    Parallel MediaPipe extraction across all MP4s in data_root.

    Parameters
    ----------
    data_root   : str   — root directory containing ISL_DATA_USER* folders
    output_root : str   — root directory to save .h5 files
                          (mirrors the input directory structure)
    target_fps  : float — target FPS for temporal downsampling.
                          None = retain all frames (native FPS).
    overwrite   : bool  — if False, skips already-extracted files
    workers     : int   — number of parallel processes
    """
    data_root   = os.path.abspath(data_root)
    output_root = os.path.abspath(output_root)

    print("\n" + "=" * 55)
    print(f"  MediaPipe Both-Hands Extraction")
    print(f"  Data root   : {data_root}")
    print(f"  Output root : {output_root}")
    print(f"  Target FPS  : {target_fps if target_fps is not None else 'native (no downsampling)'}")
    print(f"  Overwrite   : {overwrite}")
    print("=" * 55)

    mp4_files = find_all_mp4s(data_root)
    if not mp4_files:
        print("  No .mp4 files found. Check DATA_ROOT and folder structure.")
        return

    available_cores = cpu_count()
    # Reserve 2 cores for OS and other processes
    workers = min(workers, available_cores - 2)
    workers = max(1, workers)

    print(f"  CPU cores available : {available_cores}")
    print(f"  Workers to use      : {workers}")

    if target_fps is not None:
        stride = max(1, int(round(30.0 / target_fps)))
        print(f"\n  Downsampling: native 30 FPS → {target_fps} FPS")
        print(f"  Frame stride: every {stride} frame(s) retained")
        print(f"  Approx. output frames per 3s clip: {int(90 / stride)}")
    else:
        print(f"\n  No downsampling — all frames retained")

    print(f"\n  Starting extraction...\n")

    args_list = [
        (v, data_root, output_root, target_fps, overwrite)
        for v in mp4_files
    ]

    done_count  = 0
    skip_count  = 0
    empty_count = 0
    error_count = 0

    with Pool(processes=workers) as pool:
        for msg in tqdm(
            pool.imap_unordered(process_one_video, args_list),
            total=len(args_list),
            desc="Extracting",
        ):
            if msg.startswith("DONE"):
                done_count += 1
            elif msg.startswith("SKIP"):
                skip_count += 1
            elif msg.startswith("EMPTY"):
                empty_count += 1
                print(f"\n   {msg}")
            elif msg.startswith("ERROR"):
                error_count += 1
                print(f"\n   {msg}")

    print("\n" + "=" * 55)
    print(f"  EXTRACTION COMPLETE")
    print(f"  Done    : {done_count}")
    print(f"  Skipped : {skip_count}  (already exist)")
    print(f"  Empty   : {empty_count}  (no frames extracted)")
    print(f"  Errors  : {error_count}")
    print(f"  Output  : {output_root}")
    print("=" * 55 + "\n")


# ==========================================
# CLI Entry Point
# ==========================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MediaPipe Both-Hands Extraction — ISL_DATA_USER folder structure"
    )
    parser.add_argument(
        "--data_root", type=str, required=True,
        help="Root directory containing ISL_DATA_USER001 ... ISL_DATA_USER015 folders",
    )
    parser.add_argument(
        "--output_root", type=str, required=True,
        help="Root directory to save extracted .h5 files",
    )
    parser.add_argument(
        "--fps", type=float, default=None,
        help="Target FPS for temporal downsampling (e.g. 15). "
             "If not specified, all frames are retained (native FPS).",
    )
    parser.add_argument(
        "--workers", type=int, default=40,
        help="Number of parallel worker processes (default: 40)",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Re-extract even if .h5 file already exists",
    )
    args = parser.parse_args()

    extract_dataset_parallel(
        data_root   = args.data_root,
        output_root = args.output_root,
        target_fps  = args.fps,
        overwrite   = args.overwrite,
        workers     = args.workers,
    )
