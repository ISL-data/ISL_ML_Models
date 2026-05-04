import os
import re
import argparse
import numpy as np
from tqdm import tqdm
from mmpose.apis import MMPoseInferencer

VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".webm")

def list_videos_recursive(root_dir: str):
    videos = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fn in filenames:
            if fn.lower().endswith(VIDEO_EXTS):
                full = os.path.join(dirpath, fn)
                rel = os.path.relpath(full, root_dir)  # preserve subfolders
                videos.append((full, rel))
    videos.sort(key=lambda x: x[1])
    return videos

def pick_main_person(instances):
    """Pick signer as largest bbox person (handles extra people)."""
    if not instances:
        return None
    best = instances[0]
    best_area = -1.0
    for inst in instances:
        bbox = inst.get("bbox", None)
        if bbox is None or len(bbox) < 4:
            continue
        x1, y1, x2, y2 = bbox[:4]
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        if area > best_area:
            best_area = area
            best = inst
    return best

def extract_video(inferencer, video_path, out_npy_path):
    """
    Save [T,133,3] = (x,y,score). If no detection on frame -> zeros.
    """
    seq = []
    for result in inferencer(video_path):
        preds = result.get("predictions", [])
        instances = preds[0] if preds else []

        main = pick_main_person(instances)
        if main is None:
            seq.append(np.zeros((133, 3), dtype=np.float32))
            continue

        kpts = np.asarray(main["keypoints"], dtype=np.float32)          # [133,2]
        scores = np.asarray(main["keypoint_scores"], dtype=np.float32)  # [133]

        if kpts.shape != (133, 2) or scores.shape != (133,):
            raise RuntimeError(f"Unexpected shapes kpts={kpts.shape}, scores={scores.shape}")

        frame = np.concatenate([kpts, scores[:, None]], axis=1)  # [133,3]
        seq.append(frame)

    # Safeguard against corrupted/empty videos
    if len(seq) == 0:
        raise ValueError("0 frames extracted. Video is likely corrupted or empty.")

    arr = np.stack(seq, axis=0)
    os.makedirs(os.path.dirname(out_npy_path), exist_ok=True)
    np.save(out_npy_path, arr)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", required=True, help="Path to ALL_CLIPS_R2")
    ap.add_argument("--pose_root", required=True, help="Path to your existing Pose folder")
    ap.add_argument("--start_user", type=int, required=True, help="Starting user number")
    ap.add_argument("--end_user", type=int, required=True, help="Ending user number")
    ap.add_argument("--subfolder", default="Round2", help="Subfolder name to prevent overwrites")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--skip_existing", action="store_true")
    args = ap.parse_args()

    inferencer = MMPoseInferencer(pose2d="wholebody", device=args.device)

    # Loop through the user range
    for user_id in range(args.start_user, args.end_user + 1):
        user_str = f"{user_id:03d}" 
        target_clip_dir = None
        
        # --- NEW LOGIC: Search inside ALL_CLIPS_R2 for the specific user ---
        # We look for a folder containing "user001", "User001", etc.
        search_pattern = re.compile(f"user{user_str}", re.IGNORECASE)
        
        if os.path.exists(args.base_dir):
            for item in os.listdir(args.base_dir):
                full_item_path = os.path.join(args.base_dir, item)
                if os.path.isdir(full_item_path) and search_pattern.search(item):
                    target_clip_dir = full_item_path
                    break # We found the user's folder, stop searching
        
        if target_clip_dir is None:
            print(f"[WARNING] Could not find any folder for User {user_str} inside {args.base_dir}. Skipping.")
            continue

        vids = list_videos_recursive(target_clip_dir)
        print(f"\n--- Found {len(vids)} videos in {target_clip_dir} ---")

        # Set output path to: Pose / User00X / Round2
        final_pose_root = os.path.join(args.pose_root, f"User{user_str}", args.subfolder)

        for full_path, rel_path in tqdm(vids, desc=f"Extracting User {user_str}"):
            rel_no_ext = os.path.splitext(rel_path)[0]
            out_path = os.path.join(final_pose_root, rel_no_ext + ".npy")

            if args.skip_existing and os.path.exists(out_path):
                continue

            try:
                extract_video(inferencer, full_path, out_path)
            except Exception as e:
                print(f"[ERROR] {rel_path}: {e}")

if __name__ == "__main__":
    main()
