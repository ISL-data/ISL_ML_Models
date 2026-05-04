# main.py
import os
import json
import traceback
from datetime import datetime

from train import train_model

# =========================
# Paths
# =========================
DATA_ROOT = "/data/ISL_GOA_SAMPLE_h5" # set to mediapipe split paths
TRAIN_DIR = os.path.join(DATA_ROOT, "train_split")
VAL_DIR   = os.path.join(DATA_ROOT, "val_split")
TEST_DIR  = os.path.join(DATA_ROOT, "test_split")

# =========================
# Run config
# =========================
seed = 43
max_frames = 200
batch_size = 64
input_features = 21
num_coords = 3
epochs = 100
patience = 6
cooldown = 0

lr = 5e-4
weight_decay = 1e-3
num_workers = 4

# =========================
# Logging 
# =========================
RUN_ROOT = os.getcwd() 
BASE_LOG_DIR = os.path.join(RUN_ROOT, "logs")
os.makedirs(BASE_LOG_DIR, exist_ok=True)

ts = datetime.now().strftime("%Y%m%d_%H%M%S")
run_name = f"{ts}_lr{lr}_wd{weight_decay}_ep{epochs}_bs{batch_size}_mf{max_frames}_seed{seed}"

log_dir = os.path.join(BASE_LOG_DIR, run_name)
os.makedirs(log_dir, exist_ok=True)

log_path = os.path.join(log_dir, f"{run_name}.log")

header = {
    "run_name": run_name,
    "log_dir": log_dir,
    "log_path": log_path,
    "seed": seed,
    "train_dir": TRAIN_DIR,
    "val_dir": VAL_DIR,
    "test_dir": TEST_DIR,
    "max_frames": max_frames,
    "batch_size": batch_size,
    "input_features": input_features,
    "num_coords": num_coords,
    "epochs": epochs,
    "patience": patience,
    "cooldown": cooldown,
    "lr": lr,
    "weight_decay": weight_decay,
    "num_workers": num_workers,
}

with open(log_path, "w", buffering=1) as f:
    f.write("=== RUN HEADER ===\n")
    f.write(json.dumps(header, indent=2))
    f.write("\n\n")

print("Writing to log file:", log_path)
print("Train:", TRAIN_DIR)
print("Val:  ", VAL_DIR)
print("Test: ", TEST_DIR)

try:
    results = train_model(
        seed=seed,
        train_split_dir=TRAIN_DIR,
        val_split_dir=VAL_DIR,
        test_split_dir=TEST_DIR,
        log_dir=log_dir,
        log_path=log_path,
        max_frames=max_frames,
        batch_size=batch_size,
        input_features=input_features,
        num_coords=num_coords,
        epochs=epochs,
        patience=patience,
        cooldown=cooldown,
        lr=lr,
        weight_decay=weight_decay,
        num_workers=num_workers,
        use_compile=True,
    )

    summary_path = os.path.join(log_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump({"header": header, "results": results}, f, indent=2)

    print("Log:", log_path)
    print("Summary JSON:", summary_path)
    print("Test acc:", results.get("test_acc"))

except Exception as e:
    print("Training failed:", e)
    traceback.print_exc()
