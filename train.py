# train.py
import os
import sys
import copy
import random
import time
import csv

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix

from dataset_class import (
    H5SignDatasetMobileLastN,
    find_h5_files,
    label_from_filename,
    compute_normalization_stats,
)

from model import TransformerClassifier, collate_fn_packed


def he_init_weights(m):
    """Apply He initialization to linear and conv layers, appropriate initialization to other layers."""
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv1d):
        nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.MultiheadAttention):
        if hasattr(m, "in_proj_weight") and m.in_proj_weight is not None:
            nn.init.xavier_uniform_(m.in_proj_weight)
        if hasattr(m, "out_proj") and m.out_proj.weight is not None:
            nn.init.xavier_uniform_(m.out_proj.weight)
        if hasattr(m, "in_proj_bias") and m.in_proj_bias is not None:
            nn.init.zeros_(m.in_proj_bias)
        if hasattr(m, "out_proj") and m.out_proj.bias is not None:
            nn.init.zeros_(m.out_proj.bias)
    elif isinstance(m, nn.Embedding):
        nn.init.normal_(m.weight, mean=0, std=0.1)


def topk_acc(logits: torch.Tensor, y: torch.Tensor, k: int = 5) -> float:
    """Top-k accuracy for a batch."""
    k = min(k, logits.size(1))
    topk = logits.topk(k, dim=1).indices
    return (topk == y.unsqueeze(1)).any(dim=1).float().mean().item()


@torch.no_grad()
def collect_preds_on_loader(model, loader, device):
    """Collect y_true and y_pred from a loader."""
    y_true, y_pred = [], []
    for inputs, labels, lengths, padding_mask in loader:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        lengths = lengths.to(device, non_blocking=True)
        padding_mask = padding_mask.to(device, non_blocking=True)
        outputs = model(inputs, lengths=lengths, padding_mask=padding_mask)
        preds = outputs.argmax(dim=1)
        y_true.append(labels.cpu().numpy())
        y_pred.append(preds.cpu().numpy())
    return np.concatenate(y_true), np.concatenate(y_pred)


def train_model(
    seed,
    train_split_dir,
    val_split_dir,
    test_split_dir,
    log_dir,
    log_path,  # single log file path from runner
    max_frames=384,
    batch_size=32,
    input_features=21,
    num_coords=3,
    epochs=100,
    patience=6,
    lr_scheduler_patience=3,
    cooldown=2,
    min_delta=1e-2,
    lr=1e-4,
    weight_decay=1e-3,
    num_workers=4,
    use_compile=True,
):
    os.makedirs(log_dir, exist_ok=True)

    # confusion matrix saved separately (CSV only)
    cm_csv_path = os.path.join(log_dir, "confusion.csv")

    try:
        with open(log_path, "a", buffering=1) as log_file:
            orig_stdout, orig_stderr = sys.stdout, sys.stderr
            sys.stdout, sys.stderr = log_file, log_file

            try:
                # seeds
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)

                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                torch.backends.cudnn.benchmark = True

                print("=== TRAIN START ===")
                print("Device:", device)

                # validate dirs
                for p in [train_split_dir, val_split_dir, test_split_dir]:
                    if not os.path.exists(p):
                        raise FileNotFoundError(f"Split dir not found: {p}")

                # label space from TRAIN only
                train_paths = find_h5_files(train_split_dir)
                if len(train_paths) == 0:
                    raise ValueError(f"No .h5 files found under: {train_split_dir}")

                train_labels_str = [label_from_filename(p) for p in train_paths]
                train_signs = sorted(set(train_labels_str))
                num_classes = len(train_signs)

                print(f"Train files: {len(train_paths)} | num_classes: {num_classes}")

                label_encoder = LabelEncoder()
                label_encoder.fit(train_signs)

                # normalization stats from TRAIN only
                norm_stats = compute_normalization_stats(train_paths, ref_idx=0, key="intermediate")
                print("Normalization stats computed from train.")

                # datasets
                train_set = H5SignDatasetMobileLastN(
                    train_split_dir,
                    label_encoder=label_encoder,
                    n=max_frames,
                    normalize_stats=norm_stats,
                    augment=True,
                )
                val_set = H5SignDatasetMobileLastN(
                    val_split_dir,
                    label_encoder=label_encoder,
                    n=max_frames,
                    normalize_stats=norm_stats,
                    augment=False,
                )
                test_set = H5SignDatasetMobileLastN(
                    test_split_dir,
                    label_encoder=label_encoder,
                    n=max_frames,
                    normalize_stats=norm_stats,
                    augment=False,
                )

                print(f"Dataset sizes | Train: {len(train_set)} | Val: {len(val_set)} | Test: {len(test_set)}")

                train_loader = DataLoader(
                    train_set,
                    batch_size=batch_size,
                    shuffle=True,
                    collate_fn=collate_fn_packed,
                    pin_memory=True,
                    num_workers=num_workers,
                )
                val_loader = DataLoader(
                    val_set,
                    batch_size=batch_size,
                    shuffle=False,
                    collate_fn=collate_fn_packed,
                    pin_memory=True,
                    num_workers=num_workers,
                )
                test_loader = DataLoader(
                    test_set,
                    batch_size=batch_size,
                    shuffle=False,
                    collate_fn=collate_fn_packed,
                    pin_memory=True,
                    num_workers=num_workers,
                )

                # model
                model = TransformerClassifier(
                    input_size=input_features * num_coords,
                    num_classes=num_classes,
                ).to(device)

                model.apply(he_init_weights)

                print("\n=== MODEL ARCHITECTURE ===")
                print(model)
                total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                print(f"Trainable params: {total_params}")

                if use_compile:
                    try:
                        model = torch.compile(model)
                        print("Model compiled with torch.compile")
                    except Exception as e:
                        print(f"torch.compile failed: {e}. Continuing uncompiled.")

                optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode="min",
                    factor=0.5,
                    patience=lr_scheduler_patience,
                    threshold=min_delta,
                    cooldown=cooldown,
                    verbose=True,
                )
                scaler = GradScaler()

                best_val_loss = float("inf")
                best_state = None
                best_epoch = -1
                best_train_acc = 0.0
                best_val_acc = 0.0
                epochs_no_improve = 0

                print("\n=== EPOCH LOG ===")
                for epoch in range(epochs):
                    t0 = time.time()

                    # ---- Train ----
                    model.train()
                    train_correct = 0
                    train_total = 0
                    train_loss_sum = 0.0

                    for inputs, labels, lengths, padding_mask in train_loader:
                        inputs = inputs.to(device, non_blocking=True)
                        labels = labels.to(device, non_blocking=True)
                        lengths = lengths.to(device, non_blocking=True)
                        padding_mask = padding_mask.to(device, non_blocking=True)

                        optimizer.zero_grad(set_to_none=True)

                        with autocast(device_type="cuda", enabled=(device.type == "cuda")):
                            outputs = model(inputs, lengths=lengths, padding_mask=padding_mask)
                            loss = F.cross_entropy(outputs, labels)

                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()

                        preds = outputs.argmax(dim=1)
                        train_correct += (preds == labels).sum().item()
                        train_total += labels.size(0)
                        train_loss_sum += loss.item() * labels.size(0)

                    train_acc = 100.0 * train_correct / max(train_total, 1)
                    train_loss = train_loss_sum / max(train_total, 1)

                    # ---- Val ----
                    model.eval()
                    val_correct = 0
                    val_total = 0
                    val_loss_sum = 0.0

                    with torch.no_grad():
                        for inputs, labels, lengths, padding_mask in val_loader:
                            inputs = inputs.to(device, non_blocking=True)
                            labels = labels.to(device, non_blocking=True)
                            lengths = lengths.to(device, non_blocking=True)
                            padding_mask = padding_mask.to(device, non_blocking=True)

                            with autocast(device_type="cuda", enabled=(device.type == "cuda")):
                                outputs = model(inputs, lengths=lengths, padding_mask=padding_mask)
                                vloss = F.cross_entropy(outputs, labels)

                            val_loss_sum += vloss.item() * labels.size(0)
                            preds = outputs.argmax(dim=1)
                            val_correct += (preds == labels).sum().item()
                            val_total += labels.size(0)

                    val_loss = val_loss_sum / max(val_total, 1)
                    val_acc = 100.0 * val_correct / max(val_total, 1)

                    scheduler.step(val_loss)
                    lr_now = optimizer.param_groups[0]["lr"]
                    dt = time.time() - t0

                    print(
                        f"Epoch {epoch:03d} | {dt:.1f}s | lr {lr_now:.2e} | "
                        f"train_loss {train_loss:.4f} train_acc {train_acc:.2f}% | "
                        f"val_loss {val_loss:.4f} val_acc {val_acc:.2f}%"
                    )

                    if val_loss < best_val_loss - min_delta:
                        best_val_loss = val_loss
                        best_val_acc = val_acc
                        best_train_acc = train_acc
                        best_state = copy.deepcopy(model.state_dict())
                        best_epoch = epoch
                        epochs_no_improve = 0
                    else:
                        epochs_no_improve += 1


                    if epochs_no_improve >= patience:
                        print(f"Early stopping after {patience} epochs without improvement.")
                        break

                if best_state is None:
                    raise RuntimeError("best_state was never set (no validation improvement).")

                # ---- Test (best checkpoint) ----
                model.load_state_dict(best_state)
                model.eval()

                test_correct = 0
                test_total = 0
                with torch.no_grad():
                    for inputs, labels, lengths, padding_mask in test_loader:
                        inputs = inputs.to(device, non_blocking=True)
                        labels = labels.to(device, non_blocking=True)
                        lengths = lengths.to(device, non_blocking=True)
                        padding_mask = padding_mask.to(device, non_blocking=True)

                        outputs = model(inputs, lengths=lengths, padding_mask=padding_mask)
                        preds = outputs.argmax(dim=1)
                        test_correct += (preds == labels).sum().item()
                        test_total += labels.size(0)

                test_acc = 100.0 * test_correct / max(test_total, 1)

                # ---- Confusion matrix ----
                y_true, y_pred = collect_preds_on_loader(model, test_loader, device)
                cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))

                class_names = list(label_encoder.classes_)
                with open(cm_csv_path, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([""] + class_names)
                    for i, row in enumerate(cm):
                        writer.writerow([class_names[i]] + row.tolist())

                print("\n=== SUMMARY ===")
                print(f"best_epoch: {best_epoch}")
                print(f"best_val_loss: {best_val_loss:.6f}")
                print(f"best_val_acc:  {best_val_acc:.4f}%")
                print(f"best_train_acc:{best_train_acc:.4f}%")
                print(f"test_acc:      {test_acc:.4f}%")
                print(f"confusion.csv: {cm_csv_path}")
                print("=== TRAIN END ===\n")

                result_dict = {
                    "seed": seed,
                    "num_classes": num_classes,
                    "best_epoch": best_epoch,
                    "best_val_loss": float(best_val_loss),
                    "best_val_acc": float(best_val_acc),
                    "best_train_acc": float(best_train_acc),
                    "test_acc": float(test_acc),
                    "confusion_csv_path": cm_csv_path,
                    "log_path": log_path,
                    "log_dir": log_dir,
                }

                sys.stdout, sys.stderr = orig_stdout, orig_stderr
                print("Run completed. Log:", log_path)
                return result_dict

            except Exception:
                sys.stdout, sys.stderr = orig_stdout, orig_stderr
                raise

    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e), "log_path": log_path, "log_dir": log_dir}
