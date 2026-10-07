"""Experiment 2 (+ Experiment 3 ablations): train the CNN regressor.

Usage:
    python train_cnn.py --name main --normalize --scale-channel --backbone finetune --epochs 10
    python train_cnn.py --name ablation_raw --no-normalize --backbone finetune --epochs 10
    python train_cnn.py --name ablation_frozen --normalize --backbone frozen --epochs 10

Saves: results/models/<name>.pt, results/<name>_metrics.json, results/plots/<name>_pred_vs_true.png
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from dataset import NanoparticleDataset  # noqa: E402
from model_cnn import NanoMeasureNet  # noqa: E402
from evaluate import compute_metrics, save_metrics, plot_pred_vs_true  # noqa: E402

RESULTS_DIR = ROOT / "results"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_epoch(model, loader, criterion, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    total_loss, n = 0.0, 0
    all_true, all_pred = [], []
    with torch.set_grad_enabled(is_train):
        for batch in loader:
            x = batch["image"].to(DEVICE)
            y = batch["diameter_nm"].to(DEVICE)
            pred = model(x)
            loss = criterion(pred, y)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * x.size(0)
            n += x.size(0)
            all_true.extend(y.detach().cpu().numpy().tolist())
            all_pred.extend(pred.detach().cpu().numpy().tolist())
    return total_loss / n, all_true, all_pred


def train(name, normalize, use_scale_channel, backbone_mode, epochs, lr, batch_size):
    in_channels = 4 if use_scale_channel else 3
    train_ds = NanoparticleDataset("train", normalize=normalize, use_scale_channel=use_scale_channel)
    val_ds = NanoparticleDataset("val", normalize=normalize, use_scale_channel=use_scale_channel)
    test_ds = NanoparticleDataset("test", normalize=normalize, use_scale_channel=use_scale_channel)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = NanoMeasureNet(in_channels=in_channels, backbone_mode=backbone_mode).to(DEVICE)
    criterion = nn.HuberLoss(delta=2.0)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)

    history = {"train_loss": [], "val_loss": []}
    best_val = float("inf")
    best_state = None
    t0 = time.time()

    for epoch in range(epochs):
        train_loss, _, _ = run_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_true, val_pred = run_epoch(model, val_loader, criterion)
        scheduler.step(val_loss)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        print(f"  [{name}] epoch {epoch+1}/{epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    elapsed = time.time() - t0

    test_loss, test_true, test_pred = run_epoch(model, test_loader, criterion)
    metrics = compute_metrics(test_true, test_pred)
    metrics["train_time_sec"] = round(elapsed, 1)
    metrics["config"] = {"normalize": normalize, "use_scale_channel": use_scale_channel,
                           "backbone_mode": backbone_mode, "epochs": epochs,
                           "pretrained_backbone": model.pretrained}

    RESULTS_DIR.joinpath("models").mkdir(exist_ok=True, parents=True)
    torch.save({"state_dict": model.state_dict(), "in_channels": in_channels,
                 "backbone_mode": backbone_mode, "normalize": normalize,
                 "use_scale_channel": use_scale_channel},
                RESULTS_DIR / "models" / f"{name}.pt")
    save_metrics(metrics, RESULTS_DIR / f"{name}_metrics.json")
    plot_pred_vs_true(test_true, test_pred, f"CNN ({name})",
                        RESULTS_DIR / "plots" / f"{name}_pred_vs_true.png")

    with open(RESULTS_DIR / f"{name}_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"  [{name}] TEST metrics:", metrics)
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--normalize", action="store_true", default=True)
    ap.add_argument("--no-normalize", dest="normalize", action="store_false")
    ap.add_argument("--scale-channel", dest="use_scale_channel", action="store_true", default=False)
    ap.add_argument("--backbone", dest="backbone_mode", default="finetune",
                      choices=["frozen", "partial", "finetune"])
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    train(args.name, args.normalize, args.use_scale_channel, args.backbone_mode,
           args.epochs, args.lr, args.batch_size)


if __name__ == "__main__":
    main()
