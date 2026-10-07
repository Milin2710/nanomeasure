"""Experiment 3: ablation study.

Ablation A - input representation: raw vs normalized vs normalized+scale-channel
Ablation B - backbone training mode: frozen vs partial vs full fine-tune

Each run reuses train_cnn.train(); results are collected into a single table
and bar chart so the comparison is readable at a glance.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from train_cnn import train  # noqa: E402

RESULTS_DIR = ROOT / "results"


def run_ablation(epochs=6):
    configs = [
        # name, normalize, use_scale_channel, backbone_mode
        ("ablA_raw", False, False, "finetune"),
        ("ablA_normalized", True, False, "finetune"),
        ("ablA_normalized_scale", True, True, "finetune"),
        ("ablB_frozen", True, True, "frozen"),
        ("ablB_partial", True, True, "partial"),
        ("ablB_finetune", True, True, "finetune"),  # same as ablA_normalized_scale, kept for clarity
    ]

    all_results = {}
    for name, normalize, use_scale_channel, backbone_mode in configs:
        print(f"\n=== Running {name} (normalize={normalize}, scale_channel={use_scale_channel}, "
              f"backbone={backbone_mode}) ===")
        metrics = train(name, normalize, use_scale_channel, backbone_mode, epochs=epochs,
                          lr=1e-3, batch_size=32)
        all_results[name] = metrics

    with open(RESULTS_DIR / "ablation_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # --- Ablation A plot: input representation ---
    group_a = ["ablA_raw", "ablA_normalized", "ablA_normalized_scale"]
    labels_a = ["Raw pixels", "Normalized", "Normalized\n+ scale channel"]
    mae_a = [all_results[n]["MAE_nm"] for n in group_a]

    # --- Ablation B plot: backbone training mode ---
    group_b = ["ablB_frozen", "ablB_partial", "ablB_finetune"]
    labels_b = ["Frozen\nbackbone", "Partial\n(layer4 only)", "Full\nfine-tune"]
    mae_b = [all_results[n]["MAE_nm"] for n in group_b]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].bar(labels_a, mae_a, color=["#999", "#5b8fd4", "#2a6bc2"])
    axes[0].set_ylabel("Test MAE (nm)")
    axes[0].set_title("Ablation A: input representation")
    for i, v in enumerate(mae_a):
        axes[0].text(i, v + 0.05, f"{v:.2f}", ha="center")

    axes[1].bar(labels_b, mae_b, color=["#999", "#5b8fd4", "#2a6bc2"])
    axes[1].set_ylabel("Test MAE (nm)")
    axes[1].set_title("Ablation B: backbone training mode")
    for i, v in enumerate(mae_b):
        axes[1].text(i, v + 0.05, f"{v:.2f}", ha="center")

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "plots" / "ablation_comparison.png", dpi=120)
    plt.close(fig)

    print("\nAblation study complete. Results saved to results/ablation_results.json")
    return all_results


if __name__ == "__main__":
    run_ablation()
