"""Shared evaluation metrics and plotting, used by every experiment so numbers
are computed the same way everywhere (classical baseline, CNN, ablations)."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = r2_score(y_true, y_pred)
    return {"MAE_nm": round(float(mae), 4), "RMSE_nm": round(rmse, 4), "R2": round(float(r2), 4)}


def save_metrics(metrics, path):
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)


def plot_pred_vs_true(y_true, y_pred, title, out_path, uncertainty=None):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    ax = axes[0]
    if uncertainty is not None:
        ax.errorbar(y_true, y_pred, yerr=uncertainty, fmt="o", alpha=0.4, ms=3,
                     ecolor="lightgray", elinewidth=1, capsize=0)
    else:
        ax.scatter(y_true, y_pred, alpha=0.4, s=12)
    lims = [min(y_true.min(), y_pred.min()) - 0.5, max(y_true.max(), y_pred.max()) + 0.5]
    ax.plot(lims, lims, "r--", lw=1, label="ideal")
    ax.set_xlabel("Ground-truth diameter (nm)")
    ax.set_ylabel("Predicted diameter (nm)")
    ax.set_title(f"{title}: predicted vs. true")
    ax.legend()

    ax2 = axes[1]
    errors = y_pred - y_true
    ax2.hist(errors, bins=30, color="steelblue", alpha=0.8)
    ax2.axvline(0, color="r", linestyle="--", lw=1)
    ax2.set_xlabel("Prediction error (nm)")
    ax2.set_ylabel("Count")
    ax2.set_title("Error distribution")

    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)
