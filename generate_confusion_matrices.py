#!/usr/bin/env python3
"""
Generate confusion matrix figures from saved results files.
Produces:
  1. Individual confusion matrix per experiment (figures/confusion_matrices/)
  2. Grouped comparison grids (by window size, by model, by seed)
  3. Summary grid of all configurations for seed=42
"""

import os
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


RESULTS_DIR = "results"
FIGURES_DIR = "figures/confusion_matrices"


def parse_result_file(filepath):
    """Parse a results .txt file and return (accuracy, fscore, tn, fp, fn, tp)."""
    with open(filepath) as f:
        text = f.read()
    acc_match = re.search(r"Accuracy:\s*([\d.]+)%", text)
    f1_match = re.search(r"fscore:\s*([\d.]+)%", text)
    cm_match = re.search(r"tn=(\d+),\s*fp=(\d+),\s*fn=(\d+),\s*tp=(\d+)", text)
    if not cm_match:
        return None
    return {
        "accuracy": float(acc_match.group(1)) / 100 if acc_match else None,
        "f1": float(f1_match.group(1)) / 100 if f1_match else None,
        "tn": int(cm_match.group(1)),
        "fp": int(cm_match.group(2)),
        "fn": int(cm_match.group(3)),
        "tp": int(cm_match.group(4)),
    }


def parse_filename(filename):
    """Extract experiment config from filename."""
    name = filename.replace(".txt", "")
    # Aggregation
    if "only_before" in name:
        aggr = "only_before"
    elif "before_cve" in name:
        aggr = "before"
    else:
        aggr = "unknown"
    # Window size
    b_match = re.search(r"_B(\d+)_", name)
    window = int(b_match.group(1)) if b_match else None
    # Model
    if name.endswith("_transformer"):
        model = "Transformer"
    elif name.endswith("_conv1d"):
        model = "Conv1D"
    else:
        model = "unknown"
    # Seed
    s_match = re.search(r"_s(\d+)_", name)
    seed = int(s_match.group(1)) if s_match else 42
    return {"aggr": aggr, "window": window, "model": model, "seed": seed}


def plot_single_cm(tn, fp, fn, tp, title, filepath, accuracy=None, f1=None):
    """Plot and save a single confusion matrix heatmap."""
    cm = np.array([[tn, fp], [fn, tp]])
    total = cm.sum()
    cm_pct = cm / total * 100

    fig, ax = plt.subplots(figsize=(5, 4.5))
    cmap = LinearSegmentedColormap.from_list("custom", ["#f7fbff", "#2171b5"])
    im = ax.imshow(cm, cmap=cmap, interpolation="nearest")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Benign", "Vulnerable"], fontsize=11)
    ax.set_yticklabels(["Benign", "Vulnerable"], fontsize=11)
    ax.set_xlabel("Predicted", fontsize=12, fontweight="bold")
    ax.set_ylabel("Actual", fontsize=12, fontweight="bold")

    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > cm.max() * 0.6 else "black"
            ax.text(j, i, f"{cm[i, j]}\n({cm_pct[i, j]:.1f}%)",
                    ha="center", va="center", color=color, fontsize=13, fontweight="bold")

    subtitle_parts = []
    if accuracy is not None:
        subtitle_parts.append(f"Acc: {accuracy:.1%}")
    if f1 is not None:
        subtitle_parts.append(f"F1: {f1:.1%}")
    subtitle = "  |  ".join(subtitle_parts)

    ax.set_title(f"{title}\n{subtitle}" if subtitle else title, fontsize=11, pad=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(filepath, dpi=200, bbox_inches="tight")
    plt.close()


def plot_grid_cm(entries, suptitle, filepath, ncols=4):
    """Plot a grid of confusion matrices."""
    n = len(entries)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4 * nrows))
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes[np.newaxis, :]
    elif ncols == 1:
        axes = axes[:, np.newaxis]

    cmap = LinearSegmentedColormap.from_list("custom", ["#f7fbff", "#2171b5"])

    for idx, entry in enumerate(entries):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]
        cm = np.array([[entry["tn"], entry["fp"]], [entry["fn"], entry["tp"]]])
        total = cm.sum()
        cm_pct = cm / total * 100

        im = ax.imshow(cm, cmap=cmap, interpolation="nearest")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Ben", "Vuln"], fontsize=9)
        ax.set_yticklabels(["Ben", "Vuln"], fontsize=9)

        for i in range(2):
            for j in range(2):
                color = "white" if cm[i, j] > cm.max() * 0.6 else "black"
                ax.text(j, i, f"{cm[i, j]}\n({cm_pct[i, j]:.1f}%)",
                        ha="center", va="center", color=color, fontsize=10, fontweight="bold")

        subtitle = f"Acc: {entry['accuracy']:.1%}" if entry.get("accuracy") else ""
        ax.set_title(f"{entry['label']}\n{subtitle}", fontsize=9, pad=6)

    # Hide unused axes
    for idx in range(n, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].set_visible(False)

    fig.suptitle(suptitle, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(filepath, dpi=200, bbox_inches="tight")
    plt.close()


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # Parse all result files
    experiments = []
    for fname in sorted(os.listdir(RESULTS_DIR)):
        if not fname.endswith(".txt"):
            continue
        fpath = os.path.join(RESULTS_DIR, fname)
        data = parse_result_file(fpath)
        if data is None:
            continue
        config = parse_filename(fname)
        data.update(config)
        data["filename"] = fname
        experiments.append(data)

    print(f"Parsed {len(experiments)} experiment results")

    # 1. Individual confusion matrices
    individual_dir = os.path.join(FIGURES_DIR, "individual")
    os.makedirs(individual_dir, exist_ok=True)
    for exp in experiments:
        aggr_label = "symmetric" if exp["aggr"] == "before" else "only_before"
        seed_label = f"_s{exp['seed']}" if exp["seed"] != 42 else ""
        title = f"{exp['model']} | {aggr_label} | B={exp['window']}{seed_label}"
        safe_name = f"{exp['model'].lower()}_{aggr_label}_B{exp['window']}_s{exp['seed']}"
        plot_single_cm(
            exp["tn"], exp["fp"], exp["fn"], exp["tp"],
            title,
            os.path.join(individual_dir, f"cm_{safe_name}.png"),
            accuracy=exp["accuracy"],
            f1=exp["f1"],
        )
    print(f"  Saved {len(experiments)} individual confusion matrices")

    # 2. Seed=42 comparison grid: Transformer vs Conv1D across windows
    seed42 = [e for e in experiments if e["seed"] == 42]

    # 2a. Transformer vs Conv1D, symmetric window, seed=42
    before_42 = sorted(
        [e for e in seed42 if e["aggr"] == "before"],
        key=lambda e: (e["window"], e["model"]),
    )
    for e in before_42:
        e["label"] = f"{e['model']} B={e['window']}"
    plot_grid_cm(before_42,
                 "Confusion Matrices: Symmetric Window (seed=42)",
                 os.path.join(FIGURES_DIR, "grid_symmetric_seed42.png"),
                 ncols=4)
    print("  Saved grid_symmetric_seed42.png")

    # 2b. Transformer vs Conv1D, only_before, seed=42
    only_42 = sorted(
        [e for e in seed42 if e["aggr"] == "only_before"],
        key=lambda e: (e["window"], e["model"]),
    )
    for e in only_42:
        e["label"] = f"{e['model']} B={e['window']}"
    plot_grid_cm(only_42,
                 "Confusion Matrices: Only-Before Window (seed=42)",
                 os.path.join(FIGURES_DIR, "grid_only_before_seed42.png"),
                 ncols=4)
    print("  Saved grid_only_before_seed42.png")

    # 3. Transformer across all 5 seeds at B=20 (best window)
    trans_b20 = sorted(
        [e for e in experiments if e["model"] == "Transformer" and e["window"] == 20],
        key=lambda e: (e["aggr"], e["seed"]),
    )
    for e in trans_b20:
        aggr_short = "sym" if e["aggr"] == "before" else "only"
        e["label"] = f"s={e['seed']} ({aggr_short})"
    plot_grid_cm(trans_b20,
                 "Transformer B=20: Confusion Matrices Across Seeds",
                 os.path.join(FIGURES_DIR, "grid_transformer_B20_seeds.png"),
                 ncols=5)
    print("  Saved grid_transformer_B20_seeds.png")

    # 4. Conv1D across all 5 seeds at B=20
    conv_b20 = sorted(
        [e for e in experiments if e["model"] == "Conv1D" and e["window"] == 20],
        key=lambda e: (e["aggr"], e["seed"]),
    )
    for e in conv_b20:
        aggr_short = "sym" if e["aggr"] == "before" else "only"
        e["label"] = f"s={e['seed']} ({aggr_short})"
    plot_grid_cm(conv_b20,
                 "Conv1D B=20: Confusion Matrices Across Seeds",
                 os.path.join(FIGURES_DIR, "grid_conv1d_B20_seeds.png"),
                 ncols=5)
    print("  Saved grid_conv1d_B20_seeds.png")

    # 5. Best configs comparison: Transformer symmetric B=20 across seeds
    best_trans = sorted(
        [e for e in experiments if e["model"] == "Transformer" and e["aggr"] == "before" and e["window"] == 20],
        key=lambda e: e["seed"],
    )
    for e in best_trans:
        e["label"] = f"Seed {e['seed']}"
    plot_grid_cm(best_trans,
                 "Best Config (Transformer, Symmetric, B=20): Robustness Across Seeds",
                 os.path.join(FIGURES_DIR, "grid_best_config_seeds.png"),
                 ncols=5)
    print("  Saved grid_best_config_seeds.png")

    # 6. Model comparison at each window size (seed=42, symmetric)
    for b in [5, 10, 20, 40]:
        pair = sorted(
            [e for e in seed42 if e["aggr"] == "before" and e["window"] == b],
            key=lambda e: e["model"],
        )
        for e in pair:
            e["label"] = e["model"]
        plot_grid_cm(pair,
                     f"Transformer vs Conv1D at B={b} (symmetric, seed=42)",
                     os.path.join(FIGURES_DIR, f"grid_model_comparison_B{b}.png"),
                     ncols=2)
    print("  Saved model comparison grids for B=5,10,20,40")

    # 7. Aggregated/averaged confusion matrix across 5 seeds for best config
    trans_sym_b20 = [e for e in experiments
                     if e["model"] == "Transformer" and e["aggr"] == "before" and e["window"] == 20]
    avg_tn = np.mean([e["tn"] for e in trans_sym_b20])
    avg_fp = np.mean([e["fp"] for e in trans_sym_b20])
    avg_fn = np.mean([e["fn"] for e in trans_sym_b20])
    avg_tp = np.mean([e["tp"] for e in trans_sym_b20])
    avg_acc = np.mean([e["accuracy"] for e in trans_sym_b20])
    avg_f1 = np.mean([e["f1"] for e in trans_sym_b20])
    plot_single_cm(
        int(round(avg_tn)), int(round(avg_fp)),
        int(round(avg_fn)), int(round(avg_tp)),
        "Transformer Symmetric B=20\n(averaged across 5 seeds)",
        os.path.join(FIGURES_DIR, "cm_averaged_best_config.png"),
        accuracy=avg_acc, f1=avg_f1,
    )
    print("  Saved cm_averaged_best_config.png")

    print(f"\nDone! All figures saved to {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
