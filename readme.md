# Security Patch Detection via Transformer Encoder on Behavioural Repository Data

This project extends [Farhi et al. (2023)](https://arxiv.org/abs/2302.02112) by replacing their Conv1D classifier with a lightweight pre-norm transformer encoder for detecting security patches from behavioural repository event sequences. It was developed as an undergraduate dissertation project.

## Overview

Security patches are frequently committed without public disclosure, leaving downstream users exposed. This system classifies commits as security-related or benign using temporal sequences of repository events (pushes, pull requests, issues, etc.) surrounding each commit, rather than analysing source code diffs.

**Key contributions over the original Farhi study:**
- Pre-norm transformer encoder (2 layers, 4 heads, d_model=64) achieving 84.7% mean accuracy across 5 seeds with 37x fewer parameters than the Conv1D baseline
- 16 engineered temporal features (commit urgency, churn diversity, code churn, etc.) and 7 GraphQL recency features, expanding the feature set from 412 to 435
- Causal (pre-commit only) windowing mode for realistic deployment scenarios
- External validation on 212 CVEfixes repositories with zero training overlap (88.4% accuracy)
- GroupKFold cross-validation (k=10) preventing data leakage by repository

## Project Structure

```
train.py                        # Main training pipeline (GroupKFold CV, CLI)
models.py                       # Model definitions (transformer, Conv1D, custom layers)
dataset_utils.py                # Data loading, feature engineering, windowing
helper.py                       # Utility functions

analyse.py                      # Statistical analysis suite (significance tests, ablation, permutation importance)
generate_confusion_matrices.py  # Confusion matrix figure generation
generate_attention_heatmap.py   # Attention weight heatmap visualisation
generate_attention_perhead.py   # Per-head attention specialisation plots

convert_cvefixes.py             # Step 1: Extract CVEfixes SQLite to per-repo CSVs
scrape_cvefixes_repos.py        # Step 2: Scrape GitHub event data for CVEfixes repos
enrich_cvefixes_events.py       # Step 3: Enrich CVEfixes CSVs with additional event types
evaluate_cvefixes.py            # Step 4: Evaluate trained models on CVEfixes test set

run_comparison.sh               # Run full 4-window x 2-model comparison (seed 42)
run_comparison_extra_seeds.sh   # Repeat across seeds 123, 7, 256, 999
run_cvefixes_windows.sh         # Run CVEfixes evaluation across window sizes

requirements.txt                # Python dependencies
```

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install tensorflow coloredlogs
```

Requires Python 3.10+ and TensorFlow 2.16+.

## Usage

### Training

```bash
# Transformer, symmetric window B=20, 10-fold GroupKFold
python train.py -a before --model transformer -k 10 --metadata -B 20

# Conv1D baseline, same configuration
python train.py -a before --model conv1d -k 10 --metadata -B 20

# Causal (pre-commit only) window
python train.py -a only_before --model transformer -k 10 --metadata -B 20

# Custom seed
python train.py -a before --model transformer -k 10 --metadata -B 20 -s 123
```

**Key arguments:**
| Flag | Description | Default |
|------|-------------|---------|
| `-a` | Window mode: `before` (symmetric) or `only_before` (causal) | required |
| `--model` | `transformer` or `conv1d` | required |
| `-k` | Number of cross-validation folds | 10 |
| `-B` | Window size (events before target commit) | 10 |
| `--metadata` | Include static repository metadata features | off |
| `-s` | Random seed | 42 |
| `-e` | Max epochs | 50 |
| `-c` | Save confusion matrix data | off |

### Full Experiment Suite

```bash
# All 8 configurations (4 window sizes x 2 models), seed 42
bash run_comparison.sh

# Repeat for seeds 123, 7, 256, 999
bash run_comparison_extra_seeds.sh
```

### CVEfixes External Validation

```bash
# 1. Extract from CVEfixes SQLite database
python convert_cvefixes.py

# 2. Scrape GitHub events for extracted repos
python scrape_cvefixes_repos.py

# 3. Enrich with additional event types
python enrich_cvefixes_events.py

# 4. Evaluate trained models
python evaluate_cvefixes.py
```

### Analysis and Figures

```bash
# Run full statistical analysis (significance tests, ablation, permutation importance)
python analyse.py

# Generate confusion matrix figures
python generate_confusion_matrices.py

# Generate attention heatmaps (requires trained transformer model)
python generate_attention_heatmap.py
python generate_attention_perhead.py
```

## Data

Training data is not included in this repository due to size (~50GB). The dataset is constructed from GitHub Archive event streams and requires the data collection pipeline from the [original repository](https://github.com/nitzanfarhi/SecurityPatchDetection). See `data_collection/create_dataset.py` in the original project. The full dataset is available here: (https://www.kaggle.com/datasets/nitzanfarhi/detecting-security-patches-via-behavioral-data?resource=download)

The CVEfixes external validation dataset is derived from [CVEfixes v1.0.7](https://zenodo.org/record/7029359) (Bhandari et al., 2021).

## Results Summary

Best single run (causal B=20, seed 42): **87.1% accuracy, 88.2% F1**

| Configuration | Transformer Acc | Conv1D Acc | Delta |
|---------------|----------------|------------|-------|
| Symmetric B=5 | 0.848 | 0.841 | +0.007 |
| Symmetric B=20 | 0.847 | 0.828 | +0.019 |
| Causal B=5 | 0.848 | 0.694 | +0.155 |
| Causal B=20 | 0.850 | 0.820 | +0.029 |

All results are 5-seed averages (seeds 42, 123, 7, 256, 999) with GroupKFold CV (k=10).

## Citation

This project builds on the work of Farhi et al.:

```bibtex
@article{farhi2023detecting,
  title={Detecting Security Patches via Behavioral Data in Code Repositories},
  author={Farhi, Nitzan and Koenigstein, Noam and Shavitt, Yuval},
  journal={arXiv preprint arXiv:2302.02112},
  year={2023}
}
```

Original repository: [https://github.com/nitzanfarhi/SecurityPatchDetection](https://github.com/nitzanfarhi/SecurityPatchDetection)
