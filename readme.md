# Behavioural Security Patch Detection with a Transformer Encoder

This project extends [Farhi et al. (2023)](https://arxiv.org/abs/2302.02112) by replacing the Conv1D classifier with a lightweight pre-norm transformer encoder for detecting security patches from behavioural repository event sequences. Developed as an undergraduate dissertation project.

## Overview

Security patches are frequently committed without public disclosure, leaving downstream users exposed. This system classifies commits as security-related or benign using temporal sequences of repository events (pushes, pull requests, issues, releases, stargazers, etc.) surrounding each commit, never reading source code or commit messages.

**Key contributions over the original Farhi study:**
- Pre-norm transformer encoder (2 layers, 4 heads, $d_\text{model}=64$, ~147k parameters) achieving 87.0% ensemble held-out accuracy at $B=20$ symmetric (seed 42), with statistically significant gains over the Conv1D baseline ($p=0.002$, Cohen's $d=1.36$ on accuracy)
- 23 new behavioural features (16 derived temporal + 7 GraphQL recency), expanding the per-timestep feature count from ~412 to 435
- Causal (pre-commit only) windowing mode for deployment-realistic evaluation; transformer holds 0.849–0.852 across $B=5$ to $B=40$ while the Conv1D collapses to 0.694 at causal $B=5$
- External validation on 212 CVEfixes repositories (88.4% / 89.3% F1, seed 42, $B=20$ symmetric near-domain) and on a 29-repository truly-unseen subset (90.7% accuracy / 91.4% F1 / 97.4% recall at causal $B=20$)
- 19 methodological corrections to the inherited codebase, most notably replacing standard k-fold with repository-grouped GroupKFold (k=10), which removed ~6 percentage points of leakage-inflated accuracy

## Project Structure

```
.
├── train.py                        # Main training pipeline (GroupKFold CV, CLI)
├── models.py                       # Model definitions: transformer, Conv1D, custom Keras layers
├── dataset_utils.py                # Data loading, feature engineering, windowing
├── helper.py                       # Shared utilities
│
├── analyse.py                      # Statistical analysis (paired t/Wilcoxon, ablation, permutation importance, error analysis)
├── generate_confusion_matrices.py  # Confusion matrix figures (single + grids)
├── generate_attention_heatmap.py   # Aggregated attention heatmaps
├── generate_attention_perhead.py   # Per-head attention specialisation plots
├── compute_paired_stats.py         # Standalone paired-fold statistics from saved results
├── check_test_size.py              # Sanity-check held-out test sample counts per (B, mode)
│
├── convert_cvefixes.py             # Step 1: extract CVEfixes SQLite to per-repo CSVs
├── scrape_cvefixes_repos.py        # Step 2: scrape GitHub events for CVEfixes repos
├── enrich_cvefixes_events.py       # Step 3: enrich CVEfixes CSVs with additional event types via GraphQL
├── evaluate_cvefixes.py            # Step 4: evaluate trained models on CVEfixes (balanced + realistic modes)
│
├── run_comparison.sh               # Seed 42: 4 windows × 2 modes × 2 models = 16 training runs
├── run_comparison_extra_seeds.sh   # Repeat across seeds 123, 7, 256, 999
├── run_cvefixes_windows.sh         # CVEfixes external validation across window sizes
├── run_unseen_sweep.sh             # Truly-unseen 16-cell sweep (with --exclude-train-overlap)
│
├── requirements.txt                # Python dependencies
└── readme.md
```

## Setup

```bash
git clone <this-repo> SecurityPatchDetection
cd SecurityPatchDetection

python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install tensorflow coloredlogs
```

Tested with Python 3.12 and TensorFlow 2.16+. GPU recommended (the dissertation experiments used an NVIDIA RTX 3060 Laptop with 6 GB VRAM); CPU-only runs work but are 10–20× slower.

### Determinism

All experiments fix the random seed via `--seed` and enable TensorFlow's deterministic kernels (`TF_DETERMINISTIC_OPS=1` is set automatically inside `train.py`). File ordering is pinned via `sorted(os.listdir(...))`. Re-running with the same seed produces bit-identical metrics.

## Data

Training data is **not** included in this repository (~50 GB). It is constructed from GitHub Archive event streams; see `data_collection/create_dataset.py` in the [original repository](https://github.com/nitzanfarhi/SecurityPatchDetection) for the collection pipeline. The processed dataset is hosted on Kaggle:

> https://www.kaggle.com/datasets/nitzanfarhi/detecting-security-patches-via-behavioral-data

Place the per-repository CSVs under `data_collection/gh_cve_proccessed/`. Repository metadata goes in `data_collection/Test_data/repo_metadata.json`.

The CVEfixes external validation set is derived from [CVEfixes v1.0.7](https://zenodo.org/record/7029359) (Bhandari et al., 2021). The `convert_cvefixes.py → scrape_cvefixes_repos.py → enrich_cvefixes_events.py` pipeline produces enriched CSVs in `data_collection/cvefixes_test/enriched/`.

## Usage

### 1. Train a single model

```bash
# Transformer, symmetric window B=20, 10-fold GroupKFold, seed 42
python train.py -a before --model transformer -k 10 --metadata -B 20 -s 42

# Conv1D baseline, same configuration
python train.py -a before --model conv1d -k 10 --metadata -B 20 -s 42

# Causal (pre-commit only) window — deployment-realistic
python train.py -a only_before --model transformer -k 10 --metadata -B 20 -s 42

# Different seed
python train.py -a before --model transformer -k 10 --metadata -B 20 -s 123
```

**Key arguments** (`python train.py --help` for full list):

| Flag | Description | Default |
|------|-------------|---------|
| `-a` | Aggregation/window mode: `before` (symmetric) or `only_before` (causal) | required |
| `--model` | `transformer` or `conv1d` | required |
| `-k` | Number of GroupKFold folds | 10 |
| `-B` / `--backs` | Window size (events on each side, or before, depending on `-a`) | 10 |
| `--metadata` / `--meta` | Include static repository metadata features | off |
| `-s` / `--seed` | Random seed | 42 |
| `-e` | Max training epochs (early stopping with patience 8) | 50 |
| `--ratio` | Benign:vuln sampling ratio per repository | 1 |
| `-c` | Save confusion matrix data | off |
| `--window-crop` | Augment with random event masking | off |

Outputs:
- `models/Aggregate.<mode>_R1_B<B>_meta[_s<seed>]/` — saved `.keras` model files (one per ensemble fold + the single best)
- `results/Aggregate.<mode>_R1_B<B>_meta[_s<seed>]_<model>.txt` — accuracy, F1, confusion matrix at the swept-best threshold
- `<run>.log` — per-fold accuracy, F1, threshold, AUC

### 2. Run the full 80-config sweep

The dissertation reports 80 configurations: 4 window sizes × 2 modes × 2 models × 5 seeds.

```bash
# Seed 42: 16 runs (4 windows × 2 modes × 2 models)
bash run_comparison.sh

# Remaining four seeds: 64 runs
bash run_comparison_extra_seeds.sh
```

Total wall time on a single RTX 3060 Laptop: ~25–35 hours. Each individual training run takes 15–30 minutes. Each full run including dataset prep takes ~1 hour 20 minutes

### 3. Statistical analysis and figures

After all 80 runs are saved under `results/`:

```bash
# Full statistical suite (paired t-test, Wilcoxon, bootstrap CI, Cohen's d, ablation, permutation importance)
python analyse.py -a before --model transformer --metadata -B 20 -s 42

# Confusion matrix figures (single panels + grouped grids)
python generate_confusion_matrices.py

# Per-head attention specialisation (requires saved transformer model)
python generate_attention_perhead.py -a before -B 20 --metadata -s 42

# Aggregated attention heatmaps
python generate_attention_heatmap.py -a before -B 20 --metadata -s 42

# Standalone paired-fold significance from saved results
python compute_paired_stats.py
```

### 4. CVEfixes external validation

```bash
# Step 1: extract CVEfixes SQLite to per-repo CSVs (drops repos with <2 CVE-fix commits)
python convert_cvefixes.py

# Step 2: scrape GitHub events for the extracted repos (requires GITHUB_TOKEN env var)
export GITHUB_TOKEN=ghp_...
python scrape_cvefixes_repos.py

# Step 3: enrich with additional event types via GraphQL (forks, issues, PRs, releases, stargazers + inferred PushEvents)
python enrich_cvefixes_events.py

# Step 4a: evaluate on the full 212-repository near-domain set (1:1 balanced sampling)
python evaluate_cvefixes.py -a before --model transformer --backs 20 --meta --enriched --seed 42

# Step 4b: evaluate on the truly-unseen subset (29 repos, no name match + <1% commit-hash overlap with training)
python evaluate_cvefixes.py -a before --model transformer --backs 20 --meta --enriched --seed 42 --exclude-train-overlap

# Step 4c: realistic mode — natural class imbalance (test on ALL commits per repo, no benign subsampling)
python evaluate_cvefixes.py -a before --model transformer --backs 20 --meta --enriched --seed 42 --realistic
```

Sweep across all 16 (architecture × window × mode) cells on the truly-unseen subset:

```bash
bash run_unseen_sweep.sh
```

Sweep across window sizes on the near-domain 212-repo set:

```bash
bash run_cvefixes_windows.sh
```

`evaluate_cvefixes.py` flags worth knowing:

| Flag | Description |
|------|-------------|
| `--enriched` | Use enriched CSVs from `data_collection/cvefixes_test/enriched/` (recovers 5+1 missing event types; +4.3 points accuracy) |
| `--exclude-train-overlap` | Drop test repos with case-insensitive name match or ≥1% commit-hash overlap against training |
| `--realistic` | Natural imbalance: test on every commit in every repo (no 1:1 sampling). Reports ROC-AUC, AP, FPR. |
| `--all-seeds` | Iterate over all 5 experimental seeds |
| `--threshold` | Decision threshold for binarising probabilities (default 0.5; the script also sweeps for the optimal) |

## Reproducing the headline results

The main reported numbers correspond to specific commands:

| Result | Command |
|--------|---------|
| Best held-out: 87.0% / 87.6% F1 (seed 42, $B=20$ sym, transformer ensemble) | `python train.py -a before --model transformer -k 10 --metadata -B 20 -s 42` |
| Conv1D collapse at causal $B=5$ (0.694 ± 0.024) | `python train.py -a only_before --model conv1d -k 10 --metadata -B 5 -s 42` (and other seeds via `run_comparison_extra_seeds.sh`) |
| Statistical significance ($p=0.002$, $d=1.36$) | `python analyse.py -a before --model transformer --metadata -B 20 -s 42` |
| CVEfixes near-domain 88.4% / 89.3% (seed 42, $B=20$ sym) | `python evaluate_cvefixes.py -a before --model transformer --backs 20 --meta --enriched --seed 42` |
| Truly-unseen causal $B=20$ headline (90.7% / 91.4% / 97.4%) | `python evaluate_cvefixes.py -a only_before --model transformer --backs 20 --meta --enriched --seed 42 --exclude-train-overlap` |
| Natural-imbalance ROC-AUC at $B=20$ sym | `python evaluate_cvefixes.py -a before --model transformer --backs 20 --meta --enriched --seed 42 --realistic` |

## Citation

Builds on the work of Farhi et al.:

```bibtex
@article{farhi2023detecting,
  title={Detecting Security Patches via Behavioral Data in Code Repositories},
  author={Farhi, Nitzan and Koenigstein, Noam and Shavitt, Yuval},
  journal={arXiv preprint arXiv:2302.02112},
  year={2023}
}
```

Original repository: https://github.com/nitzanfarhi/SecurityPatchDetection
