# Sequence-Only Protein Secondary Structure Prediction under a CPU Budget: Accuracy–Latency Trade-offs of Frozen ESM-2 Probes and Lightweight Non-Pretrained Networks

Code, data pointers, results and manuscript for a reproducible benchmark of five MSA-free three-state secondary structure predictors: a windowed MLP, a hybrid 1D-CNN–BiLSTM, and frozen ESM-2 (8M, 35M) language models with linear or MLP heads.

**Author:** Roaissat Mohamed Zakaria (Independent Researcher, Ain El Melh, Algeria) · target journal: *Computational Biology and Chemistry* (Elsevier).

## Contribution

This is a benchmark study; it does not propose a new architecture. It provides:

1. one protocol (same homology-filtered training data, evaluation masks, seeds and stopping rule) for five predictors whose CPU cost spans more than two orders of magnitude;
2. end-to-end latency at batch size 1 on CPU and GPU, reported with parameter counts and weight size;
3. a linear-versus-MLP probe ablation and an 8M-versus-35M backbone comparison on frozen ESM-2 features;
4. results on five test sets (CB513, TS115, CASP12, CASP13, CASP14) with chain-level bootstrap confidence intervals and all ten pairwise significance tests (Holm-corrected).

## Main results (mean ± SD over seeds 42, 123, 2026)

| Model | CB513 Q3 (%) | CB513 SOV'99 (%) | TS115 Q3 (%) | CASP14 Q3 (%) | CPU ms/chain | Params (total / trainable) |
|---|---|---|---|---|---|---|
| WindowMLP | 67.95 ± 0.05 | 62.02 ± 0.25 | 70.20 ± 0.11 | 66.54 ± 0.22 | 1.50 ± 0.08 | 54 467 / 54 467 |
| Hybrid 1D-CNN-BiLSTM | 71.45 ± 0.22 | 67.66 ± 0.40 | 73.87 ± 0.36 | 71.56 ± 0.96 | 19.07 ± 7.84 | 690 519 / 690 519 |
| ESM-2 (8M) linear probe | 74.31 ± 0.02 | 67.46 ± 0.10 | 77.13 ± 0.01 | 72.66 ± 0.20 | 137.75 ± 2.58 | 7 410 044 / 963 |
| ESM-2 (8M) MLP head | 75.92 ± 0.03 | 70.08 ± 0.34 | 78.73 ± 0.04 | 72.79 ± 0.20 | 141.42 ± 2.57 | 7 429 820 / 20 739 |
| ESM-2 (35M) MLP head | 80.12 ± 0.05 | 74.45 ± 0.13 | 81.88 ± 0.08 | 74.76 ± 0.17 | 478.34 ± 30.70 | 33 300 500 / 30 979 |

CPU latency: mean ± SD over ten interleaved rounds on the first 100 CB513 chains, Intel Xeon 2.00 GHz (Google Colab), 2 threads, batch size 1 (`results_real/efficiency/`). The hybrid network is about 9x faster than the ESM-2 (8M) pipeline on CPU (median within-round ratio 9.1, IQR 7.8–9.5) but slower on GPU. Every model runs in under 1 GB of RAM. The CASP sets are small (12–20 chains); their 95% confidence intervals are 4–11 points wide (`results_real/q3_bootstrap_ci.csv`).

## Repository structure

| Path | Content |
|---|---|
| `colab/PSSP_real_benchmark.ipynb` | Full pipeline: data download, MMseqs2 leakage filter, training of all models (3 seeds), evaluation, timing, statistics |
| `colab/PSSP_efficiency_benchmark.ipynb` | CPU/GPU latency and peak memory: one process per model, ten interleaved rounds in random model order |
| `analysis/pairwise_stats.py` | All 10 pairwise tests (McNemar, Wilcoxon, chain bootstrap) with Holm correction for every seed, and Q3 bootstrap CIs |
| `make_figs.py` | Regenerates every figure and the graphical abstract from `results_real/` |
| `results_real/` | All outputs of the runs (see below) |
| `submission/` | Separate figure files (PNG 300 dpi + PDF), graphical abstract, highlights for journal submission |
| `Roaissat_PSSP_Manuscript_v4.tex` (also in `paper/`), `.docx` | Manuscript |
| `Highlights.txt`, `Cover_Letter.txt` | Submission texts |

### Experiment log (`results_real/`)

| File | Content |
|---|---|
| `environment.json` | GPU, CPU, library versions, date of the run |
| `hyperparameters.json` | Exact optimizer, learning rate, batch size, epochs and patience per model |
| `dataset_stats.json`, `leakage_filter.json`, `leakage_hits.csv` | Data sizes and the MMseqs2 homology filter (202 training chains removed) |
| `all_seed_runs.csv`, `summary_metrics.csv` | Every metric for every model and seed, and their mean/SD |
| `loss_curves.json` | Per-epoch training and validation loss for every run |
| `confusion_matrices.csv` | Confusion matrices (seed 42) |
| `predictions.npz` | Per-residue true labels and predictions for every model, seed and test set |
| `statistical_tests.csv` | Original three comparisons computed in the notebook |
| `pairwise_all.csv`, `q3_bootstrap_ci.csv`, `chain_eval_counts.json` | All ten comparisons for all seeds, Q3 confidence intervals, per-chain residue counts |
| `efficiency/` | Interleaved latency rounds, paired speed ratios, peak RSS and GPU memory (used in the paper) |
| `efficiency_run1_sequential_superseded/` | First, non-interleaved timing run; superseded because of machine noise, kept for transparency |

## Reproduce

1. Open `colab/PSSP_real_benchmark.ipynb` in Google Colab with a T4 GPU. Run once with `QUICK_TEST = True` (5–10 min), then with `QUICK_TEST = False` (about 1–2 h). It writes `results_real/`.
2. Run `colab/PSSP_efficiency_benchmark.ipynb` (about 20 min) for the latency and peak-memory measurements; copy its output to `results_real/efficiency/`.
3. Locally, from the repository root:
   ```bash
   pip install -r requirements.txt
   python analysis/pairwise_stats.py   # pairwise tests and confidence intervals
   python make_figs.py                 # figures
   ```

Data: NetSurfP-2.0 training set and the CB513, TS115, CASP12, CASP13 and CASP14 test sets from the Hugging Face dataset `proteinea/secondary_structure_prediction` (DOI 10.57967/hf/1104).

## Software environment

Runs were made on Google Colab (Tesla T4, Intel Xeon 2.00 GHz, 2 vCPUs) with Python 3.13.16, PyTorch 2.14.1 (CUDA 13.0) and Transformers 5.22.1 (`results_real/environment.json`). The same PyTorch and Transformers versions, and the NumPy/SciPy/pandas/Matplotlib versions used for the analysis scripts, are pinned in `requirements.txt`.

## Use of AI tools

Claude (Anthropic) was used to help draft and review the code and to draft, edit and proofread the manuscript text. The author ran all experiments, reviewed all content and takes full responsibility for it.
