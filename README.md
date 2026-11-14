# Sequence-only protein secondary structure prediction under a CPU budget

Reproducible benchmark of five MSA-free secondary structure predictors (WindowMLP, hybrid 1D-CNN-BiLSTM, frozen ESM-2 8M linear probe, frozen ESM-2 8M + MLP head, frozen ESM-2 35M + MLP head).

**Author:** Roaissat Mohamed Zakaria (Independent Researcher, Algeria) — target journal: *Computational Biology and Chemistry*.

## Reproduce
Open `colab/PSSP_real_benchmark.ipynb` in Google Colab (T4 GPU), set `QUICK_TEST = False` and run all cells (~1–2 h). The notebook downloads the public data (`proteinea/secondary_structure_prediction`, DOI 10.57967/hf/1104), removes training chains with >=25% identity to any test chain (MMseqs2), trains all models with seeds 42, 123 and 2026, and writes every result to `results_real/`.

## Main results (CB513, mean ± SD over 3 seeds)

| Model | CB513 Q3 | CB513 SOV'99 | TS115 Q3 | CASP14 Q3 | CPU ms/seq | Params (total / trainable) |
|---|---|---|---|---|---|---|
| WindowMLP | 67.95 ± 0.05 | 62.02 ± 0.25 | 70.20 ± 0.11 | 66.54 ± 0.22 | 1.08 | 54,467 / 54,467 |
| Hybrid 1D-CNN-BiLSTM | 71.45 ± 0.22 | 67.66 ± 0.40 | 73.87 ± 0.36 | 71.56 ± 0.96 | 14.18 | 690,519 / 690,519 |
| ESM-2 (8M) linear probe | 74.31 ± 0.02 | 67.46 ± 0.10 | 77.13 ± 0.01 | 72.66 ± 0.20 | 83.05 | 7,410,044 / 963 |
| ESM-2 (8M) MLP head | 75.92 ± 0.03 | 70.08 ± 0.34 | 78.73 ± 0.04 | 72.79 ± 0.20 | 82.55 | 7,429,820 / 20,739 |
| ESM-2 (35M) MLP head | 80.12 ± 0.05 | 74.45 ± 0.13 | 81.88 ± 0.08 | 74.76 ± 0.17 | 329.08 | 33,300,500 / 30,979 |

CPU: Intel Xeon 2.00 GHz, 2 threads, batch size 1 (Google Colab). Full results: `results_real/`.

## Files
- `colab/PSSP_real_benchmark.ipynb` — the complete training/evaluation pipeline
- `results_real/` — per-seed results, summary, statistical tests, loss curves, leakage-filter hits, predictions, figures
- `make_figs.py` — regenerates the figures from `results_real/`
- `paper/Roaissat_PSSP_Manuscript_v4.tex`, `Highlights.txt`, `Cover_Letter.txt`
