"""Regenerate all figures of the manuscript from results_real/ (no hard-coded numbers).
Outputs:
  results_real/fig_*.png                      (used by the LaTeX source)
  submission/Figure_1..3.{png,pdf}            (separate artwork files for submission)
  submission/Graphical_abstract.{tiff,pdf}    (single-panel graphical abstract)
"""
import json, os
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R = 'results_real'; S_DIR = 'submission'; os.makedirs(S_DIR, exist_ok=True)
r = pd.read_csv(f'{R}/all_seed_runs.csv')
order = ['WindowMLP', 'Hybrid 1D-CNN-BiLSTM', 'ESM-2 (8M) Linear Probe', 'ESM-2 (8M) MLP Head', 'ESM-2 (35M) MLP Head']
col = dict(zip(order, ['#d95f02', '#1b9e77', '#386cb0', '#7570b3', '#e7298a']))
g = r.groupby('model'); M = g.mean(numeric_only=True); S = g.std(numeric_only=True)
cpu = r[r.seed == 42].set_index('model')
plt.rcParams.update({'font.size': 10})

def save(fig, name, fig_no):
    fig.savefig(f'{R}/{name}.png', dpi=300)
    fig.savefig(f'{S_DIR}/Figure_{fig_no}.png', dpi=300)
    fig.savefig(f'{S_DIR}/Figure_{fig_no}.pdf')
    plt.close(fig)

# Figure 1: Q3 on the five test sets
sets = ['CB513', 'TS115', 'CASP12', 'CASP13', 'CASP14']
n = [int(r[f'{s}_n_proteins'].iloc[0]) for s in sets]
fig, ax = plt.subplots(figsize=(10, 4.6)); w = .16; x = np.arange(len(sets))
for i, m in enumerate(order):
    ax.bar(x + (i - 2) * w, [M.loc[m, f'{s}_Q3'] for s in sets], w, yerr=[S.loc[m, f'{s}_Q3'] for s in sets],
           color=col[m], edgecolor='k', lw=.5, capsize=2, label=m)
ax.set_xticks(x); ax.set_xticklabels([f'{s}\n(n={k})' for s, k in zip(sets, n)]); ax.set_ylim(60, 85); ax.set_ylabel('$Q_3$ (%)')
ax.legend(fontsize=8, ncol=3, frameon=False, loc='upper center', bbox_to_anchor=(.5, 1.15))
ax.grid(axis='y', ls='--', alpha=.4); ax.spines[['top', 'right']].set_visible(False)
fig.tight_layout(); save(fig, 'fig_benchmarks', 1)

# Figure 2: accuracy-cost trade-off
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6))
for m in order:
    q, s = M.loc[m, 'CB513_Q3'], S.loc[m, 'CB513_Q3']
    a1.errorbar(cpu.loc[m, 'CPU_ms_per_seq'], q, yerr=s, fmt='o', ms=10, color=col[m], mec='k', label=m)
    a2.errorbar(cpu.loc[m, 'Weights_MB_fp32'], q, yerr=s, fmt='o', ms=10, color=col[m], mec='k', label=m)
a1.set_xscale('log'); a2.set_xscale('log')
a1.set_xlabel('CPU latency per sequence (ms, 2 threads, batch 1, log scale)')
a2.set_xlabel('Model weights (MB, fp32, log scale)')
for a, t in [(a1, '(a)'), (a2, '(b)')]:
    a.set_ylabel('CB513 $Q_3$ (%)'); a.grid(ls='--', alpha=.4); a.spines[['top', 'right']].set_visible(False)
    a.set_title(t, loc='left', fontweight='bold')
a1.legend(fontsize=8, frameon=False, loc='lower right')
fig.tight_layout(); save(fig, 'fig_pareto', 2)

# Figure 3: per-epoch losses (seed 42)
C = json.load(open(f'{R}/loss_curves.json'))
fig, ax = plt.subplots(figsize=(8.6, 5.0))
for m in order:
    h = C[f'{m}|42']; e = range(1, len(h['train']) + 1)
    ax.plot(e, h['train'], '-o', ms=3, color=col[m], label=f'{m} (train)')
    ax.plot(e, h['val'], '--s', ms=3, color=col[m], alpha=.85, label=f'{m} (val)')
    b = h['best_epoch']; ax.plot(b, h['val'][b - 1], '*', ms=11, color=col[m], mec='k', mew=.6)
ax.set_xlabel('Epoch'); ax.set_ylabel('Cross-entropy loss (per residue)')
ax.spines[['top', 'right']].set_visible(False); ax.grid(axis='y', ls='--', alpha=.4)
ax.legend(fontsize=7, ncol=2, frameon=False)
fig.tight_layout(); save(fig, 'fig_loss_curves', 3)

# Graphical abstract: one panel, large text, readable at 13 cm width
short = {'WindowMLP': 'WindowMLP', 'Hybrid 1D-CNN-BiLSTM': 'CNN-BiLSTM', 'ESM-2 (8M) Linear Probe': 'ESM-2 8M linear',
         'ESM-2 (8M) MLP Head': 'ESM-2 8M MLP', 'ESM-2 (35M) MLP Head': 'ESM-2 35M MLP'}
off = {'WindowMLP': (8, 6), 'Hybrid 1D-CNN-BiLSTM': (8, -12), 'ESM-2 (8M) Linear Probe': (8, -10),
       'ESM-2 (8M) MLP Head': (-8, 6), 'ESM-2 (35M) MLP Head': (-8, -4)}
fig, ax = plt.subplots(figsize=(5.2, 2.6))
for m in order:
    xv, yv = cpu.loc[m, 'CPU_ms_per_seq'], M.loc[m, 'CB513_Q3']
    ax.scatter(xv, yv, s=60, color=col[m], edgecolor='k', zorder=3)
    dx, dy = off[m]
    ax.annotate(f'{short[m]} ({yv:.1f}%)', (xv, yv), xytext=(dx, dy), textcoords='offset points', fontsize=7.5,
                ha='left' if dx > 0 else 'right', va='center')
ax.set_xscale('log'); ax.set_xlim(0.6, 900); ax.set_ylim(66, 82)
ax.set_xlabel('CPU latency per sequence (ms, log scale)', fontsize=8)
ax.set_ylabel('CB513 $Q_3$ (%)', fontsize=8); ax.tick_params(labelsize=7.5)
ax.set_title('MSA-free secondary structure prediction: accuracy vs. CPU cost', fontsize=8.5)
ax.grid(ls='--', alpha=.4); ax.spines[['top', 'right']].set_visible(False)
fig.tight_layout()
fig.savefig(f'{S_DIR}/Graphical_abstract.tiff', dpi=300, pil_kwargs={'compression': 'tiff_lzw'})
fig.savefig(f'{S_DIR}/Graphical_abstract.png', dpi=300)
fig.savefig(f'{S_DIR}/Graphical_abstract.pdf'); plt.close(fig)
print('figures written')
