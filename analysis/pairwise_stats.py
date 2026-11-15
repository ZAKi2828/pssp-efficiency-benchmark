"""All 10 pairwise model comparisons on CB513 (McNemar, Wilcoxon, chain-level bootstrap) with
Holm correction, repeated for every seed, plus chain-level bootstrap 95% CIs of Q3 on every test set.

Inputs : results_real/predictions.npz        (per-residue predictions written by the Colab notebook)
         results_real/chain_eval_counts.json  (evaluated residues per chain, in file order; regenerate
                                               with --recount, which re-reads the public CSVs exactly
                                               as the notebook does)
Outputs: results_real/pairwise_all.csv, results_real/q3_bootstrap_ci.csv
Run from the repository root:  python analysis/pairwise_stats.py
"""
import itertools, json, re, sys
import numpy as np, pandas as pd
from scipy import stats

R = 'results_real'
ORDER = ['WindowMLP', 'Hybrid 1D-CNN-BiLSTM', 'ESM-2 (8M) Linear Probe', 'ESM-2 (8M) MLP Head', 'ESM-2 (35M) MLP Head']
SEEDS = [42, 123, 2026]
B = 10000

def recount():
    base = 'https://huggingface.co/datasets/proteinea/secondary_structure_prediction/resolve/main/'
    ss = {'H': 0, 'G': 0, 'I': 0, 'E': 1, 'B': 1, 'T': 2, 'S': 2, 'C': 2, '-': 2, ' ': 2, 'L': 2, 'P': 2}
    num = re.compile(r'[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?')
    out = {}
    for t in ['CB513', 'TS115', 'CASP12', 'CASP13', 'CASP14']:
        df = pd.read_csv(base + t + '.csv'); counts = []
        for _, r in df.iterrows():
            seq = str(r['input']).strip().upper(); L = len(seq); s8 = str(r['dssp8']).strip()
            y = np.array([ss.get(c, -1) for c in s8]) if len(s8) == L else np.array([{'H': 0, 'E': 1, 'C': 2}.get(c, -1) for c in str(r['dssp3']).strip()])
            m = y >= 0
            for col in ['disorder', 'cb513_mask']:
                if col in df.columns and isinstance(r[col], str):
                    v = np.array([float(x) for x in num.findall(r[col])])
                    if len(v) == L: m &= v > 0.5
            counts.append(int(m.sum()))
        out[t] = counts
    json.dump(out, open(f'{R}/chain_eval_counts.json', 'w'))

if '--recount' in sys.argv: recount()
C = json.load(open(f'{R}/chain_eval_counts.json')); P = np.load(f'{R}/predictions.npz')

def chains(m, seed, t):
    T = P[f'{m}|{seed}|{t}|true']; Pr = P[f'{m}|{seed}|{t}|pred']; n = np.array(C[t])
    assert n.sum() == len(T), 'chain counts do not match predictions'
    idx = np.cumsum(np.r_[0, n]); keep = n > 0
    corr = np.array([np.sum(T[a:b] == Pr[a:b]) for a, b in zip(idx[:-1], idx[1:])])
    return T, Pr, corr[keep], n[keep]

def compare(a, b, seed, t='CB513'):
    T, pa, ca, n = chains(a, seed, t); _, pb, cb, _ = chains(b, seed, t)
    A, Bc = pa == T, pb == T; b10 = int(np.sum(A & ~Bc)); c01 = int(np.sum(~A & Bc))
    chi2 = (abs(b10 - c01) - 1) ** 2 / (b10 + c01); log10_mc = stats.chi2.logsf(chi2, 1) / np.log(10)
    d = 100 * ca / n - 100 * cb / n; d = d[d != 0]; r = stats.rankdata(np.abs(d))
    W = min(r[d > 0].sum(), r[d < 0].sum()); k = len(d)
    z = (W - k * (k + 1) / 4) / np.sqrt(k * (k + 1) * (2 * k + 1) / 24)
    log10_wx = (np.log(2) + stats.norm.logcdf(z)) / np.log(10)          # normal approximation (two-sided)
    rng = np.random.RandomState(42); idx = rng.randint(0, len(n), size=(B, len(n)))
    bd = 100 * (ca[idx].sum(1) - cb[idx].sum(1)) / n[idx].sum(1)
    pb_ = max(2 * min(np.mean(bd <= 0), np.mean(bd >= 0)), 1 / B)
    return dict(A=a, B=b, seed=seed, N_residues=len(T), n_chains=len(n), McNemar_chi2=chi2, McNemar_log10p=log10_mc,
                Wilcoxon_W=W, Wilcoxon_log10p=log10_wx, dQ3=100 * (ca.sum() - cb.sum()) / n.sum(),
                CI_low=np.percentile(bd, 2.5), CI_high=np.percentile(bd, 97.5), Boot_p=pb_)

def holm_log10(logp):
    logp = np.asarray(logp); o = np.argsort(logp); m = len(logp); adj = np.empty(m); run = -np.inf
    for k, i in enumerate(o):
        run = max(run, min(0.0, logp[i] + np.log10(m - k))); adj[i] = run
    return adj

rows = []
for s in SEEDS:
    res = [compare(b, a, s) for a, b in itertools.combinations(ORDER, 2)]   # A = the larger/later model
    for key in ['McNemar_log10p', 'Wilcoxon_log10p']:
        for r, v in zip(res, holm_log10([r[key] for r in res])): r[key + '_Holm'] = v
    for r, v in zip(res, holm_log10(np.log10([r['Boot_p'] for r in res]))): r['Boot_p_Holm'] = 10 ** v
    rows += res
pd.DataFrame(rows).to_csv(f'{R}/pairwise_all.csv', index=False)

ci = []
for t in C:
    for m in ORDER:
        cs = []
        for s in SEEDS:
            _, _, c, n = chains(m, s, t); cs.append(c)
        c = np.mean(cs, 0); rng = np.random.RandomState(42); idx = rng.randint(0, len(n), size=(B, len(n)))
        q = 100 * c[idx].sum(1) / n[idx].sum(1)
        ci.append(dict(test_set=t, model=m, n_chains=len(n), Q3=100 * c.sum() / n.sum(),
                       CI_low=np.percentile(q, 2.5), CI_high=np.percentile(q, 97.5)))
pd.DataFrame(ci).to_csv(f'{R}/q3_bootstrap_ci.csv', index=False)
print('written pairwise_all.csv and q3_bootstrap_ci.csv')
