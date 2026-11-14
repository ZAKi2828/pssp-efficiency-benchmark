import json, pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
R='results_real'; r=pd.read_csv(f'{R}/all_seed_runs.csv')
order=['WindowMLP','Hybrid 1D-CNN-BiLSTM','ESM-2 (8M) Linear Probe','ESM-2 (8M) MLP Head','ESM-2 (35M) MLP Head']
col=dict(zip(order,['#d95f02','#1b9e77','#386cb0','#7570b3','#e7298a']))
g=r.groupby('model'); M=g.mean(numeric_only=True); S=g.std(numeric_only=True); cpu=r[r.seed==42].set_index('model')
plt.rcParams.update({'font.size':10})
# Fig 1 loss curves (seed 42)
C=json.load(open(f'{R}/loss_curves.json'))
fig,ax=plt.subplots(figsize=(8.6,5.0),dpi=300)
for m in order:
    h=C[f'{m}|42']; e=range(1,len(h['train'])+1)
    ax.plot(e,h['train'],'-o',ms=3,color=col[m],label=f'{m} (train)')
    ax.plot(e,h['val'],'--s',ms=3,color=col[m],alpha=.85,label=f'{m} (val)')
    b=h['best_epoch']; ax.plot(b,h['val'][b-1],'*',ms=11,color=col[m],mec='k',mew=.6)
ax.set_xlabel('Epoch'); ax.set_ylabel('Cross-entropy loss (per residue)')
ax.spines[['top','right']].set_visible(False); ax.grid(axis='y',ls='--',alpha=.4)
ax.legend(fontsize=7,ncol=2,frameon=False); plt.tight_layout(); plt.savefig(f'{R}/fig_loss_curves.png'); plt.close()
# Fig 2 pareto
fig,(a1,a2)=plt.subplots(1,2,figsize=(12,4.6),dpi=300)
for m in order:
    q=M.loc[m,'CB513_Q3']; s=S.loc[m,'CB513_Q3']
    a1.errorbar(cpu.loc[m,'CPU_ms_per_seq'],q,yerr=s,fmt='o',ms=10,color=col[m],mec='k',label=m)
    a2.errorbar(cpu.loc[m,'Weights_MB_fp32'],q,yerr=s,fmt='o',ms=10,color=col[m],mec='k',label=m)
a1.set_xscale('log'); a2.set_xscale('log')
a1.set_xlabel('CPU latency per sequence (ms, 2 threads, batch 1, log scale)'); a2.set_xlabel('Model weights (MB, fp32, log scale)')
for a,t in [(a1,'(a)'),(a2,'(b)')]:
    a.set_ylabel('CB513 $Q_3$ (%)'); a.grid(ls='--',alpha=.4); a.spines[['top','right']].set_visible(False); a.set_title(t,loc='left',fontweight='bold')
a1.legend(fontsize=8,frameon=False,loc='lower right'); plt.tight_layout(); plt.savefig(f'{R}/fig_pareto.png'); plt.close()
# Fig 3 benchmarks
sets=['CB513','TS115','CASP12','CASP13','CASP14']; n=[int(r[f'{s}_n_proteins'].iloc[0]) for s in sets]
fig,ax=plt.subplots(figsize=(10,4.6),dpi=300); w=.16; x=np.arange(len(sets))
for i,m in enumerate(order):
    ax.bar(x+(i-2)*w,[M.loc[m,f'{s}_Q3'] for s in sets],w,yerr=[S.loc[m,f'{s}_Q3'] for s in sets],color=col[m],edgecolor='k',lw=.5,capsize=2,label=m)
ax.set_xticks(x); ax.set_xticklabels([f'{s}\n(n={k})' for s,k in zip(sets,n)]); ax.set_ylim(60,85); ax.set_ylabel('$Q_3$ (%)')
ax.legend(fontsize=8,ncol=3,frameon=False,loc='upper center',bbox_to_anchor=(.5,1.15)); ax.grid(axis='y',ls='--',alpha=.4); ax.spines[['top','right']].set_visible(False)
plt.tight_layout(); plt.savefig(f'{R}/fig_benchmarks.png'); plt.close()
