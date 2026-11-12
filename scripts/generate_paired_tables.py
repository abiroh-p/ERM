from pathlib import Path
import pandas as pd, numpy as np, shutil, zipfile, re, json
from scipy import stats
REPO=Path(__file__).resolve().parents[1]; R=REPO/'results/paired'; S=REPO/'paper/sn'; SEC=S/'sections'; T=S.parent/'tables'
D=pd.read_csv(R/'statistics/pooled_contrasts.csv'); M=pd.read_csv(R/'statistics/condition_metrics.csv'); G=pd.read_csv(R/'statistics/gap_shares.csv'); P=pd.read_csv(R/'statistics/per_pair_contrasts.csv'); V=pd.read_csv(R/'statistics/per_seed_contrasts.csv')
models=['tfidf','distilbert','roberta']; labels=dict(tfidf='TF-IDF',distilbert='DistilBERT',roberta='RoBERTa')
contrasts=['E_vs_R_E','L_vs_C0','LE_vs_L_RE','SLE_vs_L_RSE']; cn=dict(zip(contrasts,[r'E $-$ $R_E$',r'L $-$ C0',r'L+E $-$ L+$R_E$',r'S+L+E $-$ L+$R_{SE}$']))
def fmt(x):return f'{x:+.3f}'
def ci(row,sign=True):
 f=fmt if sign else lambda x:f'{x:.3f}'
 return f"{f(row['mean'])} [{f(row['lo'])}, {f(row['hi'])}]"
def table(path,cols,head,rows):
 path.write_text('\\begin{tabular}{@{}'+cols+'@{}}\n\\toprule\n'+head+'\\\\\n\\midrule\n'+'\n'.join(' & '.join(row)+r'\\' for row in rows)+'\n\\bottomrule\n\\end{tabular}\n')
rows=[]
all_contrasts=['S_vs_R_S','E_vs_R_E','SE_vs_R_SE','L_vs_C0','LE_vs_L_RE','SLE_vs_L_RSE']
all_cn={**cn,'S_vs_R_S':r'S $-$ $R_S$','SE_vs_R_SE':r'S+E $-$ $R_{SE}$'}
for c in all_contrasts:
 row=[all_cn[c]]
 for m in models:
  selected=D.query('arm==@m and metric=="auc" and contrast==@c')
  row.append(ci(selected.iloc[0]) if len(selected) else 'not run')
 rows.append(row)
table(T/'tab_paired_pooled.tex','lccc','Contrast & TF-IDF & DistilBERT & RoBERTa',rows)
rows=[]
for m in models:
 for c in all_contrasts:
  selected=D.query('arm==@m and metric=="auc" and contrast==@c')
  if len(selected):
   r=selected.iloc[0]
   rows.append([labels[m],all_cn[c],f'{int(r.same_sign)}/8',f'{r.p_signflip:.4f}',f'{r.p_holm_six:.4f}'])
  else:
   rows.append([labels[m],all_cn[c],'not run','--','1.0000*'])
table(T/'tab_paired_multiplicity.tex','llccc','Model & Contrast & Sign agreement & Exact $p$ & Holm $p$',rows)
rows=[]
for m in models:
 for c in ['C0','LE','SLE','L_RE','L_RSE']:
  row=[labels[m],dict(C0='C0',LE='L+E',SLE='S+L+E',L_RE=r'L+$R_E$',L_RSE=r'L+$R_{SE}$')[c]]
  row += [ci(M.query('model==@m and condition==@c and metric==@met').iloc[0],False) for met in ['cross_auc','gap']]
  rows.append(row)
table(T/'tab_paired_conditions.tex','llcc','Model & Condition & Cross AUC & AUC gap',rows)
rows=[]
for tr,te in [('ISOT','McIntire'),('ISOT','WELFake_residual'),('McIntire','ISOT'),('McIntire','WELFake_residual'),('WELFake_residual','ISOT'),('WELFake_residual','McIntire')]:
 pair=tr.replace('WELFake_residual','residual')+r'$\rightarrow$'+te.replace('WELFake_residual','residual')
 rows.append([pair]+[ci(P.query('model==@m and contrast=="SLE_vs_L_RSE" and train==@tr and test==@te').iloc[0]) for m in models])
table(T/'tab_paired_pairs.tex','lccc','Train $\\rightarrow$ test & TF-IDF & DistilBERT & RoBERTa',rows)
# Save model differences through seed-wise comparisons, rather than comparing significance.
rows=[]; records=[]
for c in contrasts:
 w=V.query('contrast==@c').pivot(index='seed',columns='model',values='delta_auc'); z=w.tfidf-w.distilbert
 h=stats.t.ppf(.975,7)*z.std(ddof=1)/np.sqrt(8);rec=dict(contrast=c,comparison='TF-IDF minus DistilBERT',mean=z.mean(),lo=z.mean()-h,hi=z.mean()+h,same_sign=int((np.sign(z)==np.sign(z.mean())).sum()));records.append(rec);rows.append([cn[c],ci(rec),str(rec['same_sign'])+'/8'])
pd.DataFrame(records).to_csv(R/'statistics/paired_model_differences.csv',index=False)
table(T/'tab_paired_model_diff.tex','lcc','Contrast & TF-IDF $-$ DistilBERT & Sign agreement',rows)
for m in models:
 rows=[]
 for seed in range(8):
  rows.append([str(seed)]+[fmt(V.query('model==@m and contrast==@c and seed==@seed').iloc[0].delta_auc) for c in contrasts])
 table(T/f'tab_paired_seeds_{m}.tex','rcccc','Seed & E $-$ $R_E$ & L $-$ C0 & L+E $-$ L+$R_E$ & S+L+E $-$ L+$R_{SE}$',rows)

shutil.copy2(R/'figures/paired_validation_contrasts.pdf',S.parent/'figs/paired_validation_contrasts.pdf')
