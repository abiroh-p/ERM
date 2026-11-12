"""Reproduce the integrity audit, merged results, comparisons and scientific figures.

Run: python scripts/rebuild_paired_results.py
Dependencies: numpy, pandas, scipy, scikit-learn, matplotlib. No GPUs or article
text required. The supplied account ZIPs and archived metrics remain unchanged.
"""
import hashlib
import itertools
import json
import pathlib
import sys
import tempfile
import zipfile

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO = pathlib.Path(__file__).resolve().parents[1]
ROOT = REPO/'results/paired'
sys.path.insert(0, str(REPO/'src/experiments'))
sys.path.insert(0, str(REPO/'src/analysis'))
import paired_sampling as P
import analyze_paired as A
import confirmatory as C

MODELS = ['tfidf', 'distilbert', 'roberta']
LABELS = dict(tfidf='TF-IDF', distilbert='DistilBERT', roberta='RoBERTa')
ALLOCATIONS = {'A':[0,1,2], 'B':[3,4,5], 'C':[6,7]}
COMMON = [('E_vs_R_E','E','R_E'),('L_vs_C0','L','C0'),
          ('LE_vs_L_RE','LE','L_RE'),('SLE_vs_L_RSE','SLE','L_RSE')]
SEEDS = list(range(8))


def summary(values):
    # Independent implementation of the reported mean/t-interval and sign count.
    v=np.asarray(values,dtype=float)
    assert len(v)==8 and np.isfinite(v).all()
    m=float(v.mean()); h=float(stats.t.ppf(.975,7)*v.std(ddof=1)/np.sqrt(8))
    return dict(mean=m,lo=m-h,hi=m+h,same_sign=int((np.sign(v)==np.sign(m)).sum()),n_seeds=8)


def cross_gaps(d):
    inside=d[d.kind=='in'].set_index(['seed','cond','train']).auc
    x=d[d.kind=='cross'].copy()
    x['gap']=[inside.loc[(r.seed,r.cond,r.train)]-r.auc for r in x.itertuples()]
    return x


def contrast_values(d,ca,cb):
    x=d[d.kind=='cross']
    a=x[x.cond==ca].set_index(['seed','train','test']).auc
    b=x[x.cond==cb].set_index(['seed','train','test']).auc
    assert a.index.equals(b.index) or set(a.index)==set(b.index)
    return a-b


def main():
    (ROOT/'statistics').mkdir(exist_ok=True);(ROOT/'figures').mkdir(exist_ok=True)
    merged={}; integrity={}; manifest_hashes={}; archive_hashes={}
    old_analysis_hash=json.loads((ROOT/'legacy_freeze_manifest.json').read_text())['files']['extra/analysis_set.csv']['sha256']
    with tempfile.TemporaryDirectory() as temp:
        temp=pathlib.Path(temp)
        for account in ALLOCATIONS:
            archive=ROOT/f'accounts/ERM_Paired_Account_{account}.zip'
            archive_hashes[account]=P.file_hash(archive)
            with zipfile.ZipFile(archive) as z:
                assert z.testzip() is None
                for name in z.namelist():assert (temp/name).resolve().is_relative_to(temp.resolve())
                z.extractall(temp)
        for model in MODELS:
            frames=[];protocols=[]
            for account,seeds in ALLOCATIONS.items():
                folder=temp/f'erm_paired_account_{account}'/model
                d=pd.read_csv(folder/f'results_{model}.csv'); assert sorted(d.seed.unique())==seeds
                protocol=json.loads((folder/f'results_{model}_protocol.json').read_text());protocols.append(protocol)
                assert protocol['analysis_sha256']==old_analysis_hash
                assert protocol['entity_method']=='spacy:en_core_web_sm'
                for code_name,expected in protocol['code_sha256'].items():
                    assert P.file_hash(REPO/'src/experiments'/code_name)==expected
                config=json.loads((folder/f'results_{model}_config.json').read_text())
                assert config['python']=='3.13.16'
                for seed in seeds:
                    value=json.loads((folder/f'results_{model}_samples/seed_{seed}.json').read_text())
                    recorded=value.pop('manifest_sha256');assert P.digest(value)==recorded
                    assert value['seed']==seed and value['protocol']==P.PROTOCOL
                    if seed in manifest_hashes:assert manifest_hashes[seed]==recorded
                    manifest_hashes[seed]=recorded
                    samples={}
                    for rec in value['samples']:
                        rows=rec['rows'];assert P.digest(rows)==rec['sample_sha256']
                        ids=[r[0] for r in rows];assert len(ids)==len(set(ids))
                        assert set(r[1] for r in rows)=={0,1} and sum(r[1] for r in rows)*2==len(rows)
                        samples[rec['dataset'],rec['role'],rec['length_matched']]=rec
                    for ds in protocol['datasets']:
                        for length in [False,True]:
                            tr=samples[ds,'tr',length]['rows'];te=samples[ds,'in_',length]['rows']
                            assert not (set(r[2] for r in tr)&set(r[2] for r in te))
                    for r in d[d.seed==seed].itertuples():
                        length=bool(A.T.X.CONDITIONS[r.cond][1])
                        tr=samples[r.train,'tr',length];te=samples[r.test,'in_' if r.train==r.test else 'x',length]
                        assert r.manifest_sha256==recorded
                        assert (r.train_sample_sha256,r.test_sample_sha256)==(tr['sample_sha256'],te['sample_sha256'])
                        assert (r.n_train,r.n_test)==(len(tr['rows']),len(te['rows']))
                frames.append(d)
            assert protocols[0]==protocols[1]==protocols[2]
            d=pd.concat(frames,ignore_index=True).sort_values(['seed','cond','train','test']).reset_index(drop=True)
            conditions=A.T.PRIMARY_CONDS if model=='tfidf' else A.FOCUSED
            A.validate(d,SEEDS,conditions,A.T.LEGACY_ORDER)
            assert not d[['auc','f1','acc','ap']].isna().any().any()
            assert np.isfinite(d[['auc','f1','acc','ap']]).all().all()
            # Average precision may exceed 1 by machine roundoff (2.2e-16);
            # tolerate this in validation without modifying the supplied metrics.
            assert d[['auc','f1','acc','ap']].ge(-1e-12).all().all() and d[['auc','f1','acc','ap']].le(1+1e-12).all().all()
            d.to_csv(ROOT/f'merged/results_{model}.csv',index=False)
            merged[model]=d
            integrity[model]=dict(rows=len(d),fits=len(d)//3,seeds=SEEDS,conditions=conditions,
                                  python=config['python'],protocol=protocols[0])
    for field in ['analysis_sha256','spans_sha256','code_sha256','cfg','datasets']:
        assert len({json.dumps(j['protocol'][field],sort_keys=True) for j in integrity.values()})==1
    P.atomic_json(ROOT/'integrity_report.json',dict(status='PASS',models=integrity,
                  matches_frozen_original_analysis_csv=True,account_sha256=archive_hashes,manifest_sha256=manifest_hashes,
                  caveat='Checks saved provenance and result consistency; does not rerun training, raw-text matching or NER.'))

    summaries=[];conditions=[];pairs=[];shares=[];seed_contrasts=[]
    for model,d in merged.items():
        table=A.summaries(d);table['arm']=model;summaries.append(table)
        cross=cross_gaps(d)
        for cond in sorted(d.cond.unique()):
            inside=d[(d.cond==cond)&(d.kind=='in')].groupby('seed').auc.mean()
            x=cross[cross.cond==cond].groupby('seed').agg(cross_auc=('auc','mean'),gap=('gap','mean'))
            for metric,values in [('in_auc',inside),('cross_auc',x.cross_auc),('gap',x.gap)]:
                conditions.append(dict(model=model,condition=cond,metric=metric,**summary(values)))
        gaps=cross.groupby(['seed','cond']).gap.mean().unstack()
        for cond in ['LE','SLE']:
            values=100*(1-gaps[cond]/gaps.C0)
            shares.append(dict(model=model,condition=cond,metric='percent_baseline_gap_removed',**summary(values)))
        available=C.PRIMARY if model=='tfidf' else COMMON
        for name,ca,cb in available:
            values=contrast_values(d,ca,cb)
            pooled=values.groupby(level='seed').mean()
            ours=summary(pooled)
            official=table[(table.metric=='auc')&(table.contrast==name)].iloc[0]
            for field in ['mean','lo','hi']:assert abs(ours[field]-official[field])<1e-12
            for seed,value in pooled.items():seed_contrasts.append(dict(model=model,contrast=name,seed=int(seed),delta_auc=float(value)))
            for (train,test),v in values.groupby(level=['train','test']):
                pairs.append(dict(model=model,contrast=name,train=train,test=test,**summary(v)))
    pd.concat(summaries,ignore_index=True).to_csv(ROOT/'statistics/pooled_contrasts.csv',index=False)
    pd.DataFrame(conditions).to_csv(ROOT/'statistics/condition_metrics.csv',index=False)
    pd.DataFrame(pairs).to_csv(ROOT/'statistics/per_pair_contrasts.csv',index=False)
    pd.DataFrame(shares).to_csv(ROOT/'statistics/gap_shares.csv',index=False)
    pd.DataFrame(seed_contrasts).to_csv(ROOT/'statistics/per_seed_contrasts.csv',index=False)

    sources=json.loads((ROOT/'legacy_sources.json').read_text());comparisons=[]
    for model,d in merged.items():
        old=pd.concat([pd.read_csv(REPO/p) for p in sources[model]],ignore_index=True)
        for name,ca,cb in (C.PRIMARY if model=='tfidf' else COMMON):
            for version,df in [('archived',old),('paired',d)]:
                values=contrast_values(df,ca,cb).groupby(level='seed').mean()
                comparisons.append(dict(model=model,contrast=name,protocol=version,**summary(values)))
    pd.DataFrame(comparisons).to_csv(ROOT/'statistics/archived_vs_paired.csv',index=False)

    colors=['#0072B2','#D55E00','#009E73'];marks=['o','s','^']
    names=['E minus random E','L minus baseline','L+E minus L+random E','S+L+E minus L+random(S+E)']
    plt.rcParams.update({'font.size':14})
    fig,ax=plt.subplots(figsize=(9,5.4))
    stats_table=pd.concat(summaries,ignore_index=True)
    for j,model in enumerate(MODELS):
        for i,(name,_,_) in enumerate(COMMON):
            s=stats_table[(stats_table.arm==model)&(stats_table.metric=='auc')&(stats_table.contrast==name)].iloc[0]
            ax.errorbar(s['mean'],i+(j-1)*.20,xerr=[[s['mean']-s['lo']],[s['hi']-s['mean']]],
                        fmt=marks[j],color=colors[j],capsize=3,markersize=5,
                        label=LABELS[model] if i==0 else None)
    ax.axvline(0,color='#333333',lw=1);ax.set_yticks(range(4),names)
    ax.invert_yaxis();ax.grid(axis='x',alpha=.22)
    ax.set_xlabel('Change in cross-dataset AUC')
    ax.set_title('Document-paired validation',loc='left',pad=16)
    ax.legend(loc='upper center',bbox_to_anchor=(0.5,-0.20),ncol=3,frameon=False);ax.spines[['top','right']].set_visible(False)
    fig.tight_layout()
    for ext in ['png','svg','pdf']:fig.savefig(ROOT/f'figures/paired_validation_contrasts.{ext}',dpi=300,bbox_inches='tight')
    plt.close(fig)
    P.atomic_json(ROOT/'statistics/analysis_metadata.json',dict(
        input_sha256={str(p.relative_to(ROOT)):P.file_hash(p) for p in sorted((ROOT/'merged').glob('*.csv'))},
        inference='Descriptive t-intervals over 8 seeds; six pairs averaged within seed; no population-level causal or equivalence claim.',
        corrections='Holm over 6 planned contrasts per model and outcome; omitted transformer contrasts assigned p=1.',
        limitations='Legacy-vs-paired differences also involve software changes; transformer validation uses 8 of 13 conditions.'))
    print('PASS: integrity, original-data hash, complete 2,088 rows / 696 fits, paired samples, independent statistics and figures')
    print(pd.concat(summaries).query("metric == 'auc' and family == 'planned_descriptive'")[['model','contrast','mean','lo','hi','p_holm_six']].to_string(index=False))

if __name__=='__main__':main()
