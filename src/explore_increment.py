"""Exploratory: cross-validated incremental value of ensemble scores over the confound proxies.
Second-level repeated CV (LR) on repeat-averaged out-of-fold scores; no participant's own label
was used to produce its score."""
import numpy as np, pandas as pd, statsmodels.api as sm
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
CONF=["duration_s","muscle_temporal_emg","line_ratio","drift_ratio","transient_frac","blink_frontal_kurt"]
C=pd.read_csv(__import__("paths").PREV_CONF).set_index("subject")
S=pd.read_csv("results/oof_scores.csv.gz"); avg=S.groupby(["method","subject"]).score.mean().unstack(0)
y=S.drop_duplicates("subject").set_index("subject").loc[avg.index,"label"].values; Cv=C.loc[avg.index,CONF].values
def cvauc(X):
    a=[]
    for r in range(20):
        p=np.zeros(len(y))
        for tr,te in StratifiedKFold(5,shuffle=True,random_state=r).split(X,y):
            p[te]=make_pipeline(StandardScaler(),LogisticRegression(max_iter=5000)).fit(X[tr],y[tr]).predict_proba(X[te])[:,1]
        a.append(roc_auc_score(y,p))
    return np.mean(a),np.std(a,ddof=1)
rows=[dict(model="confounds only (LR)",AUC=cvauc(Cv)[0],SD=cvauc(Cv)[1])]
Z=(Cv-Cv.mean(0))/Cv.std(0); base=sm.Logit(y,sm.add_constant(Z)).fit(disp=0)
for m in ["E2_soft_vote","E4_stacking","E5_topk_soft","E6_CA_weighted","B1_best_single","B3_early_fusion"]:
    s=avg[m].values; a,sd=cvauc(np.column_stack([Cv,s]))
    e=(s-s.mean())/s.std(); full=sm.Logit(y,sm.add_constant(np.column_stack([Z,e]))).fit(disp=0); lr=2*(full.llf-base.llf)
    rows.append(dict(model=f"confounds + {m}",AUC=a,SD=sd,LRT=lr,p_LRT=stats.chi2.sf(lr,1),OR_per_SD=np.exp(full.params[-1])))
    rows.append(dict(model=f"{m} alone (LR recal.)",AUC=cvauc(s[:,None])[0],SD=cvauc(s[:,None])[1]))
T=pd.DataFrame(rows); T.round(4).to_csv("results/table_exploratory_increment.csv",index=False); print(T.round(3).to_string(index=False))
