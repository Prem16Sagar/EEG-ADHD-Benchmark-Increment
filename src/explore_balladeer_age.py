"""Exploratory (BALLADEER): do neural scores add beyond age/sex and all confounds? Age-sex matched subset AUC."""
import sys, numpy as np, pandas as pd, statsmodels.api as sm
from scipy import stats
from sklearn.metrics import roc_auc_score
sys.argv = sys.argv[:1]; sys.path.insert(0, "src")
from run_cohort import CONF, DEMO
S = pd.read_csv("results/balladeer_primary_oof_scores.csv.gz"); F = pd.read_csv("derived/features_balladeer.csv").set_index("subject")
avg = S.groupby(["method", "subject"]).score.mean().unstack(0); y = F.loc[avg.index, "label"].values
def lrt(cols, m):
    Z = F.loc[avg.index, cols]; Z = ((Z-Z.mean())/Z.std()).values
    b = sm.Logit(y, sm.add_constant(Z)).fit(disp=0, method="bfgs", maxiter=3000)
    e = avg[m].values; e = (e-e.mean())/e.std()
    f = sm.Logit(y, sm.add_constant(np.column_stack([Z, e]))).fit(disp=0, method="bfgs", maxiter=3000)
    l = 2*(f.llf-b.llf); return l, stats.chi2.sf(l, 1)
# age/sex matched subset: greedy 1:1 nearest age within 1 year, same sex
d = F.loc[avg.index, ["label", "age", "sex_male"]].copy(); ctrl = d[d.label == 0]; used = set(); pairs = []
for s, r in d[d.label == 1].sample(frac=1, random_state=0).iterrows():
    c = ctrl[(ctrl.sex_male == r.sex_male) & (~ctrl.index.isin(used)) & ((ctrl.age-r.age).abs() <= 1)]
    if len(c): m = (c.age-r.age).abs().idxmin(); used.add(m); pairs += [s, m]
sub = d.loc[pairs]; ys = sub.label.values
rows = []
for m in ["B0_demo_vote", "B0_acq_vote", "B0_confound_vote", "B2_F1_spectral_lr", "B1_best_single", "E2_soft_vote", "E5_topk_soft", "E6_CA_weighted"]:
    l1, p1 = lrt(DEMO, m); l2, p2 = lrt(CONF, m)
    rows.append(dict(score=m, AUC_all=roc_auc_score(y, avg[m]), LRT_over_age_sex=l1, p_age_sex=p1, LRT_over_all_conf=l2, p_all_conf=p2,
                     AUC_matched=roc_auc_score(ys, avg.loc[sub.index, m])))
T = pd.DataFrame(rows); T.round(4).to_csv("results/balladeer_exploratory_age_matched.csv", index=False)
print("matched pairs:", len(pairs)//2, "age ADHD/ctrl: %.2f/%.2f" % (sub[sub.label == 1].age.mean(), sub[sub.label == 0].age.mean()))
print(T.round(3).to_string(index=False))
