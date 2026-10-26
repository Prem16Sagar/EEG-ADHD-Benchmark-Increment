"""A3 RQ1 reporting: participant-bootstrap CIs (conditional on fitted models), DeLong (descriptive), Holm for family S."""
import json, glob, numpy as np, pandas as pd, sys
from sklearn.metrics import roc_auc_score, log_loss
from statsmodels.stats.multitest import multipletests
sys.path.insert(0, "src"); from analyse import delong
rng = np.random.default_rng(2026)
def boot(df, nb=2000):
    y = df.label.values; i1, i0 = np.where(y == 1)[0], np.where(y == 0)[0]; out = []
    for _ in range(nb):
        i = np.r_[rng.choice(i1, len(i1)), rng.choice(i0, len(i0))]
        b, f = df.score_base.values[i], df.score_full.values[i]
        out.append((roc_auc_score(y[i], f) - roc_auc_score(y[i], b), log_loss(y[i], b) - log_loss(y[i], f)))
    o = np.array(out); return np.percentile(o[:, 0], [2.5, 97.5]), np.percentile(o[:, 1], [2.5, 97.5])
rows = []
for f in sorted(glob.glob("results/A3_rq1*_oof.csv")):
    df = pd.read_csv(f); tag = f.split("/")[-1].replace("_oof.csv", "")
    ci_auc, ci_ll = boot(df); a, b, z, p_dl = delong(df.label.values, df.score_full.values, df.score_base.values)
    rows.append(dict(run=tag, n=len(df), auc_base=b, auc_full=a, dAUC=a - b, dAUC_CI_lo=ci_auc[0], dAUC_CI_hi=ci_auc[1],
                     dlogloss=log_loss(df.label, df.score_base) - log_loss(df.label, df.score_full),
                     dlogloss_CI_lo=ci_ll[0], dlogloss_CI_hi=ci_ll[1], delong_p_two_sided_descriptive=p_dl))
T = pd.DataFrame(rows)
P = {}
for f in glob.glob("results/A3_rq1*_summary.json"):
    j = json.load(open(f)); P[f.split("/")[-1].replace("_summary.json", "")] = j.get("p_perm_one_sided")
T["p_perm_one_sided"] = T.run.map(P)
fam = T.run.isin(["A3_rq1sec_acq_R1", "A3_rq1sec_nobeh_R1"]) & T.p_perm_one_sided.notna()
if fam.sum() == 2: T.loc[fam, "p_holm_family_S"] = multipletests(T.loc[fam, "p_perm_one_sided"], method="holm")[1]
T.round(4).to_csv("results/A3_rq1_table.csv", index=False); print(T.round(3).to_string(index=False))
