"""Metrics and confirmatory tests for BALLADEER (addendum A2), same definitions as analyse.py / analyse_a1.py.
Usage: python src/analyse_cohort.py primary|sensitivity"""
import sys, numpy as np, pandas as pd, statsmodels.api as sm
from scipy import stats
from sklearn.metrics import roc_auc_score
from statsmodels.stats.multitest import multipletests
sys.path.insert(0, "src")
V = sys.argv[1]; sys.argv = sys.argv[:1]
from analyse import delong, boot_idx, calib_slope
from run_cohort import CONF, DEMO, ACQ

v = V
S = pd.read_csv(f"results/balladeer_{v}_oof_scores.csv.gz")
F = pd.read_csv("derived/features_balladeer.csv").set_index("subject")
avg = S.groupby(["method", "subject"]).score.mean().unstack(0)
y = S.drop_duplicates("subject").set_index("subject").loc[avg.index, "label"].values
Cv = F.loc[avg.index, CONF]; boots = list(boot_idx(y))
rows = []
for m in avg.columns:
    per = S[S.method == m].groupby("rep").apply(lambda d: roc_auc_score(d.label, d.score))
    s = avg[m].values; bs = [roc_auc_score(y[i], s[i]) for i in boots]; pred = s > 0.5
    rho = {c: stats.spearmanr(s, Cv[c]).statistic for c in CONF}
    rows.append(dict(method=m, AUC_mean=per.mean(), AUC_sd_reps=per.std(ddof=1), AUC_avgscore=roc_auc_score(y, s),
                     CI_lo=np.percentile(bs, 2.5), CI_hi=np.percentile(bs, 97.5),
                     sens=(pred & (y == 1)).sum()/(y == 1).sum(), spec=(~pred & (y == 0)).sum()/(y == 0).sum(),
                     brier=np.mean((s-y)**2), calib_slope=calib_slope(y, s), max_abs_rho_conf=max(abs(x) for x in rho.values()),
                     **{f"rho_{c}": rho[c] for c in ["age", "sex_male", "muscle_temporal_emg", "motion_acc_sd", "line_ratio"]}))
T = pd.DataFrame(rows).set_index("method"); T.round(3).to_csv(f"results/balladeer_{v}_table_methods.csv")

def dl(a, b):
    A_, B_, z, p = delong(y, avg[a].values, avg[b].values)
    d = [roc_auc_score(y[i], avg[a].values[i]) - roc_auc_score(y[i], avg[b].values[i]) for i in boots]
    return dict(auc_a=A_, auc_b=B_, delta=A_-B_, CI_lo=np.percentile(d, 2.5), CI_hi=np.percentile(d, 97.5), z=z, p=p)
fam1 = [dict(test="T1 E2 vs B1", **dl("E2_soft_vote", "B1_best_single")),
        dict(test="T2 E2 vs B0", **dl("E2_soft_vote", "B0_confound_vote")),
        dict(test="T3 E6 vs B0", **dl("E6_CA_weighted", "B0_confound_vote"))]
Z = (Cv-Cv.mean())/Cv.std(); base = sm.Logit(y, sm.add_constant(Z.values)).fit(disp=0, method="bfgs", maxiter=3000)
e6 = (avg["E6_CA_weighted"]-avg["E6_CA_weighted"].mean())/avg["E6_CA_weighted"].std()
full = sm.Logit(y, sm.add_constant(np.column_stack([Z.values, e6.values]))).fit(disp=0, method="bfgs", maxiter=3000)
lr = 2*(full.llf-base.llf); fam1.append(dict(test="T4 E6 increment over confounds (LRT)", delta=full.params[-1], z=lr, p=stats.chi2.sf(lr, 1)))
fam2 = [dict(test=f"A1 {s} vs B0", **dl(s, "B0_confound_vote")) for s in ["S_all", "S_E6", "S_E5"]]
out = []
for fam in (fam1, fam2):
    t = pd.DataFrame(fam); t["p_holm"] = multipletests(t.p, method="holm")[1]; out.append(t)
out = pd.concat(out); out.round(4).to_csv(f"results/balladeer_{v}_table_tests.csv", index=False)
pd.set_option("display.width", 250)
show = [m for m in ["B0_confound_vote", "B0_acq_vote", "B0_demo_vote", "B4_confound_lr", "B1_best_single", "B3_early_fusion",
        "B2_F1_spectral_lr", "B2_F2_aperiodic_lr", "B2_F3_connectivity_lr", "B2_F4_complexity_lr", "B2_F5_riemann_lr",
        "E1_hard_vote", "E2_soft_vote", "E3_weighted_soft", "E4_stacking", "E5_topk_soft", "E6_CA_weighted", "E7_CA_exclude",
        "E8_channel_vote", "S_all", "S_E6", "S_E5"] if m in T.index]
print(T.loc[show, ["AUC_mean", "AUC_sd_reps", "CI_lo", "CI_hi", "sens", "spec", "calib_slope", "rho_age", "rho_muscle_temporal_emg", "rho_motion_acc_sd"]].round(3))
print(out.round(4).to_string(index=False))
