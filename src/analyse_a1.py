"""Addendum A1 analysis: nested stacking vs confound benchmark, EMG checks."""
import sys, numpy as np, pandas as pd, statsmodels.api as sm
from scipy import stats
from sklearn.metrics import roc_auc_score
from statsmodels.stats.multitest import multipletests
from patsy import dmatrix
sys.path.insert(0, "src")
from analyse import delong, boot_idx, calib_slope, CONF, PREV_CONF

S = pd.read_csv("results/a1_oof_scores.csv.gz")
C = pd.read_csv(PREV_CONF)[["subject"]+CONF].set_index("subject")
avg = S.groupby(["method", "subject"]).score.mean().unstack(0)
y = S.drop_duplicates("subject").set_index("subject").loc[avg.index, "label"].values
Cv = C.loc[avg.index]; boots = list(boot_idx(y))

rows = []
for m in avg.columns:
    per = S[S.method == m].groupby("rep").apply(lambda d: roc_auc_score(d.label, d.score))
    s = avg[m].values; bs = [roc_auc_score(y[i], s[i]) for i in boots]
    pred = s > 0.5
    rows.append(dict(method=m, AUC_mean=per.mean(), AUC_sd_reps=per.std(ddof=1), AUC_avgscore=roc_auc_score(y, s),
                     CI_lo=np.percentile(bs, 2.5), CI_hi=np.percentile(bs, 97.5),
                     sens=(pred & (y == 1)).sum()/(y == 1).sum(), spec=(~pred & (y == 0)).sum()/(y == 0).sum(),
                     brier=np.mean((s-y)**2), calib_slope=calib_slope(y, s),
                     rho_emg=stats.spearmanr(s, Cv.muscle_temporal_emg).statistic,
                     rho_duration=stats.spearmanr(s, Cv.duration_s).statistic))
R = pd.DataFrame(rows).set_index("method"); R.round(3).to_csv("results/a1_table_methods.csv")

tests = []
for name, a in [("A1_T1 S_all vs B0", "S_all"), ("A1_T2 S_E6 vs B0", "S_E6"), ("A1_T3 S_E5 vs B0", "S_E5")]:
    A, B, z, p = delong(y, avg[a].values, avg["B0_confound_vote"].values)
    d = [roc_auc_score(y[i], avg[a].values[i]) - roc_auc_score(y[i], avg["B0_confound_vote"].values[i]) for i in boots]
    tests.append(dict(test=name, auc_a=A, auc_b=B, delta=A-B, CI_lo=np.percentile(d, 2.5), CI_hi=np.percentile(d, 97.5), z=z, p=p))
T = pd.DataFrame(tests); T["p_holm"] = multipletests(T.p, method="holm")[1]; T.round(4).to_csv("results/a1_table_tests.csv", index=False)

# EMG checks
spl = np.column_stack([np.asarray(dmatrix(f"cr(x, df=3) - 1", {"x": Cv[c].values})) for c in CONF])
base = sm.Logit(y, sm.add_constant(spl)).fit(disp=0, method="bfgs", maxiter=2000)
emg_rows = []
for m in ["E2", "E5", "E6", "E2_nontemporal", "E5_nontemporal", "E6_nontemporal"]:
    e = (avg[m]-avg[m].mean())/avg[m].std()
    full = sm.Logit(y, sm.add_constant(np.column_stack([spl, e]))).fit(disp=0, method="bfgs", maxiter=2000)
    lr = 2*(full.llf-base.llf)
    terc = pd.qcut(Cv.muscle_temporal_emg, 3, labels=False).values
    strat = [roc_auc_score(y[terc == t], avg[m].values[terc == t]) for t in range(3)]
    emg_rows.append(dict(score=m, LRT_over_spline_confounds=lr, p=stats.chi2.sf(lr, 1),
                         AUC_EMG_tertile1=strat[0], AUC_EMG_tertile2=strat[1], AUC_EMG_tertile3=strat[2],
                         rho_emg=stats.spearmanr(avg[m], Cv.muscle_temporal_emg).statistic))
terc = pd.qcut(Cv.muscle_temporal_emg, 3, labels=False).values
emg_rows.append(dict(score="EMG proxy alone", AUC_EMG_tertile1=roc_auc_score(y, Cv.muscle_temporal_emg)))
E = pd.DataFrame(emg_rows); E.round(4).to_csv("results/a1_table_emg_checks.csv", index=False)
print("n per EMG tertile:", np.bincount(terc), "ADHD per tertile:", [int(y[terc == t].sum()) for t in range(3)])

pd.set_option("display.width", 220)
print(R[["AUC_mean","AUC_sd_reps","AUC_avgscore","CI_lo","CI_hi","sens","spec","brier","calib_slope","rho_emg","rho_duration"]].round(3))
print(T.round(4).to_string(index=False)); print(E.round(3).to_string(index=False))
