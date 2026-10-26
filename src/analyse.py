"""Step 4: metrics, uncertainty, confound dependence and confirmatory tests (protocol.yaml)."""
import numpy as np, pandas as pd, statsmodels.api as sm
from scipy import stats
from sklearn.metrics import roc_auc_score
from statsmodels.stats.multitest import multipletests

CONF = ["duration_s","muscle_temporal_emg","line_ratio","drift_ratio","transient_frac","blink_frontal_kurt"]
import sys as _s, os as _o; _s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__))); from paths import PREV_CONF
rng = np.random.default_rng(0); NB = 2000

def delong(y, a, b):
    """Paired DeLong test for two correlated ROC AUCs. Returns (aucA, aucB, z, p)."""
    y = np.asarray(y); pos, neg = y == 1, y == 0
    def comps(s):
        X, Y = s[pos], s[neg]
        psi = (X[:, None] > Y[None, :]) + 0.5*(X[:, None] == Y[None, :])
        return psi.mean(), psi.mean(1), psi.mean(0)
    A, va10, va01 = comps(np.asarray(a)); B, vb10, vb01 = comps(np.asarray(b))
    s10 = np.cov(np.vstack([va10, vb10])); s01 = np.cov(np.vstack([va01, vb01]))
    S = s10/pos.sum() + s01/neg.sum()
    var = S[0, 0] + S[1, 1] - 2*S[0, 1]
    z = (A-B)/np.sqrt(var) if var > 0 else 0.0
    return A, B, z, 2*stats.norm.sf(abs(z))

def boot_idx(y):
    i1, i0 = np.where(y == 1)[0], np.where(y == 0)[0]
    for _ in range(NB):
        yield np.r_[rng.choice(i1, len(i1)), rng.choice(i0, len(i0))]   # stratified participant bootstrap

def calib_slope(y, p):
    p = np.clip(p, 1e-4, 1-1e-4); x = np.log(p/(1-p))
    return sm.Logit(y, sm.add_constant(x)).fit(disp=0).params[1]

def main():
    S = pd.read_csv("results/oof_scores.csv.gz")
    C = pd.read_csv(PREV_CONF)[["subject"]+CONF].set_index("subject")
    avg = S.groupby(["method","subject"]).score.mean().unstack(0)
    y = S.drop_duplicates("subject").set_index("subject").loc[avg.index, "label"].values
    Cv = C.loc[avg.index]
    boots = list(boot_idx(y))
    rows = []
    for meth in avg.columns:
        per = S[S.method == meth].groupby("rep").apply(lambda d: roc_auc_score(d.label, d.score))
        prd = S[S.method == meth].assign(pred=lambda d: d.score > 0.5)
        ba = prd.groupby("rep").apply(lambda d: pd.Series(dict(
            sens=(d.pred & (d.label == 1)).sum()/(d.label == 1).sum(),
            spec=(~d.pred & (d.label == 0)).sum()/(d.label == 0).sum()))).mean()
        s = avg[meth].values
        bs = [roc_auc_score(y[i], s[i]) for i in boots]
        rho = {c: stats.spearmanr(s, Cv[c]).statistic for c in CONF}
        rows.append(dict(method=meth, AUC_mean=per.mean(), AUC_sd_reps=per.std(ddof=1),
                         AUC_avgscore=roc_auc_score(y, s), CI_lo=np.percentile(bs, 2.5), CI_hi=np.percentile(bs, 97.5),
                         bal_acc=(ba.sens+ba.spec)/2, sens=ba.sens, spec=ba.spec,
                         brier=np.mean((s-y)**2), calib_slope=calib_slope(y, s),
                         max_abs_rho_conf=max(abs(v) for v in rho.values()),
                         rho_duration=rho["duration_s"], rho_line=rho["line_ratio"], rho_emg=rho["muscle_temporal_emg"]))
    R = pd.DataFrame(rows).set_index("method")
    R.round(3).to_csv("results/table_all_methods.csv")

    # Confirmatory tests on repeat-averaged scores
    tests = []
    for name, a, b in [("T1 E2 vs B1", "E2_soft_vote", "B1_best_single"),
                       ("T2 E2 vs B0", "E2_soft_vote", "B0_confound_vote"),
                       ("T3 E6 vs B0", "E6_CA_weighted", "B0_confound_vote")]:
        A, B, z, p = delong(y, avg[a].values, avg[b].values)
        d = [roc_auc_score(y[i], avg[a].values[i]) - roc_auc_score(y[i], avg[b].values[i]) for i in boots]
        tests.append(dict(test=name, auc_a=A, auc_b=B, delta=A-B, delta_CI_lo=np.percentile(d, 2.5),
                          delta_CI_hi=np.percentile(d, 97.5), stat=z, p=p))
    Z = (Cv - Cv.mean())/Cv.std()
    base = sm.Logit(y, sm.add_constant(Z.values)).fit(disp=0)
    e6 = (avg["E6_CA_weighted"] - avg["E6_CA_weighted"].mean())/avg["E6_CA_weighted"].std()
    full = sm.Logit(y, sm.add_constant(np.column_stack([Z.values, e6.values]))).fit(disp=0)
    lr = 2*(full.llf - base.llf)
    tests.append(dict(test="T4 E6 increment over confounds (LRT)", delta=full.params[-1], stat=lr, p=stats.chi2.sf(lr, 1)))
    T = pd.DataFrame(tests)
    T["p_holm"] = multipletests(T.p, method="holm")[1]
    T.round(4).to_csv("results/table_confirmatory_tests.csv", index=False)

    show = ["B0_confound_vote","B4_confound_lr","B4_confound_lgbm","B1_best_single","B3_early_fusion",
            "B2_F1_spectral_lr","B2_F2_aperiodic_lr","B2_F3_connectivity_lr","B2_F4_complexity_lr","B2_F5_riemann_lr",
            "E1_hard_vote","E2_soft_vote","E3_weighted_soft","E4_stacking","E5_topk_soft","E6_CA_weighted","E7_CA_exclude","E8_channel_vote"]
    pd.set_option("display.width", 200)
    print(R.loc[show, ["AUC_mean","AUC_sd_reps","CI_lo","CI_hi","bal_acc","sens","spec","brier","calib_slope","max_abs_rho_conf","rho_duration"]].round(3))
    print(); print(T.round(4).to_string(index=False))

if __name__ == "__main__":
    main()
