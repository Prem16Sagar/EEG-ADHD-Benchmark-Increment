"""A3 section 6 (secondary contrasts) and section 2 sensitivity models. Written after lock, before any RQ1 result.
Same pipeline as rq1_A3.fold, parametrised by the tabular (non-EEG) block, the EEG families and the LR estimator.

Usage: python src/rq1_A3_sec.py <variant> <R> <B>
  variants: acq        base = demo+behaviour+acquisition; full = base + EEG stack     (Holm family S)
            nobeh      base = demo;                     full = demo + EEG stack      (Holm family S)
            tunedC     primary contrast with inner-CV-tuned C in {0.01,0.1,1,10}     (sensitivity)
            spectral   primary contrast with spectral-only EEG block (POST-HOC)      (sensitivity)
"""
import sys, json, warnings, numpy as np, pandas as pd
from joblib import Parallel, delayed
from scipy.special import logit
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, "src")
import rq1_A3 as M
warnings.filterwarnings("ignore")
ACQ = ["duration_s", "muscle_temporal_emg", "line_ratio", "drift_ratio", "transient_frac", "blink_frontal_kurt", "motion_acc_sd"]

def config(variant):
    tab = {"acq": M.DEMO + M.BEH + ACQ, "nobeh": M.DEMO}.get(variant, M.DEMO + M.BEH)
    fams = {"F1_spectral": M.FAM["F1_spectral"]} if variant == "spectral" else M.FAM
    return tab, fams, variant == "tunedC"

def est(tuned, seed):
    if tuned:
        return LogisticRegressionCV(Cs=[0.01, 0.1, 1, 10], cv=StratifiedKFold(5, shuffle=True, random_state=seed),
                                    scoring="roc_auc", class_weight="balanced", max_iter=5000)
    return LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000)

def fold(D, y, tr, te, rep, k, variant):
    tab, fams, tuned = config(variant); s = 1000 * rep + k
    tab_pipe = lambda: make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(), est(tuned, s))
    fam_pipe = lambda f: make_pipeline(M.CovTangent() if f == "F5_riemann" else SimpleImputer(strategy="median"), StandardScaler(), est(tuned, s))
    Xd = D[tab].values
    pb = M.p1(tab_pipe().fit(Xd[tr], y[tr]), Xd[te])
    inner = StratifiedKFold(5, shuffle=True, random_state=s)
    Ztr = np.zeros((len(tr), len(fams))); Zte = np.zeros((len(te), len(fams)))
    for j, (f, cols) in enumerate(fams.items()):
        X = D[cols].values
        for itr, ite in inner.split(X[tr], y[tr]):
            Ztr[ite, j] = logit(M.p1(fam_pipe(f).fit(X[tr][itr], y[tr][itr]), X[tr][ite]))
        Zte[:, j] = logit(M.p1(fam_pipe(f).fit(X[tr], y[tr]), X[te]))
    pf = M.p1(tab_pipe().fit(np.column_stack([Xd[tr], Ztr]), y[tr]), np.column_stack([Xd[te], Zte]))
    return pb, pf

def run_once(D, y, R, variant, eeg_perm=None):
    if eeg_perm is not None:
        D = D.copy(); D[M.EEG_COLS] = D[M.EEG_COLS].values[eeg_perm]
    Sb = np.zeros((R, len(D))); Sf = np.zeros((R, len(D)))
    for r in range(R):
        for k, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=r).split(D, y)):
            Sb[r, te], Sf[r, te] = fold(D, y, tr, te, r, k, variant)
    sb, sf = Sb.mean(0), Sf.mean(0)
    return dict(auc_base=roc_auc_score(y, sb), auc_full=roc_auc_score(y, sf), dauc=roc_auc_score(y, sf) - roc_auc_score(y, sb),
                dlogloss=log_loss(y, sb) - log_loss(y, sf)), sb, sf

def one_perm(D, y, R, variant, b):
    return run_once(D, y, R, variant, M.strat_perm(D.stratum.values, np.random.default_rng(10_000 + b)))[0]["dauc"]

if __name__ == "__main__":
    sys.path.insert(0, "src"); import rq1_A3_sec as S
    variant, R, B = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    D = M.load("primary"); y = D.label.values
    obs, sb, sf = S.run_once(D, y, R, variant)
    pd.DataFrame(dict(subject=D.subject, label=y, score_base=sb, score_full=sf)).to_csv(f"results/A3_rq1sec_{variant}_R{R}_oof.csv", index=False)
    p = None
    if B > 0:
        null = Parallel(n_jobs=2, verbose=2)(delayed(S.one_perm)(D, y, R, variant, b) for b in range(B))
        pd.Series(null, name="dauc_perm").to_csv(f"results/A3_rq1sec_{variant}_R{R}_null.csv", index=False)
        p = (1 + np.sum(np.array(null) >= obs["dauc"])) / (B + 1)
    json.dump(dict(variant=variant, n=int(len(y)), R=R, B=B, p_perm_one_sided=p, **obs),
              open(f"results/A3_rq1sec_{variant}_R{R}_summary.json", "w"), indent=1)
    print(variant, obs, "p =", p)
