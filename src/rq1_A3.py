"""A3 RQ1: incremental value of EEG beyond age, sex and Slackline L1 behaviour (BALLADEER, CGX L1).

Usage:
  python src/rq1_A3.py timing                       # timing test, labels shuffled (no real-label result)
  python src/rq1_A3.py run <label> <R> <B> [tag]    # label in primary|S1|S2; only after A3 is locked
"""
import sys, time, json, warnings, numpy as np, pandas as pd
from joblib import Parallel, delayed
from scipy.special import logit
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from pyriemann.tangentspace import TangentSpace
warnings.filterwarnings("ignore")

CH = ["Fp1","Fp2","F3","F4","C3","C4","P3","P4","O1","O2","F7","F8","T7","T8","P7","P8","Fz","Cz","Pz"]
BANDS = ["delta","theta","alpha","beta"]
FAM = {"F1_spectral": [f"rel_{b}_mean_{c}" for b in BANDS for c in CH],
       "F2_aperiodic": [f"{p}_{c}" for p in ["ap_exp","ap_off","alpha_cf","alpha_pw"] for c in CH],
       "F3_connectivity": [f"wpli_{b}_{c}" for b in BANDS for c in CH],
       "F4_complexity": [f"{p}_{c}" for p in ["perm","higuchi"] for c in CH],
       "F5_riemann": [f"cov_{i}_{j}" for i in range(19) for j in range(19)]}
EEG_COLS = sum(FAM.values(), [])
DEMO = ["age", "sex_male"]
BEH = ["B1_omission_rate", "B2_commission_rate", "B3_mean_RT_hits", "B4_RT_cv_hits"]

class CovTangent(BaseEstimator, TransformerMixin):
    """Flattened 19x19 covariances -> Riemannian tangent space at the training-fold mean (as A2)."""
    def fit(self, X, y=None):
        self.ts_ = TangentSpace(metric="riemann").fit(np.asarray(X).reshape(-1, 19, 19)); return self
    def transform(self, X):
        return self.ts_.transform(np.asarray(X).reshape(-1, 19, 19))

def lr(): return LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000)
def fam_pipe(f): return make_pipeline(CovTangent() if f == "F5_riemann" else SimpleImputer(strategy="median"), StandardScaler(), lr())
def tab_pipe(): return make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(), lr())
def p1(m, X): return np.clip(m.predict_proba(X)[:, 1], 1e-6, 1 - 1e-6)

def load(label="primary"):
    D = pd.read_csv("derived/features_balladeer.csv"); D = D[D.n_bad <= 4].copy()     # A2 analysable set
    B = pd.read_csv("derived/behaviour_L1_A3.csv"); B = B[B.s2_ok]                    # A3 s1-s2
    D = D.merge(B[["subject"] + BEH], on="subject", how="inner")
    if label == "S1":
        D = D[(D.group == 1) | (D.diagnosed == "no")].copy(); D["label"] = (D.group == 1).astype(int)
    elif label == "S2":
        D["label"] = (D.group == 1).astype(int)
    D = D.sort_values("subject").reset_index(drop=True)
    D["stratum"] = ((D.age - 6) // 2).astype(int).astype(str) + "_" + D.sex_male.astype(int).astype(str)
    return D

def fold(D, y, tr, te, rep, k):
    """One outer fold: returns test-fold probabilities of M_base and M_full. All fitting on tr only."""
    Xd = D[DEMO + BEH].values
    base = tab_pipe().fit(Xd[tr], y[tr]); pb = p1(base, Xd[te])
    inner = StratifiedKFold(5, shuffle=True, random_state=1000 * rep + k)
    Ztr = np.zeros((len(tr), len(FAM))); Zte = np.zeros((len(te), len(FAM)))
    for j, (f, cols) in enumerate(FAM.items()):
        X = D[cols].values
        for itr, ite in inner.split(X[tr], y[tr]):                       # inner OOF scores for the meta-model
            Ztr[ite, j] = logit(p1(fam_pipe(f).fit(X[tr][itr], y[tr][itr]), X[tr][ite]))
        Zte[:, j] = logit(p1(fam_pipe(f).fit(X[tr], y[tr]), X[te]))       # refit on full outer-training set
    meta = tab_pipe().fit(np.column_stack([Xd[tr], Ztr]), y[tr])
    pf = p1(meta, np.column_stack([Xd[te], Zte]))
    return pb, pf

def run_once(D, y, R, eeg_perm=None):
    """dAUC on OOF scores averaged over R outer repeats (seeds 0..R-1). eeg_perm permutes EEG rows."""
    if eeg_perm is not None:
        D = D.copy(); D[EEG_COLS] = D[EEG_COLS].values[eeg_perm]
    Sb = np.zeros((R, len(D))); Sf = np.zeros((R, len(D))); per_fold = []
    for r in range(R):
        for k, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=r).split(D, y)):
            pb, pf = fold(D, y, tr, te, r, k); Sb[r, te] = pb; Sf[r, te] = pf
            per_fold.append(roc_auc_score(y[te], pf) - roc_auc_score(y[te], pb))
    sb, sf = Sb.mean(0), Sf.mean(0)
    return dict(auc_base=roc_auc_score(y, sb), auc_full=roc_auc_score(y, sf),
                dauc=roc_auc_score(y, sf) - roc_auc_score(y, sb), dauc_perfold_mean=float(np.mean(per_fold)),
                dlogloss=log_loss(y, sb) - log_loss(y, sf)), sb, sf

def strat_perm(strata, rng):
    idx = np.arange(len(strata)); out = idx.copy()
    for s in np.unique(strata):
        m = idx[strata == s]; out[m] = rng.permutation(m)
    return out

def one_perm(D, y, R, b):
    rng = np.random.default_rng(10_000 + b)
    return run_once(D, y, R, strat_perm(D.stratum.values, rng))[0]["dauc"]

if __name__ == "__main__":
    sys.path.insert(0, "src"); import rq1_A3 as M      # workers unpickle functions from the module, not __main__
    load, run_once, one_perm = M.load, M.run_once, M.one_perm
    mode = sys.argv[1]
    if mode == "timing":
        D = load("primary"); y = np.random.default_rng(999).permutation(D.label.values)   # shuffled labels only
        print("n =", len(D)); t = time.time(); run_once(D, y, 1); t1 = time.time() - t
        t = time.time(); Parallel(n_jobs=2)(delayed(one_perm)(D, y, 1, b) for b in range(4)); t4 = time.time() - t
        est = {f"R{R}_B{B}_hours": round(t4 / 4 * R * B / 3600, 2) for R in (1, 3) for B in (500, 1000)}
        print(json.dumps(dict(sec_one_run_R1=round(t1, 1), sec_per_perm_R1_parallel=round(t4 / 4, 1), **est)))
    elif mode == "run":
        label, R, B = sys.argv[2], int(sys.argv[3]), int(sys.argv[4]); tag = sys.argv[5] if len(sys.argv) > 5 else label
        D = load(label); y = D.label.values
        obs, sb, sf = run_once(D, y, R)
        pd.DataFrame(dict(subject=D.subject, label=y, score_base=sb, score_full=sf)).to_csv(f"results/A3_rq1_{tag}_oof.csv", index=False)
        null = Parallel(n_jobs=2, verbose=2)(delayed(one_perm)(D, y, R, b) for b in range(B))
        pd.Series(null, name="dauc_perm").to_csv(f"results/A3_rq1_{tag}_null.csv", index=False)
        p = (1 + np.sum(np.array(null) >= obs["dauc"])) / (B + 1) if B > 0 else None   # B = 0: point estimate only
        json.dump(dict(label=label, n=int(len(y)), n_pos=int(y.sum()), R=R, B=B, p_perm_one_sided=p, **obs),
                  open(f"results/A3_rq1_{tag}_summary.json", "w"), indent=1)
        print(obs, "p =", p)
