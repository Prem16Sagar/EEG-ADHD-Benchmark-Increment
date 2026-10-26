"""Exploratory cross-cohort transfer (addendum A2): train on one cohort, test on the other.
Neural families only; alignment variants: none / per-cohort feature z-scoring (+ Riemannian re-centring for F5).
Alignment uses only unlabelled feature distributions of each cohort; target labels are used only for scoring."""
import sys, warnings, numpy as np, pandas as pd
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.linear_model import LogisticRegression
from pyriemann.utils.mean import mean_riemann
from pyriemann.utils.base import invsqrtm
sys.path.insert(0, "src")
import run_ensembles as R
warnings.filterwarnings("ignore")
rng = np.random.default_rng(0)

def cohorts():
    N, fam, _ = R.load()
    B = pd.read_csv("derived/features_balladeer.csv"); B = B[B.n_bad <= 4].reset_index(drop=True)
    return N, B, fam

def recentre(D, cols):
    C = D[cols].values.reshape(-1, 19, 19); W = invsqrtm(mean_riemann(C))
    D = D.copy(); D[cols] = np.einsum("ij,njk,kl->nil", W, C, W).reshape(len(D), -1); return D

def zscore(D, cols):
    D = D.copy(); X = D[cols].astype(float); D[cols] = (X - X.median()) / (X.std() + 1e-12); return D

def prepare(N, B, fam, align):
    N, B = N.copy(), B.copy()
    if align == "cohort_norm":
        for D in (N, B):
            pass
        N = recentre(N, fam["F5_riemann"]); B = recentre(B, fam["F5_riemann"])
        other = sum([fam[f] for f in fam if f != "F5_riemann"], [])
        N = zscore(N, other); B = zscore(B, other)
    return N, B

def transfer(src, tgt, fam, seed=0):
    L = R.learners(seed); out = {}; ys, yt = src.label.values, tgt.label.values
    probs = []
    for f in fam:
        for l in L:
            m = R.member_pipe(f, clone(L[l])).fit(src[fam[f]].values, ys)
            p = R.proba(m, tgt[fam[f]].values); out[f"M_{f}__{l}"] = p; probs.append(p)
    out["E2_soft_vote"] = np.mean(probs, 0)
    return {k: roc_auc_score(yt, v) for k, v in out.items()}, out

def boot_ci(y, s, nb=2000):
    i1, i0 = np.where(y == 1)[0], np.where(y == 0)[0]
    b = [roc_auc_score(y[i], s[i]) for i in (np.r_[rng.choice(i1, len(i1)), rng.choice(i0, len(i0))] for _ in range(nb))]
    return np.percentile(b, [2.5, 97.5])

def identifiability(N, B, fam):
    """How well can neural features tell the two cohorts apart? (5-fold CV LR, per family)"""
    D = pd.concat([N.assign(cohort=0), B.assign(cohort=1)], ignore_index=True); res = {}
    for f in fam:
        p = cross_val_predict(R.member_pipe(f, LogisticRegression(max_iter=5000)), D[fam[f]].values, D.cohort.values,
                              cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")[:, 1]
        res[f] = roc_auc_score(D.cohort, p)
    return res

def main():
    N0, B0, fam = cohorts(); rows = []
    for align in ["none", "cohort_norm"]:
        N, B = prepare(N0, B0, fam, align)
        for name, src, tgt in [("Nasrabadi->BALLADEER", N, B), ("BALLADEER->Nasrabadi", B, N)]:
            aucs, scores = transfer(src, tgt, fam)
            for k in ["E2_soft_vote"] + [f"M_{f}__lr" for f in fam] + ["M_F5_riemann__svm"]:
                lo, hi = boot_ci(tgt.label.values, scores[k])
                rows.append(dict(alignment=align, direction=name, model=k, AUC=aucs[k], CI_lo=lo, CI_hi=hi))
        for f, a in identifiability(N, B, fam).items():
            rows.append(dict(alignment=align, direction="cohort identifiability", model=f, AUC=a))
    T = pd.DataFrame(rows); T.round(3).to_csv("results/cross_cohort_transfer.csv", index=False)
    pd.set_option("display.width", 200); print(T.round(3).to_string(index=False))

if __name__ == "__main__":
    main()
