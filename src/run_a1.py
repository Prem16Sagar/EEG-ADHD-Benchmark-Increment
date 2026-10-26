"""Addendum A1 (configs/protocol_addendum_A1.yaml): nested stacking of the confound benchmark with
neural members, and a non-temporal-channel variant. Same folds, seeds and members as run_ensembles.py."""
import sys, time, warnings, numpy as np, pandas as pd
from joblib import Parallel, delayed
from scipy.stats import spearmanr
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from pyriemann.tangentspace import TangentSpace
sys.path.insert(0, "src")
from run_ensembles import load, learners, CONF, CH, proba
warnings.filterwarnings("ignore")
TEMPORAL = ["T7", "T8", "F7", "F8", "P7", "P8"]

class CovTangentSub(BaseEstimator, TransformerMixin):
    def __init__(self, idx=None): self.idx = idx
    def _m(self, X):
        C = np.asarray(X).reshape(-1, 19, 19)
        return C if self.idx is None else C[:, self.idx][:, :, self.idx]
    def fit(self, X, y=None): self.ts_ = TangentSpace(metric="riemann").fit(self._m(X)); return self
    def transform(self, X): return self.ts_.transform(self._m(X))

def pipe(family, est, idx):
    pre = [CovTangentSub(idx)] if family == "F5_riemann" else [SimpleImputer(strategy="median")]
    return make_pipeline(*pre, StandardScaler(), est)

def members(D, fam, tr, te, ytr, L, inner, idx):
    names = [(f, l) for f in fam for l in L]
    io = np.zeros((len(tr), len(names))); tp = np.zeros((len(te), len(names)))
    for j, (f, l) in enumerate(names):
        X = D[fam[f]].values
        for a, b in inner.split(X[tr], ytr):
            io[b, j] = proba(pipe(f, clone(L[l]), idx).fit(X[tr][a], ytr[a]), X[tr][b])
        tp[:, j] = proba(pipe(f, clone(L[l]), idx).fit(X[tr], ytr), X[te])
    return io, tp

def ens(io, tp, ytr, Ctr):
    auc = np.array([roc_auc_score(ytr, io[:, j]) for j in range(io.shape[1])])
    mr = np.nan_to_num(np.array([[abs(spearmanr(io[:, j], Ctr[:, c]).statistic) for c in range(Ctr.shape[1])]
                                 for j in range(io.shape[1])])).max(1)
    w6 = np.maximum(0, auc-0.5)*(1-mr); w6 = w6/w6.sum() if w6.sum() > 0 else np.ones_like(w6)/len(w6)
    top = np.argsort(-auc)[:5]
    return dict(E2=(io.mean(1), tp.mean(1)), E5=(io[:, top].mean(1), tp[:, top].mean(1)), E6=(io @ w6, tp @ w6))

def run_fold(rep, k, tr, te, D, fam):
    y = D["label"].values; ytr = y[tr]; Ctr = D.loc[tr, CONF].values
    L = learners(rep); inner = StratifiedKFold(5, shuffle=True, random_state=1000*rep + k)
    # confound vote: inner OOF + test
    Xc = D[CONF].values; ci = np.zeros(len(tr)); ct = np.zeros(len(te))
    for l in L:
        for a, b in inner.split(Xc[tr], ytr):
            ci[b] += proba(make_pipeline(StandardScaler(), clone(L[l])).fit(Xc[tr][a], ytr[a]), Xc[tr][b])/len(L)
        ct += proba(make_pipeline(StandardScaler(), clone(L[l])).fit(Xc[tr], ytr), Xc[te])/len(L)
    out = {"B0_confound_vote": ct}
    meta = lambda Xi, Xt: LogisticRegression(C=1.0, max_iter=5000).fit(Xi, ytr).predict_proba(Xt)[:, 1]
    for tag, chs in [("", CH), ("_nontemporal", [c for c in CH if c not in TEMPORAL])]:
        famx = {f: ([c for c in cols if c.rsplit("_", 1)[-1] in chs] if f != "F5_riemann" else cols) for f, cols in fam.items()}
        idx = None if tag == "" else [CH.index(c) for c in chs]
        io, tp = members(D, famx, tr, te, ytr, L, inner, idx)
        E = ens(io, tp, ytr, Ctr)
        for e, (i_s, t_s) in E.items(): out[f"{e}{tag}"] = t_s
        out[f"S_all{tag}"] = meta(np.column_stack([ci, io]), np.column_stack([ct, tp]))
        out[f"S_E6{tag}"] = meta(np.column_stack([ci, E["E6"][0]]), np.column_stack([ct, E["E6"][1]]))
        out[f"S_E5{tag}"] = meta(np.column_stack([ci, E["E5"][0]]), np.column_stack([ct, E["E5"][1]]))
    return [dict(rep=rep, fold=k, subject=D.subject[i], label=int(y[i]), method=m, score=float(s[n]))
            for m, s in out.items() for n, i in enumerate(te)]

def main():
    D, fam, _ = load(); y = D["label"].values
    jobs = [(r, k, tr, te) for r in range(10) for k, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=r).split(D, y))]
    t0 = time.time()
    res = Parallel(n_jobs=2, verbose=5)(delayed(run_fold)(r, k, tr, te, D, fam) for r, k, tr, te in jobs)
    pd.DataFrame([x for rows in res for x in rows]).to_csv("results/a1_oof_scores.csv.gz", index=False)
    print("done in %.1f min" % ((time.time()-t0)/60))

if __name__ == "__main__":
    main()
