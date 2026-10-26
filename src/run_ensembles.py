"""Step 3: nested, participant-wise evaluation of members, ensembles and baselines
(configs/protocol.yaml). Writes out-of-fold scores for every method, repeat and
participant, plus per-fold inner-loop member statistics.
"""
import sys, time, json, warnings, numpy as np, pandas as pd
from joblib import Parallel, delayed
from scipy.stats import spearmanr
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from lightgbm import LGBMClassifier
from pyriemann.tangentspace import TangentSpace
warnings.filterwarnings("ignore")

CH = ["Fp1","Fp2","F3","F4","C3","C4","P3","P4","O1","O2","F7","F8","T7","T8","P7","P8","Fz","Cz","Pz"]
BANDS = ["delta","theta","alpha","beta"]
CONF = ["duration_s","muscle_temporal_emg","line_ratio","drift_ratio","transient_frac","blink_frontal_kurt"]
import sys as _s, os as _o; _s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__))); from paths import PREV_CONF
N_REPEATS = int(sys.argv[1]) if len(sys.argv) > 1 else 10

class CovTangent(BaseEstimator, TransformerMixin):
    """Flattened 19x19 covariances -> Riemannian tangent space (reference fit on training data)."""
    def fit(self, X, y=None):
        self.ts_ = TangentSpace(metric="riemann").fit(np.asarray(X).reshape(-1, 19, 19)); return self
    def transform(self, X):
        return self.ts_.transform(np.asarray(X).reshape(-1, 19, 19))

def learners(seed):
    return {
        "lr":   LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000),
        "lda":  LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
        "svm":  SVC(C=1.0, gamma="scale", class_weight="balanced", probability=True, random_state=seed),
        "rf":   RandomForestClassifier(n_estimators=300, min_samples_leaf=3, class_weight="balanced", random_state=seed, n_jobs=1),
        "lgbm": LGBMClassifier(n_estimators=200, num_leaves=15, learning_rate=0.05, subsample=0.8, subsample_freq=1,
                               colsample_bytree=0.8, class_weight="balanced", random_state=seed, n_jobs=1, verbose=-1),
        "knn":  KNeighborsClassifier(n_neighbors=15, weights="distance"),
    }

def load():
    D = pd.read_csv("derived/features_nasrabadi.csv")
    C = pd.read_csv(PREV_CONF)[["subject"] + CONF]
    D = D.merge(C, on="subject").sort_values("subject").reset_index(drop=True)
    fam = {
        "F1_spectral":     [f"rel_{b}_mean_{c}" for b in BANDS for c in CH],
        "F2_aperiodic":    [f"{p}_{c}" for p in ["ap_exp","ap_off","alpha_cf","alpha_pw"] for c in CH],
        "F3_connectivity": [f"wpli_{b}_{c}" for b in BANDS for c in CH],
        "F4_complexity":   [f"{p}_{c}" for p in ["perm","higuchi"] for c in CH],
        "F5_riemann":      [f"cov_{i}_{j}" for i in range(19) for j in range(19)],
    }
    chan = {c: [f"rel_{b}_mean_{c}" for b in BANDS] + [f"{p}_{c}" for p in ["ap_exp","ap_off","alpha_cf","alpha_pw"]]
               + [f"wpli_{b}_{c}" for b in BANDS] + [f"perm_{c}", f"higuchi_{c}"] for c in CH}
    return D, fam, chan

def member_pipe(family, est):
    pre = [CovTangent()] if family == "F5_riemann" else [SimpleImputer(strategy="median")]
    return make_pipeline(*pre, StandardScaler(), est)

def proba(m, X): return m.predict_proba(X)[:, 1]

def run_fold(rep, k, tr, te, D, fam, chan):
    seed = rep
    y = D["label"].values; ytr = y[tr]
    Ctr = D.loc[tr, CONF].values
    L = learners(seed)
    names = [(f, l) for f in fam for l in L]
    inner = StratifiedKFold(5, shuffle=True, random_state=1000*rep + k)
    inner_oof = np.zeros((len(tr), len(names))); test_p = np.zeros((len(te), len(names)))
    for j, (f, l) in enumerate(names):
        X = D[fam[f]].values
        for itr, ite in inner.split(X[tr], ytr):
            m = member_pipe(f, clone(L[l])).fit(X[tr][itr], ytr[itr])
            inner_oof[ite, j] = proba(m, X[tr][ite])
        test_p[:, j] = proba(member_pipe(f, clone(L[l])).fit(X[tr], ytr), X[te])
    auc_in = np.array([roc_auc_score(ytr, inner_oof[:, j]) for j in range(len(names))])
    rho = np.array([[abs(spearmanr(inner_oof[:, j], Ctr[:, c]).statistic) for c in range(len(CONF))] for j in range(len(names))])
    rho = np.nan_to_num(rho); maxrho = rho.max(1)
    out = {}
    def wsoft(w):
        w = np.asarray(w, float); return test_p.mean(1) if w.sum() <= 0 else (test_p*w).sum(1)/w.sum()
    out["E1_hard_vote"] = (test_p > 0.5).mean(1)
    out["E2_soft_vote"] = test_p.mean(1)
    out["E3_weighted_soft"] = wsoft(np.maximum(0, auc_in-0.5))
    stk = LogisticRegression(C=1.0, max_iter=5000).fit(inner_oof, ytr)
    out["E4_stacking"] = stk.predict_proba(test_p)[:, 1]
    top = np.argsort(-auc_in)[:5]; out["E5_topk_soft"] = test_p[:, top].mean(1)
    out["E6_CA_weighted"] = wsoft(np.maximum(0, auc_in-0.5)*(1-maxrho))
    keep = np.where(maxrho <= 0.30)[0]
    if len(keep) == 0: keep = np.argsort(maxrho)[:5]
    out["E7_CA_exclude"] = test_p[:, keep].mean(1)
    votes = []
    for c in CH:
        Xc = D[chan[c]].values
        m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), clone(L["lr"])).fit(Xc[tr], ytr)
        votes.append(proba(m, Xc[te]) > 0.5)
    out["E8_channel_vote"] = np.mean(votes, 0)
    best = int(np.argmax(auc_in)); out["B1_best_single"] = test_p[:, best]
    Xc = D[CONF].values
    cp = []
    for l in L:
        p = proba(make_pipeline(StandardScaler(), clone(L[l])).fit(Xc[tr], ytr), Xc[te]); cp.append(p)
        if l in ("lr", "lgbm"): out[f"B4_confound_{l}"] = p
    out["B0_confound_vote"] = np.mean(cp, 0)
    for f in fam: out[f"B2_{f}_lr"] = test_p[:, names.index((f, "lr"))]
    # B3 early fusion: F1-F4 concatenated + F5 tangent features
    from sklearn.compose import ColumnTransformer
    cols14 = sum([fam[f] for f in ["F1_spectral","F2_aperiodic","F3_connectivity","F4_complexity"]], [])
    ct = ColumnTransformer([("a", SimpleImputer(strategy="median"), cols14), ("r", CovTangent(), fam["F5_riemann"])])
    m = make_pipeline(ct, StandardScaler(), clone(L["lr"])).fit(D.loc[tr], ytr)
    out["B3_early_fusion"] = proba(m, D.loc[te])
    for j, (f, l) in enumerate(names): out[f"M_{f}__{l}"] = test_p[:, j]
    stats = pd.DataFrame(dict(rep=rep, fold=k, member=[f"{f}__{l}" for f, l in names], inner_auc=auc_in, max_rho=maxrho,
                              **{f"rho_{c}": rho[:, i] for i, c in enumerate(CONF)},
                              w_E3=np.maximum(0, auc_in-0.5), w_E6=np.maximum(0, auc_in-0.5)*(1-maxrho),
                              in_E7=np.isin(np.arange(len(names)), keep), in_E5=np.isin(np.arange(len(names)), top),
                              is_B1=np.arange(len(names)) == best))
    rows = [dict(rep=rep, fold=k, subject=D.subject[i], label=int(y[i]), method=meth, score=float(s[n]))
            for meth, s in out.items() for n, i in enumerate(te)]
    return rows, stats

def main():
    D, fam, chan = load()
    y = D["label"].values
    jobs = []
    for rep in range(N_REPEATS):
        for k, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=rep).split(D, y)):
            jobs.append((rep, k, tr, te))
    t0 = time.time()
    res = Parallel(n_jobs=2, verbose=5)(delayed(run_fold)(r, k, tr, te, D, fam, chan) for r, k, tr, te in jobs)
    scores = pd.DataFrame([row for rows, _ in res for row in rows])
    stats = pd.concat([s for _, s in res])
    scores.to_csv("results/oof_scores.csv.gz", index=False)
    stats.to_csv("results/inner_member_stats.csv", index=False)
    json.dump(dict(n_repeats=N_REPEATS, minutes=round((time.time()-t0)/60, 1)), open("results/run_info.json", "w"))
    print("done in %.1f min" % ((time.time()-t0)/60))

if __name__ == "__main__":
    main()
