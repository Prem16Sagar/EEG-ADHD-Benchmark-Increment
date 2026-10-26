"""A3 section 7: RQ2a within-session cross-level reliability (descriptive) and RQ2b level transfer (exploratory).
BALLADEER CGX Slackline L1 / L6 / L11. A2 exclusion (>4 bad channels) applied per level recording.

Implementation choices (not fixed in A3 text; set here before any RQ2 result):
- Riemannian family enters the ICC as tangent-space coordinates at the label-free Riemannian mean of all
  recordings pooled across levels; the other families use their raw feature values.
- Model scores for ICC and transfer: the five family L2-LRs (C=1, as RQ1) and an EEG-only stack
  (L2-LR on inner-OOF logits of the five families). Outer 5-fold, seeds 0-9 (R=10); inner 5-fold, seed 1000*r+k.
- Transfer: train on level A of outer-training children, test on level B of held-out children; within-level
  reference = train and test on level B with the same splits. Label-permutation p: B=500, R=1, uncorrected.
"""
import sys, json, warnings, numpy as np, pandas as pd
from joblib import Parallel, delayed
from scipy.special import logit
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from pyriemann.utils.mean import mean_riemann
from pyriemann.tangentspace import TangentSpace
sys.path.insert(0, "src")
import rq1_A3 as M
warnings.filterwarnings("ignore")
LEVELS = [1, 6, 11]
rng = np.random.default_rng(2026)

def load_levels():
    F = {1: pd.read_csv("derived/features_balladeer.csv")}
    for L in (6, 11): F[L] = pd.read_csv(f"derived/features_balladeer_L{L}.csv")
    F = {L: d[d.n_bad <= 4].set_index("subject") for L, d in F.items()}
    lab = F[1][["label", "group", "diagnosed", "age", "sex_male"]]
    return F, lab

# ------------------------------------------------------------------ ICC (McGraw & Wong 1996 forms; Koo & Li 2016)
def icc(Y):
    """Y: n x k x p (subjects, levels, features). Returns ICC(A,1), ICC(C,1) per feature."""
    n, k, _ = Y.shape; gm = Y.mean((0, 1)); rm = Y.mean(1); cm = Y.mean(0)
    msr = k * ((rm - gm) ** 2).sum(0) / (n - 1)
    msc = n * ((cm - gm) ** 2).sum(0) / (k - 1)
    sse = ((Y - rm[:, None, :] - cm[None, :, :] + gm) ** 2).sum((0, 1)); mse = sse / ((n - 1) * (k - 1))
    a = (msr - mse) / (msr + (k - 1) * mse + k / n * (msc - mse)); c = (msr - mse) / (msr + (k - 1) * mse)
    return a, c

def reliability(F):
    subs = sorted(set.intersection(*[set(F[L].index) for L in LEVELS]))
    cov = M.FAM["F5_riemann"]
    allC = np.concatenate([F[L].loc[subs, cov].values.reshape(-1, 19, 19) for L in LEVELS])
    ts = TangentSpace(metric="riemann").fit(allC)                     # label-free pooled reference
    blocks = {}
    for f, cols in M.FAM.items():
        if f == "F5_riemann":
            blocks[f] = np.stack([ts.transform(F[L].loc[subs, cov].values.reshape(-1, 19, 19)) for L in LEVELS], 1)
        else:
            X = np.stack([F[L].loc[subs, cols].values.astype(float) for L in LEVELS], 1)
            keep = ~np.isnan(X).any((0, 1)); blocks[f] = X[:, :, keep]   # alpha_cf can be missing: complete features only
    rows = []
    for f, Y in blocks.items():
        a, c = icc(Y)
        bs = []
        for _ in range(2000):
            i = rng.integers(0, len(subs), len(subs)); ab, cb = icc(Y[i]); bs.append((np.nanmedian(ab), np.nanmedian(cb)))
        bs = np.array(bs)
        rows.append(dict(family=f, n_subjects=len(subs), n_features=Y.shape[2],
                         ICC_A1_median=np.nanmedian(a), ICC_A1_q25=np.nanpercentile(a, 25), ICC_A1_q75=np.nanpercentile(a, 75),
                         ICC_A1_median_CI_lo=np.percentile(bs[:, 0], 2.5), ICC_A1_median_CI_hi=np.percentile(bs[:, 0], 97.5),
                         ICC_C1_median=np.nanmedian(c), ICC_C1_q25=np.nanpercentile(c, 25), ICC_C1_q75=np.nanpercentile(c, 75),
                         ICC_C1_median_CI_lo=np.percentile(bs[:, 1], 2.5), ICC_C1_median_CI_hi=np.percentile(bs[:, 1], 97.5)))
    return pd.DataFrame(rows), subs

# ------------------------------------------------------------------ cross-level scoring
def fit_stack(Xs, y, tr, inner_seed):
    """Fit family LRs + EEG-only stack on training rows tr of the source-level matrices Xs (dict fam->array)."""
    inner = StratifiedKFold(5, shuffle=True, random_state=inner_seed)
    Z = np.zeros((len(tr), len(M.FAM))); models = {}
    for j, f in enumerate(M.FAM):
        X = Xs[f]
        for itr, ite in inner.split(X[tr], y[tr]):
            Z[ite, j] = logit(M.p1(M.fam_pipe(f).fit(X[tr][itr], y[tr][itr]), X[tr][ite]))
        models[f] = M.fam_pipe(f).fit(X[tr], y[tr])
    meta = M.lr().fit(Z, y[tr])
    return models, meta

def score(models, meta, Xt, te):
    Z = np.column_stack([logit(M.p1(models[f], Xt[f][te])) for f in M.FAM])
    out = {f"M_{f}": M.p1(models[f], Xt[f][te]) for f in M.FAM}; out["stack"] = M.p1(meta, Z); return out

def cross_level(F, lab, src, tgts, R, y_override=None):
    """Train on level src, score held-out children at each level in tgts. Children must have src and all tgts."""
    subs = sorted(set.intersection(*[set(F[L].index) for L in [src] + tgts]))
    y = lab.loc[subs, "label"].values if y_override is None else y_override
    Xs = {L: {f: F[L].loc[subs, cols].values for f, cols in M.FAM.items()} for L in set([src] + tgts)}
    S = {(L, m): np.zeros((R, len(subs))) for L in tgts for m in list(map(lambda f: f"M_{f}", M.FAM)) + ["stack"]}
    for r in range(R):
        for k, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=r).split(subs, y)):
            models, meta = fit_stack(Xs[src], y, tr, 1000 * r + k)
            for L in tgts:
                for m, s in score(models, meta, Xs[L], te).items(): S[(L, m)][r, te] = s
    return subs, y, {key: v.mean(0) for key, v in S.items()}

def perm_auc(F, lab, src, tgt, b):
    subs = sorted(set(F[src].index) & set(F[tgt].index))
    y = np.random.default_rng(20_000 + b).permutation(lab.loc[subs, "label"].values)
    _, _, S = cross_level(F, lab, src, [tgt], 1, y_override=y)
    return roc_auc_score(y, S[(tgt, "stack")])

def boot_ci(y, s, nb=2000):
    i1, i0 = np.where(y == 1)[0], np.where(y == 0)[0]
    b = [roc_auc_score(y[i], s[i]) for i in (np.r_[rng.choice(i1, len(i1)), rng.choice(i0, len(i0))] for _ in range(nb))]
    return np.percentile(b, [2.5, 97.5])

if __name__ == "__main__":
    sys.path.insert(0, "src"); import rq2_A3 as Q
    B = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    F, lab = Q.load_levels()
    rel, subs3 = Q.reliability(F); rel.round(4).to_csv("results/A3_rq2a_reliability.csv", index=False)
    print(rel.round(3).to_string(index=False), flush=True)
    # model-score ICC across levels (children with all three levels): train on L1, score L1/L6/L11 held-out
    _, y3, S3 = Q.cross_level(F, lab, 1, LEVELS, 10)
    ms = []
    for m in [f"M_{f}" for f in M.FAM] + ["stack"]:
        Y = np.stack([logit(np.clip(S3[(L, m)], 1e-6, 1 - 1e-6)) for L in LEVELS], 1)[:, :, None]
        a, c = Q.icc(Y); ms.append(dict(model=m, n=len(y3), ICC_A1=a[0], ICC_C1=c[0]))
    pd.DataFrame(ms).round(4).to_csv("results/A3_rq2a_modelscore_icc.csv", index=False); print(pd.DataFrame(ms).round(3), flush=True)
    # transfer and within-level reference
    rows, oof = [], []
    for src, tgts in [(1, [1, 6, 11]), (6, [6, 1]), (11, [11, 1])]:
        for tgt in tgts:
            subs, y, S = Q.cross_level(F, lab, src, [tgt], 10)
            for m in [f"M_{f}" for f in M.FAM] + ["stack"]:
                s = S[(tgt, m)]; lo, hi = Q.boot_ci(y, s)
                rows.append(dict(train=f"L{src}", test=f"L{tgt}", model=m, n=len(y), n_pos=int(y.sum()),
                                 AUC=roc_auc_score(y, s), CI_lo=lo, CI_hi=hi))
            oof.append(pd.DataFrame(dict(train=src, test=tgt, subject=subs, label=y, stack=S[(tgt, "stack")])))
    T = pd.DataFrame(rows)
    pvals = {}
    for src, tgt in [(1, 6), (1, 11), (6, 1), (11, 1)]:
        null = Parallel(n_jobs=2)(delayed(Q.perm_auc)(F, lab, src, tgt, b) for b in range(B))
        _, y1, S1 = Q.cross_level(F, lab, src, [tgt], 1)                 # observed with the same R=1 as the null
        obs = roc_auc_score(y1, S1[(tgt, "stack")])
        pvals[(f"L{src}", f"L{tgt}")] = (1 + np.sum(np.array(null) >= obs)) / (B + 1); print(src, tgt, pvals[(f"L{src}", f"L{tgt}")], flush=True)
    T["p_perm_stack_uncorrected"] = [pvals.get((a, b)) if m == "stack" else None for a, b, m in zip(T.train, T.test, T.model)]
    T.round(4).to_csv("results/A3_rq2b_transfer.csv", index=False)
    pd.concat(oof).to_csv("results/A3_rq2b_stack_oof.csv", index=False)
    pd.set_option("display.width", 200); print(T.round(3).to_string(index=False))
