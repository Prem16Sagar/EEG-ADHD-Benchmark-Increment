"""POST-HOC (not in locked A3; requested by the user 2026-10-21 after seeing RQ2b):
age/sex-adjusted level transfer. Does EEG add to age + sex when a model trained on one level is applied to another?

dAUC = AUC([age, sex] + EEG-stack meta-model) - AUC([age, sex] LR), held-out children, same outer splits.
EEG block = five family L2-LRs (C=1) with inner-OOF logits feeding the meta-model (as RQ1 M_full, without behaviour).
Point estimates R=10; participant bootstrap CI; stratified (2-year age band x sex) EEG-row permutation, R=1, B=500,
one-sided, uncorrected (exploratory). The same EEG permutation is applied to the source and target levels.
"""
import sys, json, warnings, numpy as np, pandas as pd
from joblib import Parallel, delayed
from scipy.special import logit
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
sys.path.insert(0, "src")
import rq1_A3 as M
import rq2_A3 as Q
warnings.filterwarnings("ignore")
DEMO = ["age", "sex_male"]
PAIRS = [(1, 1), (1, 6), (1, 11), (6, 6), (6, 1), (11, 11), (11, 1)]

def data(F, lab, src, tgt):
    subs = sorted(set(F[src].index) & set(F[tgt].index))
    y = lab.loc[subs, "label"].values; D = lab.loc[subs, DEMO].values.astype(float)
    X = {L: {f: F[L].loc[subs, c].values for f, c in M.FAM.items()} for L in {src, tgt}}
    strata = (((lab.loc[subs, "age"] - 6) // 2).astype(int).astype(str) + "_" + lab.loc[subs, "sex_male"].astype(str)).values
    return subs, y, D, X, strata

def run(y, D, X, src, tgt, R, perm=None):
    if perm is not None: X = {L: {f: v[perm] for f, v in X[L].items()} for L in X}
    Sb = np.zeros((R, len(y))); Sf = np.zeros((R, len(y)))
    for r in range(R):
        for k, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=r).split(D, y)):
            inner = StratifiedKFold(5, shuffle=True, random_state=1000 * r + k)
            Z = np.zeros((len(tr), len(M.FAM))); Zt = np.zeros((len(te), len(M.FAM)))
            for j, f in enumerate(M.FAM):
                Xs, Xt = X[src][f], X[tgt][f]
                for itr, ite in inner.split(Xs[tr], y[tr]):
                    Z[ite, j] = logit(M.p1(M.fam_pipe(f).fit(Xs[tr][itr], y[tr][itr]), Xs[tr][ite]))
                Zt[:, j] = logit(M.p1(M.fam_pipe(f).fit(Xs[tr], y[tr]), Xt[te]))
            Sb[r, te] = M.p1(M.tab_pipe().fit(D[tr], y[tr]), D[te])
            Sf[r, te] = M.p1(M.tab_pipe().fit(np.column_stack([D[tr], Z]), y[tr]), np.column_stack([D[te], Zt]))
    sb, sf = Sb.mean(0), Sf.mean(0)
    return roc_auc_score(y, sb), roc_auc_score(y, sf), sb, sf

def one_perm(y, D, X, src, tgt, strata, b):
    p = M.strat_perm(strata, np.random.default_rng(30_000 + b))
    ab, af, _, _ = run(y, D, X, src, tgt, 1, p); return af - ab

if __name__ == "__main__":
    sys.path.insert(0, "src"); import rq2_age_posthoc as P
    B = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    F, lab = Q.load_levels(); rng = np.random.default_rng(2027); rows = []; oof = []
    for src, tgt in P.PAIRS:
        subs, y, D, X, strata = P.data(F, lab, src, tgt)
        ab, af, sb, sf = P.run(y, D, X, src, tgt, 10)
        i1, i0 = np.where(y == 1)[0], np.where(y == 0)[0]; bs = []
        for _ in range(2000):
            i = np.r_[rng.choice(i1, len(i1)), rng.choice(i0, len(i0))]
            bs.append(roc_auc_score(y[i], sf[i]) - roc_auc_score(y[i], sb[i]))
        p = None
        if src != tgt and B > 0:
            ab1, af1, _, _ = P.run(y, D, X, src, tgt, 1)
            null = Parallel(n_jobs=2)(delayed(P.one_perm)(y, D, X, src, tgt, strata, b) for b in range(B))
            p = (1 + np.sum(np.array(null) >= af1 - ab1)) / (B + 1)
        rows.append(dict(train=f"L{src}", test=f"L{tgt}", n=len(y), n_pos=int(y.sum()), AUC_demo=ab, AUC_demo_EEG=af,
                         dAUC=af - ab, dAUC_CI_lo=np.percentile(bs, 2.5), dAUC_CI_hi=np.percentile(bs, 97.5), p_perm_one_sided_uncorrected=p))
        oof.append(pd.DataFrame(dict(train=src, test=tgt, subject=subs, label=y, score_demo=sb, score_demo_eeg=sf)))
        print(rows[-1], flush=True)
    pd.DataFrame(rows).round(4).to_csv("results/POSTHOC_rq2_age_adjusted_transfer.csv", index=False)
    pd.concat(oof).to_csv("results/POSTHOC_rq2_age_adjusted_oof.csv", index=False)
    print(pd.DataFrame(rows).round(3).to_string(index=False))
