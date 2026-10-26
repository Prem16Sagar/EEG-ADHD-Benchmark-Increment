"""A3 section 8: RQ3 (device + task + day transfer, BALLADEER CGX L1 <-> Emotiv AttentionRobots), H_wPLI, RQ4 (cross-cohort).
All on the 10 shared channels (src/shared10.py). Exploratory/descriptive except H_wPLI (one-sided, Holm across 3).

Implementation choices fixed here before any RQ3/RQ4/H_wPLI result:
- Models: five family L2-LRs (C=1, class_weight balanced; F5 via tangent space at the training-domain Riemannian mean)
  and an EEG-only stack (L2-LR on inner-OOF family logits). RQ3 additionally reports dAUC of age+sex+EEG-stack over
  age+sex (consistent with the post-hoc RQ2 age check). R=10 outer 5-fold; participant bootstrap 95% CIs.
- RQ3 sample: children with a usable Emotiv AR recording AND an analysable CGX L1 recording (A2 set).
- H_wPLI: F3 wPLI LR trained on all 121 Nasrabadi children; tested on BALLADEER children aged 7-12 at CGX L6, CGX L11
  (n_bad <= 4 at that level) and Emotiv AR (usable); one-sided label permutation (B=1000); Holm across the three.
- RQ4: Nasrabadi <-> BALLADEER CGX L1, whole source -> whole target, alignment none / per-cohort (z-score; F5 re-centred),
  all ages and 7-12 only; cohort identifiability (5-fold CV LR) per family.
"""
import sys, json, warnings, numpy as np, pandas as pd
from scipy.special import logit
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests
from pyriemann.tangentspace import TangentSpace
from pyriemann.utils.mean import mean_riemann
from pyriemann.utils.base import invsqrtm
sys.path.insert(0, "src")
from shared10 import FAM10
warnings.filterwarnings("ignore")
rng = np.random.default_rng(2028)
FAMS = list(FAM10)

class Cov10(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): self.ts_ = TangentSpace(metric="riemann").fit(np.asarray(X).reshape(-1, 10, 10)); return self
    def transform(self, X): return self.ts_.transform(np.asarray(X).reshape(-1, 10, 10))

def lr(): return LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000)
def fpipe(f): return make_pipeline(Cov10() if f == "F5_riemann" else SimpleImputer(strategy="median"), StandardScaler(), lr())
def p1(m, X): return np.clip(m.predict_proba(X)[:, 1], 1e-6, 1 - 1e-6)

def fit_eeg(Xs, y, seed):
    inner = StratifiedKFold(5, shuffle=True, random_state=seed); Z = np.zeros((len(y), len(FAMS))); ms = {}
    for j, f in enumerate(FAMS):
        for a, b in inner.split(Xs[f], y): Z[b, j] = logit(p1(fpipe(f).fit(Xs[f][a], y[a]), Xs[f][b]))
        ms[f] = fpipe(f).fit(Xs[f], y)
    return ms, Z

def eeg_scores(ms, Xt):
    return np.column_stack([logit(p1(ms[f], Xt[f])) for f in FAMS])

def boot(y, s, s0=None, nb=2000):
    i1, i0 = np.where(y == 1)[0], np.where(y == 0)[0]; out = []
    for _ in range(nb):
        i = np.r_[rng.choice(i1, len(i1)), rng.choice(i0, len(i0))]
        out.append(roc_auc_score(y[i], s[i]) - (roc_auc_score(y[i], s0[i]) if s0 is not None else 0))
    return np.percentile(out, [2.5, 97.5])

def mats(D, cols_by_fam): return {f: D[c].values for f, c in cols_by_fam.items()}

# ------------------------------------------------------------------ RQ3
def rq3():
    lab = pd.read_csv("derived/features_balladeer.csv"); lab = lab[lab.n_bad <= 4].set_index("subject")[["label", "age", "sex_male"]]
    E = pd.read_csv("derived/features10_emotiv_AR.csv"); E = E[E.usable].set_index("subject")
    C = pd.read_csv("derived/features10_cgx_L1.csv").set_index("subject")
    subs = sorted(set(E.index) & set(lab.index) & set(C.index))
    y = lab.loc[subs, "label"].values; Dm = lab.loc[subs, ["age", "sex_male"]].values.astype(float)
    dom = {"CGX_L1": mats(C.loc[subs], FAM10), "Emotiv_AR": mats(E.loc[subs], FAM10)}
    rows = []
    for src, tgt in [("Emotiv_AR", "Emotiv_AR"), ("CGX_L1", "CGX_L1"), ("CGX_L1", "Emotiv_AR"), ("Emotiv_AR", "CGX_L1")]:
        S = {m: np.zeros((10, len(y))) for m in FAMS + ["stack", "demo", "demo_eeg"]}
        for r in range(10):
            for k, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=r).split(Dm, y)):
                Xs = {f: v[tr] for f, v in dom[src].items()}; Xt = {f: v[te] for f, v in dom[tgt].items()}
                ms, Z = fit_eeg(Xs, y[tr], 1000 * r + k); Zt = eeg_scores(ms, Xt)
                for j, f in enumerate(FAMS): S[f][r, te] = Zt[:, j]
                S["stack"][r, te] = p1(lr().fit(Z, y[tr]), Zt)
                tp = lambda: make_pipeline(StandardScaler(), lr())
                S["demo"][r, te] = p1(tp().fit(Dm[tr], y[tr]), Dm[te])
                S["demo_eeg"][r, te] = p1(tp().fit(np.column_stack([Dm[tr], Z]), y[tr]), np.column_stack([Dm[te], Zt]))
        S = {m: v.mean(0) for m, v in S.items()}
        for m in FAMS + ["stack"]:
            lo, hi = boot(y, S[m]); rows.append(dict(analysis="RQ3", train=src, test=tgt, model=m, n=len(y), n_pos=int(y.sum()),
                                                      AUC=roc_auc_score(y, S[m]), CI_lo=lo, CI_hi=hi))
        lo, hi = boot(y, S["demo_eeg"], S["demo"])
        rows.append(dict(analysis="RQ3", train=src, test=tgt, model="dAUC demo+EEG minus demo", n=len(y), n_pos=int(y.sum()),
                         AUC=roc_auc_score(y, S["demo_eeg"]) - roc_auc_score(y, S["demo"]), CI_lo=lo, CI_hi=hi,
                         note=f"AUC demo {roc_auc_score(y, S['demo']):.3f}; demo+EEG {roc_auc_score(y, S['demo_eeg']):.3f}"))
        print(src, tgt, "stack", round(rows[-2]["AUC"], 3), "dAUC", round(rows[-1]["AUC"], 3), flush=True)
    return rows

# ------------------------------------------------------------------ H_wPLI
def h_wpli():
    N = pd.read_csv("derived/features10_nasrabadi.csv"); cols = FAM10["F3_connectivity"]
    m = fpipe("F3_connectivity").fit(N[cols].values, N.label.values)
    base = pd.read_csv("derived/features_balladeer.csv").set_index("subject")[["label", "age"]]
    tests = {}
    for L in (6, 11):
        Fq = pd.read_csv(f"derived/features_balladeer_L{L}.csv"); ok = set(Fq[Fq.n_bad <= 4].subject)
        tests[f"CGX_L{L}"] = pd.read_csv(f"derived/features10_cgx_L{L}.csv").set_index("subject").loc[sorted(ok & set(base.index))]
    E = pd.read_csv("derived/features10_emotiv_AR.csv"); E = E[E.usable].set_index("subject")
    tests["Emotiv_AR"] = E.loc[sorted(set(E.index) & set(base.index))]
    rows = []
    for name, T in tests.items():
        b = base.loc[T.index]; keep = (b.age >= 7) & (b.age <= 12); T = T[keep.values]; y = b.label.values[keep.values]
        s = p1(m, T[cols].values); auc = roc_auc_score(y, s)
        null = np.array([roc_auc_score(rng.permutation(y), s) for _ in range(1000)])
        lo, hi = boot(y, s)
        rows.append(dict(analysis="H_wPLI", train="Nasrabadi (all 121)", test=f"{name} (7-12 y)", model="F3_wPLI_LR", n=len(y),
                         n_pos=int(y.sum()), AUC=auc, CI_lo=lo, CI_hi=hi, p_one_sided=(1 + np.sum(null >= auc)) / 1001))
    ph = multipletests([r["p_one_sided"] for r in rows], method="holm")[1]
    for r, p in zip(rows, ph): r["p_holm"] = p
    for r in rows: print(r["test"], round(r["AUC"], 3), r["p_one_sided"], round(r["p_holm"], 3), flush=True)
    return rows

# ------------------------------------------------------------------ RQ4
def recentre(D, cols):
    Cm = D[cols].values.reshape(-1, 10, 10); W = invsqrtm(mean_riemann(Cm)); D = D.copy()
    D[cols] = np.einsum("ij,njk,kl->nil", W, Cm, W).reshape(len(D), -1); return D

def align(D):
    D = recentre(D, FAM10["F5_riemann"]); other = sum([FAM10[f] for f in FAMS if f != "F5_riemann"], [])
    X = D[other].astype(float); D[other] = (X - X.median()) / (X.std() + 1e-12); return D

def rq4():
    N = pd.read_csv("derived/features10_nasrabadi.csv")
    B0 = pd.read_csv("derived/features_balladeer.csv"); B0 = B0[B0.n_bad <= 4][["subject", "label", "age"]]
    B = pd.read_csv("derived/features10_cgx_L1.csv").merge(B0, on="subject")
    rows = []
    for ages in ["all", "7-12"]:
        Bx = B if ages == "all" else B[(B.age >= 7) & (B.age <= 12)]
        for al in ["none", "cohort_norm"]:
            Na, Ba = (N, Bx) if al == "none" else (align(N), align(Bx))
            for name, S_, T_ in [("Nasrabadi->BALLADEER", Na, Ba), ("BALLADEER->Nasrabadi", Ba, Na)]:
                ys, yt = S_.label.values, T_.label.values
                ms, Z = fit_eeg(mats(S_, FAM10), ys, 0); Zt = eeg_scores(ms, mats(T_, FAM10))
                sc = {f: Zt[:, j] for j, f in enumerate(FAMS)}; sc["stack"] = p1(lr().fit(Z, ys), Zt)
                for mname, s in sc.items():
                    lo, hi = boot(yt, s)
                    rows.append(dict(analysis="RQ4", train=name.split("->")[0], test=name.split("->")[1], ages=ages, alignment=al,
                                     model=mname, n=len(yt), n_pos=int(yt.sum()), AUC=roc_auc_score(yt, s), CI_lo=lo, CI_hi=hi))
            D = pd.concat([Na.assign(cohort=0), Ba.assign(cohort=1)], ignore_index=True)
            for f in FAMS:
                p = cross_val_predict(fpipe(f), D[FAM10[f]].values, D.cohort.values, cv=StratifiedKFold(5, shuffle=True, random_state=0),
                                      method="predict_proba")[:, 1]
                rows.append(dict(analysis="RQ4", train="cohort identifiability", test="", ages=ages, alignment=al, model=f,
                                 n=len(D), AUC=roc_auc_score(D.cohort, p)))
    T = pd.DataFrame(rows)
    print(T[(T.model == "stack") | (T.train == "cohort identifiability")].round(3).to_string(index=False), flush=True)
    return rows

if __name__ == "__main__":
    which = sys.argv[1]
    if which in ("rq3", "all"): pd.DataFrame(rq3()).round(4).to_csv("results/A3_rq3_device_transfer.csv", index=False)
    if which in ("hwpli", "all"): pd.DataFrame(h_wpli()).round(4).to_csv("results/A3_h_wpli.csv", index=False)
    if which in ("rq4", "all"): pd.DataFrame(rq4()).round(4).to_csv("results/A3_rq4_cross_cohort10.csv", index=False)
