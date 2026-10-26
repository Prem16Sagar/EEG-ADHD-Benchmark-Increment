"""Addendum A4 post-hoc sensitivity analyses S1-S6 (see configs/protocol_addendum_A4_posthoc.yaml).
Usage: python src/posthoc_A4.py s1|s2|s3|s4feat|s4|s5|s6score|s6 [B]
"""
import sys, os, glob, json, ast, warnings, numpy as np, pandas as pd
CMD = sys.argv[1] if len(sys.argv) > 1 else ""; B = int(sys.argv[2]) if len(sys.argv) > 2 else 500; sys.argv = sys.argv[:1]
from joblib import Parallel, delayed
from scipy.special import logit
from scipy import signal
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, "src")
import rq1_A3 as M
warnings.filterwarnings("ignore")
OUT = "results/A4"; os.makedirs(OUT, exist_ok=True)
import sys as _s, os as _o; _s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__))); from paths import BALLADEER_ROOT; ROOT = BALLADEER_ROOT.rstrip("/") + "/"
rng = np.random.default_rng(4040)

# ---------------------------------------------------------------- generic RQ1-style contrast
def fold(D, y, tr, te, rep, k, tab, fams):
    Xd = D[tab].values.astype(float)
    pb = M.p1(M.tab_pipe().fit(Xd[tr], y[tr]), Xd[te])
    inner = StratifiedKFold(5, shuffle=True, random_state=1000 * rep + k)
    Z = np.zeros((len(tr), len(fams))); Zt = np.zeros((len(te), len(fams)))
    for j, (f, cols) in enumerate(fams.items()):
        X = D[cols].values
        fp = "F5_riemann" if f == "F5_riemann" else "F1_spectral"
        for a, b in inner.split(X[tr], y[tr]): Z[b, j] = logit(M.p1(M.fam_pipe(fp).fit(X[tr][a], y[tr][a]), X[tr][b]))
        Zt[:, j] = logit(M.p1(M.fam_pipe(fp).fit(X[tr], y[tr]), X[te]))
    pf = M.p1(M.tab_pipe().fit(np.column_stack([Xd[tr], Z]), y[tr]), np.column_stack([Xd[te], Zt]))
    return pb, pf

def run(D, y, R, tab, fams, perm=None):
    if perm is not None:
        D = D.copy(); cols = sum(fams.values(), []); D[cols] = D[cols].values[perm]
    Sb = np.zeros((R, len(y))); Sf = np.zeros((R, len(y))); pfold = []
    for r in range(R):
        for k, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=r).split(D, y)):
            Sb[r, te], Sf[r, te] = fold(D, y, tr, te, r, k, tab, fams)
            pfold.append(roc_auc_score(y[te], Sf[r, te]) - roc_auc_score(y[te], Sb[r, te]))
    sb, sf = Sb.mean(0), Sf.mean(0)
    return dict(auc_base=roc_auc_score(y, sb), auc_full=roc_auc_score(y, sf), dauc=roc_auc_score(y, sf) - roc_auc_score(y, sb),
                dauc_perfold_mean=float(np.mean(pfold))), sb, sf

def one_perm(D, y, tab, fams, strata, b):
    return run(D, y, 1, tab, fams, M.strat_perm(strata, np.random.default_rng(50_000 + b)))[0]["dauc"]

def boot(y, sb, sf, nb=2000):
    i1, i0 = np.where(y == 1)[0], np.where(y == 0)[0]; d = []
    for _ in range(nb):
        i = np.r_[rng.choice(i1, len(i1)), rng.choice(i0, len(i0))]; d.append(roc_auc_score(y[i], sf[i]) - roc_auc_score(y[i], sb[i]))
    return np.percentile(d, [2.5, 97.5])

def contrast(tag, D, tab, fams, B):
    y = D.label.values; strata = D.stratum.values
    est, sb, sf = run(D, y, 10, tab, fams); lo, hi = boot(y, sb, sf)
    p = None
    if B > 0:
        o1 = run(D, y, 1, tab, fams)[0]["dauc"]
        null = np.array(Parallel(n_jobs=2)(delayed(one_perm)(D, y, tab, fams, strata, b) for b in range(B)))
        p = (1 + np.sum(null >= o1)) / (B + 1); est["dauc_R1_for_p"] = o1
    res = dict(analysis=tag, n=int(len(y)), n_pos=int(y.sum()), **est, dauc_CI_lo=lo, dauc_CI_hi=hi, B=B, p_perm_one_sided=p)
    pd.DataFrame(dict(subject=D.subject, label=y, score_base=sb, score_full=sf)).to_csv(f"{OUT}/{tag}_oof.csv", index=False)
    json.dump(res, open(f"{OUT}/{tag}_summary.json", "w"), indent=1, default=float); print(res, flush=True); return res

# ---------------------------------------------------------------- simple CV benchmark (A2 learners)
def bench(X, y, R=10, learners="vote"):
    sys.argv = sys.argv[:1]
    import run_ensembles as RE
    S = np.zeros((R, len(y))); per = []
    for r in range(R):
        L = RE.learners(r)
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=r).split(X, y):
            ms = L if learners == "vote" else {"lr": L["lr"]}
            S[r, te] = np.mean([RE.proba(make_pipeline(StandardScaler(), clone(m)).fit(X[tr], y[tr]), X[te]) for m in ms.values()], 0)
        per.append(roc_auc_score(y, S[r]))
    s = S.mean(0); i1, i0 = np.where(y == 1)[0], np.where(y == 0)[0]
    bs = [roc_auc_score(y[i], s[i]) for i in (np.r_[rng.choice(i1, len(i1)), rng.choice(i0, len(i0))] for _ in range(2000))]
    return dict(AUC_avgscore=roc_auc_score(y, s), CI_lo=np.percentile(bs, 2.5), CI_hi=np.percentile(bs, 97.5), AUC_mean_reps=float(np.mean(per)))

def load_L1():
    D = M.load("primary"); return D

# ---------------------------------------------------------------- S4 features
def f2_refit_from_array(X):
    import extract_features as EF
    from fooof import FOOOF
    x = signal.detrend(X.astype(float), axis=-1); x = signal.filtfilt(*EF.NT, x, axis=-1)
    b, a = signal.butter(4, 1 / (EF.FS / 2), btype="high"); x = signal.filtfilt(b, a, x, axis=-1)
    wins = EF.fixedK_windows(x); f, P = EF.welch(np.stack(wins, 0)); Pm = P.mean(0); out = {}
    for ci, ch in enumerate(EF.CHANNELS):
        fm = FOOOF(peak_width_limits=(1, 8), max_n_peaks=4, aperiodic_mode="fixed", verbose=False); fm.fit(f, Pm[ci], [2, 40])
        off, ex = fm.aperiodic_params_; pk = fm.peak_params_
        al = pk[(pk[:, 0] >= 7) & (pk[:, 0] <= 13)] if len(pk) else np.empty((0, 3))
        cf, pw = (al[np.argmax(al[:, 1]), 0], al[np.argmax(al[:, 1]), 1]) if len(al) else (np.nan, 0.0)
        out.update({f"ap_exp_{ch}": ex, f"ap_off_{ch}": off, f"alpha_cf_{ch}": cf, f"alpha_pw_{ch}": pw, f"r2_{ch}": fm.r_squared_})
    return out

def s4feat():
    for L, d in [(1, "derived/balladeer_npz"), (6, "derived/balladeer_npz_L6"), (11, "derived/balladeer_npz_L11")]:
        rows = Parallel(n_jobs=2)(delayed(lambda p: dict(subject=os.path.basename(p)[:-4], **f2_refit_from_array(np.load(p)["X"])))(p)
                                  for p in sorted(glob.glob(d + "/*.npz")))
        T = pd.DataFrame(rows); T.to_csv(f"derived/features_F2refit_L{L}.csv", index=False)
        print(L, T.shape, "median R2", round(float(T.filter(like="r2_").median().median()), 3), flush=True)

# ---------------------------------------------------------------- S6 scoring
def s6score():
    inv = pd.read_csv("balladeer/A3_file_inventory_121.csv")
    FL = {l["level"]: l["flags"] for l in json.load(open(ROOT + "slackline_flags_info.json"))["slackline_levels_flags_info"]}
    tgt_names = {"circle", "square"}; NT = sum(f["flag_type"] in tgt_names for f in FL["Level6"]); NN = len(FL["Level6"]) - NT
    rows = []
    for _, r in inv.iterrows():
        if not isinstance(r.SlacklineLvl6_tagspath, str): continue
        E = pd.DataFrame([ast.literal_eval(v)["reactionOrOmission"][0] for v in pd.read_csv(ROOT + r.user + "/" + r.SlacklineLvl6_tagspath)["value"]])
        for c in ["reacted", "duplicated"]: E[c] = E[c].astype(str) == "True"
        E["ft"] = E.flagType.apply(lambda x: x[0])
        hits = E[E.ft.isin([0, 1]) & E.reacted & ~E["duplicated"]]; comm = E[E.ft.isin([2, 3]) & E.reacted & ~E["duplicated"]]
        rt = hits.reactionTime.values
        rows.append(dict(subject=r.user, n_hit=len(hits), n_comm=len(comm), s2_ok=bool(len(hits) <= NT and len(comm) <= NN),
                         B1_omission_rate=(NT - min(len(hits), NT)) / NT, B2_commission_rate=min(len(comm), NN) / NN,
                         B3_mean_RT_hits=rt.mean() if len(rt) >= 3 else np.nan,
                         B4_RT_cv_hits=rt.std(ddof=1) / rt.mean() if len(rt) >= 3 else np.nan))
    T = pd.DataFrame(rows); T.to_csv("derived/behaviour_L6_A4.csv", index=False)
    print("targets", NT, "nontargets", NN, T.shape, "s2_ok", int(T.s2_ok.sum()), "hits<3", int((T.n_hit < 3).sum()))
    print(T.drop(columns="subject").describe().T[["mean", "50%", "max"]].round(3))

if __name__ == "__main__":
    sys.path.insert(0, "src"); import posthoc_A4 as P
    if CMD == "s1":
        D = P.load_L1(); D = D[D.age <= 17].reset_index(drop=True)
        P.contrast("S1_age_le17_rq1", D, M.DEMO + M.BEH, M.FAM, B)
        y = D.label.values
        rows = [dict(model="age+sex LR", **P.bench(D[M.DEMO].values.astype(float), y, learners="lr")),
                dict(model="age+sex 6-learner vote", **P.bench(D[M.DEMO].values.astype(float), y))]
        pd.DataFrame(rows).round(4).to_csv(f"{OUT}/S1_age_le17_benchmarks.csv", index=False); print(pd.DataFrame(rows).round(3))
    elif CMD == "s2":
        D = P.load_L1()
        bands = pd.cut(D.age, [5, 8, 11, 14, 17, 18], labels=["b6_8", "b9_11", "b12_14", "b15_17", "b18"])
        for c in bands.cat.categories: D[c] = (bands == c).astype(int)
        P.contrast("S2_flexible_base_rq1", D, list(bands.cat.categories) + ["sex_male"] + M.BEH, M.FAM, B)
    elif CMD == "s3":
        A2 = pd.read_csv("results/balladeer_primary_oof_scores.csv.gz")
        A2 = A2[A2.method.isin(["B0_demo_vote", "B2_F1_spectral_lr"])].groupby(["method", "subject"]).score.mean().unstack(0)
        A3 = pd.read_csv("results/A3_rq1_primary_est_oof.csv").set_index("subject")
        D = P.load_L1().set_index("subject"); S = pd.concat([A2, A3[["score_base", "score_full"]]], axis=1).loc[D.index]
        ad = D[D.label == 1]; ct = D[D.label == 0]; res = {m: [] for m in S.columns}; npairs = []
        for it in range(500):
            r = np.random.default_rng(60_000 + it); used = set(); pairs = []
            for c in r.permutation(ct.index):
                cand = [a for a in ad.index if a not in used and ad.at[a, "sex_male"] == ct.at[c, "sex_male"] and abs(ad.at[a, "age"] - ct.at[c, "age"]) <= 1]
                if cand: a = cand[r.integers(len(cand))]; used.add(a); pairs.append((a, c))
            npairs.append(len(pairs)); pi = r.integers(0, len(pairs), len(pairs)); pr = [pairs[i] for i in pi]
            ids = [a for a, _ in pr] + [c for _, c in pr]; yy = np.r_[np.ones(len(pr)), np.zeros(len(pr))]
            for m in S.columns: res[m].append(roc_auc_score(yy, S.loc[ids, m].values))
        T = pd.DataFrame([dict(score=m, AUC_median=np.median(v), lo=np.percentile(v, 2.5), hi=np.percentile(v, 97.5)) for m, v in res.items()])
        T["median_pairs"] = np.median(npairs); T.round(4).to_csv(f"{OUT}/S3_matched_subsets.csv", index=False); print(T.round(3))
    elif CMD == "s4feat":
        P.s4feat()
    elif CMD == "s4":
        import rq2_A3 as Q
        F2c = [f"{p}_{c}" for p in ["ap_exp", "ap_off", "alpha_cf", "alpha_pw"] for c in M.CH]
        lab = pd.read_csv("derived/features_balladeer.csv"); okL = {1: set(lab[lab.n_bad <= 4].subject)}
        for L in (6, 11): t = pd.read_csv(f"derived/features_balladeer_L{L}.csv"); okL[L] = set(t[t.n_bad <= 4].subject)
        R = {L: pd.read_csv(f"derived/features_F2refit_L{L}.csv").set_index("subject") for L in (1, 6, 11)}
        subs = sorted(set.intersection(*[okL[L] & set(R[L].index) for L in (1, 6, 11)]))
        X = np.stack([R[L].loc[subs, F2c].values.astype(float) for L in (1, 6, 11)], 1); keep = ~np.isnan(X).any((0, 1))
        a, c = Q.icc(X[:, :, keep])
        out = dict(n=len(subs), n_features=int(keep.sum()), ICC_A1_median=float(np.nanmedian(a)), ICC_A1_q25=float(np.nanpercentile(a, 25)),
                   ICC_A1_q75=float(np.nanpercentile(a, 75)), median_fit_R2_L1=float(R[1].filter(like="r2_").median().median()))
        D = P.load_L1(); old = D[F2c].copy(); D = D.drop(columns=F2c).merge(R[1][F2c].reset_index(), on="subject")
        y = D.label.values
        out["F2refit_LR_AUC_R10"] = P.bench(D[F2c].fillna(D[F2c].median()).values, y, learners="lr")["AUC_avgscore"]
        r = P.contrast("S4_F2refit_rq1", D, M.DEMO + M.BEH, M.FAM, 0); out["rq1_dauc_with_F2refit"] = r["dauc"]
        json.dump(out, open(f"{OUT}/S4_F2refit_summary.json", "w"), indent=1); print(out)
    elif CMD == "s5":
        N = pd.read_csv("derived/features_nasrabadi.csv")[["subject", "label"]].merge(pd.read_csv("derived/confounds_from_previous_paper.csv").drop(columns="label"), on="subject").sort_values("subject")
        y = N.label.values; sets = {"duration only": ["duration_s"],
                                    "recording-condition proxies": ["line_ratio", "muscle_temporal_emg", "drift_ratio", "transient_frac", "blink_frontal_kurt"],
                                    "all six (= B0)": ["duration_s", "line_ratio", "muscle_temporal_emg", "drift_ratio", "transient_frac", "blink_frontal_kurt"]}
        T = pd.DataFrame([dict(benchmark=k, **P.bench(N[v].values.astype(float), y)) for k, v in sets.items()])
        T.round(4).to_csv(f"{OUT}/S5_nasrabadi_benchmark_decomposition.csv", index=False); print(T.round(3))
    elif CMD == "s6score":
        P.s6score()
    elif CMD == "s6":
        F = pd.read_csv("derived/features_balladeer_L6.csv"); F = F[(F.n_bad <= 4) & F.diagnosed.isin(["yes", "no"])]
        Bh = pd.read_csv("derived/behaviour_L6_A4.csv"); Bh = Bh[Bh.s2_ok]
        D = F.merge(Bh[["subject"] + M.BEH], on="subject").sort_values("subject").reset_index(drop=True)
        D["stratum"] = ((D.age - 6) // 2).astype(int).astype(str) + "_" + D.sex_male.astype(int).astype(str)
        P.contrast("S6_L6_behaviour_rq1", D, M.DEMO + M.BEH, M.FAM, B)
