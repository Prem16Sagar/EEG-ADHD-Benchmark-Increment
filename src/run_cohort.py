"""Run the locked Nasrabadi pipeline (protocol.yaml + addendum A1) on BALLADEER (addendum A2).
Usage: python src/run_cohort.py primary|sensitivity"""
import sys, time, warnings, numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
sys.path.insert(0, "src")
VARIANT = sys.argv[1] if len(sys.argv) > 1 else "primary"; sys.argv = sys.argv[:1]   # run_ensembles parses argv at import
import run_ensembles as R
import run_a1 as A
warnings.filterwarnings("ignore")
DEMO = ["age", "sex_male"]
ACQ = ["duration_s", "muscle_temporal_emg", "line_ratio", "drift_ratio", "transient_frac", "blink_frontal_kurt", "motion_acc_sd"]
CONF = DEMO + ACQ
R.CONF = CONF; A.CONF = CONF

def load(variant):
    D = pd.read_csv("derived/features_balladeer.csv")
    D = D[D.n_bad <= 4].copy()                      # addendum A2 exclusion rule (>4 of 19 bad channels)
    if variant == "sensitivity":
        D = D[(D.group == 1) | (D.diagnosed == "no")].copy()
        D["label"] = (D.group == 1).astype(int)
    D = D.sort_values("subject").reset_index(drop=True)
    CH, B = R.CH, R.BANDS
    fam = {"F1_spectral": [f"rel_{b}_mean_{c}" for b in B for c in CH],
           "F2_aperiodic": [f"{p}_{c}" for p in ["ap_exp", "ap_off", "alpha_cf", "alpha_pw"] for c in CH],
           "F3_connectivity": [f"wpli_{b}_{c}" for b in B for c in CH],
           "F4_complexity": [f"{p}_{c}" for p in ["perm", "higuchi"] for c in CH],
           "F5_riemann": [f"cov_{i}_{j}" for i in range(19) for j in range(19)]}
    chan = {c: [f"rel_{b}_mean_{c}" for b in B] + [f"{p}_{c}" for p in ["ap_exp", "ap_off", "alpha_cf", "alpha_pw"]]
               + [f"wpli_{b}_{c}" for b in B] + [f"perm_{c}", f"higuchi_{c}"] for c in CH}
    return D, fam, chan

def extra_benchmarks(rep, k, tr, te, D):
    y = D.label.values; L = R.learners(rep); out = {}
    for name, cols in [("B0_acq_vote", ACQ), ("B0_demo_vote", DEMO)]:
        X = D[cols].values
        out[name] = np.mean([R.proba(make_pipeline(StandardScaler(), clone(L[l])).fit(X[tr], y[tr]), X[te]) for l in L], 0)
    return [dict(rep=rep, fold=k, subject=D.subject[i], label=int(y[i]), method=m, score=float(s[n]))
            for m, s in out.items() for n, i in enumerate(te)]

def job(rep, k, tr, te, D, fam, chan):
    R.CONF = CONF; A.CONF = CONF          # set inside the worker process (module globals are not shipped by joblib)
    rows, stats = R.run_fold(rep, k, tr, te, D, fam, chan)
    a1 = [r for r in A.run_fold(rep, k, tr, te, D, fam) if r["method"].startswith("S_") and "nontemporal" not in r["method"]]
    return rows + a1 + extra_benchmarks(rep, k, tr, te, D), stats

def main(variant):
    D, fam, chan = load(variant); y = D.label.values
    print(variant, "n =", len(D), "ADHD =", int(y.sum()), flush=True)
    jobs = [(r, k, tr, te) for r in range(10) for k, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=r).split(D, y))]
    t0 = time.time()
    res = Parallel(n_jobs=2, verbose=5)(delayed(job)(r, k, tr, te, D, fam, chan) for r, k, tr, te in jobs)
    pd.DataFrame([x for rows, _ in res for x in rows]).to_csv(f"results/balladeer_{variant}_oof_scores.csv.gz", index=False)
    pd.concat([s for _, s in res]).to_csv(f"results/balladeer_{variant}_inner_member_stats.csv", index=False)
    print("done in %.1f min" % ((time.time()-t0)/60))

if __name__ == "__main__":
    main(VARIANT)
