"""A3 RQ2 data: BALLADEER CGX Slackline Level 6 / Level 11 -> arrays -> features F1-F5.
Identical processing to A2 L1 (balladeer_convert.convert, balladeer_features.main), parametrised by level.
Covariance ridge 1e-3*trace/19 added exactly as for the stored L1 features (A2 deviation 1; verified on UB0004).
Label-free. The release version reads raw files in place and never deletes them.

Usage: python src/balladeer_level.py convert <level>
       python src/balladeer_level.py features <level>
"""
import sys, os, glob, json, time, warnings, numpy as np, pandas as pd
from scipy import signal, stats
sys.path.insert(0, "src")
import feat_utils as FU
from balladeer_convert import CH19, ROOT, interpolate_bad
warnings.filterwarnings("ignore")

def out_dir(level): d = f"derived/balladeer_npz_L{level}"; os.makedirs(d, exist_ok=True); return d

def convert_file(user, path, level):
    d = pd.read_csv(path, usecols=["timestamps"] + [f"{c}(uV)" for c in CH19] + ["A2(uV)", "ACC32(mg)", "ACC33(mg)", "ACC34(mg)"])
    t = d.timestamps.values; fs = (len(t) - 1) / (t[-1] - t[0])
    X = d[[f"{c}(uV)" for c in CH19]].values.T - d["A2(uV)"].values[None, :]
    X = signal.detrend(X, axis=-1, type="linear")
    X128 = signal.resample_poly(X, 32, 125, axis=-1)
    acc = d[["ACC32(mg)", "ACC33(mg)", "ACC34(mg)"]].values
    motion = float(np.std(np.linalg.norm(acc - acc.mean(0), axis=1)))
    Xf = FU.preprocess(X128, lp=30); sd = Xf.std(1); med = np.median(sd)
    bad = [i for i in range(19) if sd[i] < 1e-6 or sd[i] > 5 * med]
    X128c = interpolate_bad(X128, bad)
    x45 = FU.preprocess(X128c, lp=45)
    conf = FU.confound_features(X128c, x45)
    fr = [CH19.index(c) for c in ["Fp1", "Fp2"]]
    conf.update(blink_frontal_kurt=float(stats.kurtosis(x45[fr], axis=1).mean()), motion_acc_sd=motion,
                fs_est=float(fs), n_bad=len(bad), bad_channels=",".join(CH19[i] for i in bad))
    np.savez_compressed(os.path.join(out_dir(level), f"{user}.npz"), X=X128c.astype(np.float32))
    return dict(user=user, **conf)

def targets(level):
    """Release version: the CGX file of each participant is the single *_EEG_CGX* file in the folder of the
    participant's TAGS file listed in balladeer/A3_file_inventory_121.csv (checked: identical to the files used, 356/356)."""
    inv = pd.read_csv("balladeer/A3_file_inventory_121.csv")
    out = {}
    for _, r in inv.iterrows():
        tp = r.get(f"SlacklineLvl{level}_tagspath")
        if not isinstance(tp, str): continue
        c = sorted(glob.glob(os.path.join(ROOT, r.user, os.path.dirname(tp), "*_EEG_CGX*.csv")))
        if c: out[r.user] = c[0]
    return out

def convert(level):
    """Release version: converts every available recording once. Raw files are never modified or deleted."""
    T = targets(level); log = f"derived/balladeer_confounds_L{level}.csv"; rows = []
    for u, p in T.items():
        try:
            rows.append(convert_file(u, p, level)); print(u, rows[-1]["n_bad"], round(rows[-1]["duration_s"], 1), flush=True)
        except Exception as e:
            print("FAIL", u, repr(e), flush=True)
    pd.DataFrame(rows).to_csv(log, index=False); print("converted", len(rows), "of", len(T))

def features(level):
    import extract_features as EF
    rows, covs = [], []
    for f in sorted(glob.glob(out_dir(level) + "/*.npz")):
        X = np.load(f)["X"].astype(float)
        X30 = EF.base_filter(X, 30); wins = EF.fixedK_windows(X30)
        feats = dict(user=os.path.basename(f)[:-4])
        rp = [EF.relpow(w, EF.BANDS4) for w in wins]
        for b in EF.BANDS4:
            arr = np.array([r[b] for r in rp])
            for ci, ch in enumerate(EF.CHANNELS): feats[f"rel_{b}_mean_{ch}"] = arr[:, ci].mean()
        pe = np.array([EF.perm_entropy_multi(w) for w in wins]); hg = np.array([EF.higuchi_multi(w) for w in wins])
        for ci, ch in enumerate(EF.CHANNELS): feats[f"perm_{ch}"] = pe[:, ci].mean(); feats[f"higuchi_{ch}"] = hg[:, ci].mean()
        feats.update(EF.wpli_bandfeatures(X30)); feats.update(EF.fooof_features(wins))
        C = np.mean([np.cov(w) for w in wins], 0); C = C + 1e-3 * np.trace(C) / 19 * np.eye(19)   # A2 ridge
        covs.append(C); rows.append(feats)
    D = pd.DataFrame(rows); covs = np.array(covs)
    for i in range(19):
        for j in range(19): D[f"cov_{i}_{j}"] = covs[:, i, j]
    demo = pd.DataFrame(json.load(open(ROOT + "/users_demographics.json")))
    conf = pd.read_csv(f"derived/balladeer_confounds_L{level}.csv")
    D = D.merge(demo, on="user").merge(conf, on="user")
    D["label"] = (D.diagnosed == "yes").astype(int); D["sex_male"] = (D.gender == 1).astype(int)
    D.rename(columns={"user": "subject", "emg": "muscle_temporal_emg", "line": "line_ratio", "drift": "drift_ratio",
                      "transient": "transient_frac"}, inplace=True)
    D["level"] = level
    D.to_csv(f"derived/features_balladeer_L{level}.csv", index=False)
    print("saved", D.shape, "n_bad>4:", int((D.n_bad > 4).sum()))

if __name__ == "__main__":
    {"convert": convert, "convert-watch": convert, "features": features}[sys.argv[1]](int(sys.argv[2]))
