"""BALLADEER participant-level features F1-F5 with the identical code used for Nasrabadi
(extract_features.py), applied to the A2-referenced, 128-Hz, interpolated arrays."""
import sys, glob, os, json, warnings, numpy as np, pandas as pd
sys.path.insert(0, "src")
import extract_features as EF
warnings.filterwarnings("ignore")
import sys as _s, os as _o; _s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__))); from paths import BALLADEER_ROOT; DEMO = _o.path.join(BALLADEER_ROOT, "users_demographics.json")

def main():
    rows, covs = [], []
    files = sorted(glob.glob("derived/balladeer_npz/*.npz"))
    for n, f in enumerate(files):
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
        covs.append(np.mean([np.cov(w) for w in wins], 0)); rows.append(feats)
        if (n+1) % 20 == 0: print(n+1, flush=True)
    D = pd.DataFrame(rows); covs = np.array(covs)
    for i in range(19):
        for j in range(19): D[f"cov_{i}_{j}"] = covs[:, i, j]
    demo = pd.DataFrame(json.load(open(DEMO)))
    conf = pd.concat([pd.read_csv(p) for p in glob.glob("derived/balladeer_confounds*.csv")]).drop_duplicates("user", keep="last")
    D = D.merge(demo, on="user").merge(conf, on="user")
    D["label"] = (D.diagnosed == "yes").astype(int)
    D["sex_male"] = (D.gender == 1).astype(int)
    D.rename(columns={"user": "subject", "emg": "muscle_temporal_emg", "line": "line_ratio", "drift": "drift_ratio",
                      "transient": "transient_frac"}, inplace=True)
    D.to_csv("derived/features_balladeer.csv", index=False)
    print("saved", D.shape, "min eig", float(np.linalg.eigvalsh(covs).min()))

if __name__ == "__main__":
    main()
