"""BALLADEER (CGX Slackline Level 1) -> compact per-participant arrays (addendum A2).
Label-free and per participant: A2 re-reference, 500->128 Hz, bad-channel interpolation,
acquisition confounds computed with the previous paper's functions (feat_utils)."""
import sys, os, glob, json, warnings, numpy as np, pandas as pd
from scipy import signal, stats
sys.path.insert(0, "src")
import feat_utils as FU
warnings.filterwarnings("ignore")
CH19 = ["Fp1","Fp2","F3","F4","C3","C4","P3","P4","O1","O2","F7","F8","T7","T8","P7","P8","Fz","Cz","Pz"]
import sys as _s, os as _o; _s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__))); from paths import BALLADEER_ROOT as ROOT
OUT = "derived/balladeer_npz"; os.makedirs(OUT, exist_ok=True)

def interpolate_bad(X128, bad):
    if not bad: return X128
    import mne
    info = mne.create_info(CH19, 128.0, "eeg"); raw = mne.io.RawArray(X128*1e-6, info, verbose="ERROR")
    raw.set_montage("standard_1020", verbose="ERROR"); raw.info["bads"] = [CH19[i] for i in bad]
    raw.interpolate_bads(reset_bads=True, verbose="ERROR")
    return raw.get_data()*1e6

def convert(user, rel):
    f = os.path.join(ROOT, user, "SlacklineLvl1", rel)
    d = pd.read_csv(f, usecols=["timestamps"] + [f"{c}(uV)" for c in CH19] + ["A2(uV)", "ACC32(mg)", "ACC33(mg)", "ACC34(mg)"])
    t = d.timestamps.values; fs = (len(t)-1)/(t[-1]-t[0])
    X = d[[f"{c}(uV)" for c in CH19]].values.T - d["A2(uV)"].values[None, :]          # A2 reference
    X = signal.detrend(X, axis=-1, type="linear")                                       # remove DC offset (dry electrodes)
    X128 = signal.resample_poly(X, 32, 125, axis=-1)                                    # 500 -> 128 Hz
    acc = d[["ACC32(mg)", "ACC33(mg)", "ACC34(mg)"]].values
    motion = float(np.std(np.linalg.norm(acc - acc.mean(0), axis=1)))
    # bad channels (after the analysis filter)
    Xf = FU.preprocess(X128, lp=30); sd = Xf.std(1); med = np.median(sd)
    bad = [i for i in range(19) if sd[i] < 1e-6 or sd[i] > 5*med]
    X128c = interpolate_bad(X128, bad)
    x45 = FU.preprocess(X128c, lp=45)
    conf = FU.confound_features(X128c, x45)
    fr = [CH19.index(c) for c in ["Fp1", "Fp2"]]
    conf.update(blink_frontal_kurt=float(stats.kurtosis(x45[fr], axis=1).mean()), motion_acc_sd=motion,
                fs_est=float(fs), n_bad=len(bad), bad_channels=",".join(CH19[i] for i in bad))
    np.savez_compressed(os.path.join(OUT, f"{user}.npz"), X=X128c.astype(np.float32))
    return dict(user=user, **conf)

if __name__ == "__main__":
    man = pd.read_csv("balladeer/lvl1_manifest.csv")
    done = {os.path.basename(p)[:-4] for p in glob.glob(OUT + "/*.npz")}
    shard = int(sys.argv[1]) if len(sys.argv) > 1 else -1
    rows = []
    for k, (u, r) in enumerate(zip(man.user, man.rel)):
        if shard >= 0 and k % 2 != shard: continue
        if u in done or not os.path.exists(os.path.join(ROOT, u, "SlacklineLvl1", r)): continue
        rows.append(convert(u, r)); print(u, rows[-1]["n_bad"], round(rows[-1]["duration_s"], 1), flush=True)
    if rows:
        log = f"derived/balladeer_confounds{'' if shard < 0 else '_s'+str(shard)}.csv"
        old = pd.read_csv(log) if os.path.exists(log) else pd.DataFrame()
        pd.concat([old, pd.DataFrame(rows)]).drop_duplicates("user", keep="last").to_csv(log, index=False)
