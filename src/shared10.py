"""A3 RQ3/RQ4/H_wPLI data: features on the 10 channels shared by Nasrabadi, BALLADEER-CGX and BALLADEER-Emotiv
(F7, F3, T7, P7, O1, O2, P8, T8, F4, F8). Same feature definitions as F1-F5 (extract_features.py), generalised
to 10 channels (wPLI node strength divided by 9; covariance 10x10 with the A2 ridge 1e-3*trace/10).
Native reference of each device is kept (as in A2 cross-cohort). Label-free.

Emotiv (A3 section 8 data): EPOC X / EPOC+ AttentionRobotsDesktop recordings, 14 ch, 128 Hz, native CMS/DRL reference.
Rules fixed here before any Emotiv result: bad channel = SD after 1-30 Hz filter < 1e-6 or > 5 x median of the
14 channels, interpolated (MNE spherical spline, standard_1020); recording excluded if > 4 of 14 bad,
duration < 180 s (fixed-K needs 60 s), EEG.Interpolated (packet-loss) fraction > 10 %, or effective sampling rate
more than 5 % below nominal (dropped samples). 256-Hz recordings are downsampled to 128 Hz (resample_poly 1/2).
(Rate rules finalised after a label-free first pass flagged 3 files at 256 Hz and 4 with dropped samples.)

Usage: python src/shared10.py emotiv | nasrabadi | cgx <level> | all
"""
import sys, os, glob, json, math, warnings, numpy as np, pandas as pd
from scipy import signal
sys.path.insert(0, "src")
import extract_features as EF
from fooof import FOOOF
warnings.filterwarnings("ignore")
CH10 = ["F7", "F3", "T7", "P7", "O1", "O2", "P8", "T8", "F4", "F8"]
EMO14 = ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"]
CH19 = ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2", "F7", "F8", "T7", "T8", "P7", "P8", "Fz", "Cz", "Pz"]
import sys as _s, os as _o; _s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__))); from paths import BALLADEER_ROOT as ROOT

def perm_entropy(X, order=3):
    st = np.stack([X[:, i:X.shape[1] - (order - 1) + i] for i in range(order)], 0)
    oi = np.argsort(st, axis=0); code = sum(oi[k] * (order ** k) for k in range(order)); out = []
    for ch in range(X.shape[0]):
        _, c = np.unique(code[ch], return_counts=True); p = c / c.sum(); out.append(-np.sum(p * np.log(p)) / np.log(math.factorial(order)))
    return np.array(out)

def higuchi(X, kmax=8):
    N = X.shape[1]; n = X.shape[0]; logL = []; logk = []
    for k in range(1, kmax + 1):
        Lk = np.zeros(n)
        for m in range(k):
            idx = np.arange(m, N, k)
            if len(idx) < 2: continue
            Lk += np.sum(np.abs(np.diff(X[:, idx], axis=1)), axis=1) * (N - 1) / (len(idx) - 1) / k
        logL.append(np.log(Lk / k + 1e-12)); logk.append(np.log(1.0 / k))
    logL = np.array(logL); logk = np.array(logk)
    return np.array([np.polyfit(logk, logL[:, c], 1)[0] for c in range(n)])

def features10(X):
    """X: 10 x T at 128 Hz (CH10 order), microvolts."""
    X30 = EF.base_filter(X.astype(float), 30); wins = EF.fixedK_windows(X30); n = len(CH10); f = {}
    rp = [EF.relpow(w, EF.BANDS4) for w in wins]
    for b in EF.BANDS4:
        arr = np.array([r[b] for r in rp])
        for i, ch in enumerate(CH10): f[f"rel_{b}_mean_{ch}"] = arr[:, i].mean()
    pe = np.array([perm_entropy(w) for w in wins]); hg = np.array([higuchi(w) for w in wins])
    for i, ch in enumerate(CH10): f[f"perm_{ch}"] = pe[:, i].mean(); f[f"higuchi_{ch}"] = hg[:, i].mean()
    for b, (lo, hi) in EF.BANDS4.items():
        bb, aa = EF.butter_band(lo, hi); Z = signal.hilbert(signal.filtfilt(bb, aa, X30, axis=-1), axis=-1); Ws = []
        for w in EF.fixedK_windows(Z):
            im = (w[:, None, :] * np.conj(w[None, :, :])).imag
            W = np.abs(im.sum(-1)) / (np.abs(im).sum(-1) + 1e-12); np.fill_diagonal(W, 0); Ws.append(W)
        s = np.mean(Ws, 0).sum(1) / (n - 1)
        for i, ch in enumerate(CH10): f[f"wpli_{b}_{ch}"] = s[i]
    fr, P = EF.welch(np.stack(wins, 0)); Pm = P.mean(0)
    for i, ch in enumerate(CH10):
        fm = FOOOF(peak_width_limits=(1, 8), max_n_peaks=4, aperiodic_mode="fixed", verbose=False); fm.fit(fr, Pm[i], [2, 30])
        off, ex = fm.aperiodic_params_; pk = fm.peak_params_
        a = pk[(pk[:, 0] >= 7) & (pk[:, 0] <= 13)] if len(pk) else np.empty((0, 3))
        cf, pw = (a[np.argmax(a[:, 1]), 0], a[np.argmax(a[:, 1]), 1]) if len(a) else (np.nan, 0.0)
        f.update({f"ap_exp_{ch}": ex, f"ap_off_{ch}": off, f"alpha_cf_{ch}": cf, f"alpha_pw_{ch}": pw})
    C = np.mean([np.cov(w) for w in wins], 0); C = C + 1e-3 * np.trace(C) / n * np.eye(n)
    for i in range(n):
        for j in range(n): f[f"cov_{i}_{j}"] = C[i, j]
    return f

FAM10 = {"F1_spectral": [f"rel_{b}_mean_{c}" for b in EF.BANDS4 for c in CH10],
         "F2_aperiodic": [f"{p}_{c}" for p in ["ap_exp", "ap_off", "alpha_cf", "alpha_pw"] for c in CH10],
         "F3_connectivity": [f"wpli_{b}_{c}" for b in EF.BANDS4 for c in CH10],
         "F4_complexity": [f"{p}_{c}" for p in ["perm", "higuchi"] for c in CH10],
         "F5_riemann": [f"cov_{i}_{j}" for i in range(10) for j in range(10)]}

# ------------------------------------------------------------------ sources
def emotiv_one(path):
    d = pd.read_csv(path, skiprows=1, usecols=lambda c: c in ["Timestamp", "EEG.Interpolated"] + [f"EEG.{c}" for c in EMO14])
    d = d.dropna(subset=[f"EEG.{c}" for c in EMO14])
    t = d.Timestamp.values; dur = t[-1] - t[0]; fs = (len(t) - 1) / dur
    interp = float((d["EEG.Interpolated"].fillna(0) != 0).mean())
    X = d[[f"EEG.{c}" for c in EMO14]].values.T.astype(float)
    nominal = 256.0 if fs > 192 else 128.0                       # a few EPOC+ files were recorded at 256 Hz
    if nominal == 256.0: X = signal.resample_poly(X, 1, 2, axis=-1)
    X = signal.detrend(X, axis=-1, type="linear")
    Xf = EF.base_filter(X, 30); sd = Xf.std(1); med = np.median(sd)
    bad = [i for i in range(14) if sd[i] < 1e-6 or sd[i] > 5 * med]
    if 0 < len(bad) <= 4:
        import mne
        info = mne.create_info(EMO14, 128.0, "eeg"); raw = mne.io.RawArray(X * 1e-6, info, verbose="ERROR")
        raw.set_montage("standard_1020", verbose="ERROR"); raw.info["bads"] = [EMO14[i] for i in bad]
        raw.interpolate_bads(reset_bads=True, verbose="ERROR"); X = raw.get_data() * 1e6
    meta = dict(duration_s=dur, fs_est=fs, interp_frac=interp, n_bad=len(bad), bad_channels=",".join(EMO14[i] for i in bad),
                headset="EPOCPLUS" if "EPOCPLUS" in path else "EPOCX", nominal_fs=nominal)
    ok = len(bad) <= 4 and dur >= 180 and interp <= 0.10 and abs(fs / nominal - 1) <= 0.05   # sample-loss guard: within 5 % of nominal rate
    return X[[EMO14.index(c) for c in CH10]], meta, ok

def run_emotiv():
    rows = []
    for p in sorted(glob.glob(ROOT + "/UB*/AttentionRobotsDesktop/*/*EPOC*.csv")):
        u = os.path.basename(p)[:6]
        try:
            X, meta, ok = emotiv_one(p)
            f = features10(X) if ok else {}
        except Exception as e:
            meta, ok, f = dict(error=repr(e)), False, {}
        rows.append(dict(subject=u, usable=ok, **meta, **f)); print(u, ok, {k: meta.get(k) for k in ["n_bad", "interp_frac"]}, flush=True)
    for p in sorted(glob.glob(ROOT + "/UB*/AttentionRobotsDesktop/*/*GAME_DATA*.csv")):
        u = os.path.basename(p)[:6]; g = pd.read_csv(p).iloc[0].to_dict(); g.pop("maxBots", None)
        for r in rows:
            if r["subject"] == u: r.update({f"AR_{k}": v for k, v in g.items()})
    D = pd.DataFrame(rows); D.to_csv("derived/features10_emotiv_AR.csv", index=False)
    print("emotiv saved", D.shape, "usable", int(D.usable.sum()))

def run_nasrabadi():
    from data_loader import load_dataset
    rows = []
    for s in load_dataset(EF.RAW):
        X = s.data.astype(float)[[CH19.index(c) for c in CH10]]
        rows.append(dict(subject=s.subject_id, label=s.label, **features10(X)))
    D = pd.DataFrame(rows); D.to_csv("derived/features10_nasrabadi.csv", index=False); print("nasrabadi saved", D.shape)

def run_cgx(level):
    d = "derived/balladeer_npz" if level == 1 else f"derived/balladeer_npz_L{level}"; rows = []
    for f in sorted(glob.glob(d + "/*.npz")):
        X = np.load(f)["X"].astype(float)[[CH19.index(c) for c in CH10]]
        rows.append(dict(subject=os.path.basename(f)[:-4], **features10(X)))
    D = pd.DataFrame(rows); D.to_csv(f"derived/features10_cgx_L{level}.csv", index=False); print("cgx", level, "saved", D.shape)

if __name__ == "__main__":
    a = sys.argv[1]
    if a in ("emotiv", "all"): run_emotiv()
    if a in ("nasrabadi", "all"): run_nasrabadi()
    if a == "cgx": run_cgx(int(sys.argv[2]))
    if a == "all":
        for L in (1, 6, 11): run_cgx(L)
