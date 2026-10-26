"""Step 2: participant-level feature families F1-F5 (protocol.yaml).

F1/F3/F4 reuse the previous paper's code (extract_neural_features.py, functions copied
verbatim below) so they can be checked against its committed feature matrix.
F2 (FOOOF aperiodic + alpha peak) and F5 (window covariance) are new.
Feature extraction is label-free and per-participant, so it cannot leak.
"""
import sys, math, warnings, numpy as np, pandas as pd
from scipy import signal
sys.path.insert(0, "src")
from data_loader import load_dataset, CHANNELS
from fooof import FOOOF
warnings.filterwarnings("ignore")

import sys as _s, os as _o; _s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__))); from paths import NASRABADI_ROOT as RAW
FS = 128.0; K = 15; L = 512
BANDS4 = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30)}

# ---- copied from previous paper: src/features/extract_neural_features.py ----
def butter_band(lo, hi): return signal.butter(4, [lo/(FS/2), hi/(FS/2)], btype="band")
NT = signal.iirnotch(50, 30, FS)
def base_filter(x, hi):
    x = signal.detrend(x, axis=-1); x = signal.filtfilt(*NT, x, axis=-1)
    b, a = butter_band(1, hi); return signal.filtfilt(b, a, x, axis=-1)
def fixedK_windows(X):
    nfull = X.shape[1]//L; assert nfull >= K
    idx = np.round(np.linspace(0, nfull-1, K)).astype(int)
    return [X[:, j*L:(j+1)*L] for j in idx]
def welch(x, n=256):
    return signal.welch(x, FS, nperseg=min(n, x.shape[-1]), window="hann", axis=-1, detrend="constant")
def relpow(seg, bands):
    f, P = welch(seg); hi = max(h for _, h in bands.values()); m0 = (f >= 1) & (f <= hi)
    tot = np.trapezoid(P[..., m0], f[m0], axis=-1) + 1e-12
    return {b: np.trapezoid(P[..., (f >= lo) & (f < h)], f[(f >= lo) & (f < h)], axis=-1)/tot for b, (lo, h) in bands.items()}
def perm_entropy_multi(X, order=3):
    st = np.stack([X[:, i:X.shape[1]-(order-1)+i] for i in range(order)], 0)
    oi = np.argsort(st, axis=0); code = sum(oi[k]*(order**k) for k in range(order))
    ent = np.empty(19)
    for ch in range(19):
        _, c = np.unique(code[ch], return_counts=True); p = c/c.sum()
        ent[ch] = -np.sum(p*np.log(p))/np.log(math.factorial(order))
    return ent
def higuchi_multi(X, kmax=8):
    N = X.shape[1]; logL = []; logk = []
    for k in range(1, kmax+1):
        Lk = np.zeros(19)
        for m in range(k):
            idx = np.arange(m, N, k)
            if len(idx) < 2: continue
            Lk += np.sum(np.abs(np.diff(X[:, idx], axis=1)), axis=1)*(N-1)/(len(idx)-1)/k
        logL.append(np.log(Lk/k+1e-12)); logk.append(np.log(1.0/k))
    logL = np.array(logL); logk = np.array(logk)
    return np.array([np.polyfit(logk, logL[:, ch], 1)[0] for ch in range(19)])
def wpli_bandfeatures(Xfull):
    feats = {}
    for b, (lo, hi) in BANDS4.items():
        bb, aa = butter_band(lo, hi); Z = signal.hilbert(signal.filtfilt(bb, aa, Xfull, axis=-1), axis=-1)
        Ws = []
        for w in fixedK_windows(Z):
            im = (w[:, None, :]*np.conj(w[None, :, :])).imag
            W = np.abs(im.sum(-1))/(np.abs(im).sum(-1)+1e-12); np.fill_diagonal(W, 0); Ws.append(W)
        strength = np.mean(Ws, 0).sum(1)/18.0
        for ci, ch in enumerate(CHANNELS): feats[f"wpli_{b}_{ch}"] = strength[ci]
    return feats
# ---------------------------------------------------------------------------

def fooof_features(wins):
    """F2: FOOOF on the mean (over K windows) PSD per channel, 2-30 Hz."""
    f, P = welch(np.stack(wins, 0))            # (K,19,F)
    Pm = P.mean(0)
    out = {}
    for ci, ch in enumerate(CHANNELS):
        fm = FOOOF(peak_width_limits=(1, 8), max_n_peaks=4, aperiodic_mode="fixed", verbose=False)
        fm.fit(f, Pm[ci], [2, 30])
        off, exp = fm.aperiodic_params_
        pk = fm.peak_params_
        a = pk[(pk[:, 0] >= 7) & (pk[:, 0] <= 13)] if len(pk) else np.empty((0, 3))
        cf, pw = (a[np.argmax(a[:, 1]), 0], a[np.argmax(a[:, 1]), 1]) if len(a) else (np.nan, 0.0)
        out.update({f"ap_exp_{ch}": exp, f"ap_off_{ch}": off, f"alpha_cf_{ch}": cf, f"alpha_pw_{ch}": pw,
                    f"_fooof_r2_{ch}": fm.r_squared_})
    return out

def main():
    rows, covs = [], []
    for n, s in enumerate(load_dataset(RAW)):
        X30 = base_filter(s.data.astype(float), 30)
        wins = fixedK_windows(X30)
        feats = dict(subject=s.subject_id, label=s.label)
        rp = [relpow(w, BANDS4) for w in wins]
        for b in BANDS4:
            arr = np.array([r[b] for r in rp])
            for ci, ch in enumerate(CHANNELS): feats[f"rel_{b}_mean_{ch}"] = arr[:, ci].mean()
        pe = np.array([perm_entropy_multi(w) for w in wins]); hg = np.array([higuchi_multi(w) for w in wins])
        for ci, ch in enumerate(CHANNELS): feats[f"perm_{ch}"] = pe[:, ci].mean(); feats[f"higuchi_{ch}"] = hg[:, ci].mean()
        feats.update(wpli_bandfeatures(X30))
        feats.update(fooof_features(wins))
        C = np.mean([np.cov(w) for w in wins], 0)                      # F5: mean window covariance
        covs.append(C)
        rows.append(feats)
        if (n+1) % 20 == 0: print(f"  {n+1}/121", flush=True)
    D = pd.DataFrame(rows)
    covs = np.array(covs)
    for i in range(19):
        for j in range(19): D[f"cov_{i}_{j}"] = covs[:, i, j]
    D.to_csv("derived/features_nasrabadi.csv", index=False)
    print("saved", D.shape, "min eigenvalue of covs:", float(np.min(np.linalg.eigvalsh(covs))))

if __name__ == "__main__":
    main()
