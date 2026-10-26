"""feat_utils.py — preprocessing, fixed-K segmentation, and confound-robust
feature extraction for the ADHD/Control signal-existence experiment.

All steps are subject-independent and stateless (filters fixed a priori);
per-subject ICA is fit on that subject's own data only. Nothing here uses
labels or other subjects, so feature extraction cannot leak.
"""
import numpy as np
from scipy import signal, stats
import antropy as ant
from data_loader import CHANNELS

FS = 128.0
L = 512          # 4 s window
K = 15           # fixed windows per subject
NYQ = FS / 2
TEMP = ["T7", "T8"]

# ---- fixed filters -------------------------------------------------------
_hp = signal.butter(4, 1/NYQ, btype="high")
_notch = signal.iirnotch(50, 30, FS)
_lp30 = signal.butter(4, 30/NYQ, btype="low")
_lp45 = signal.butter(4, 45/NYQ, btype="low")

def preprocess(x, lp=30):
    """x: (19, T) float -> filtered. lp in {30,45}."""
    x = signal.detrend(x, axis=-1, type="constant")
    x = signal.filtfilt(*_notch, x, axis=-1)
    x = signal.filtfilt(*_hp, x, axis=-1)
    b, a = (_lp30 if lp == 30 else _lp45)
    return signal.filtfilt(b, a, x, axis=-1)

def fixed_k_windows(x, k=K, L=L, mode="even"):
    """x:(19,T) -> (k,19,L). Deterministic even sampling of non-overlapping windows."""
    n = x.shape[-1] // L
    if n < k:
        raise ValueError(f"only {n} windows available (<{k})")
    if mode == "even":
        idx = np.round(np.linspace(0, n-1, k)).astype(int)
    else:  # random (for stability check)
        rng = np.random.default_rng(1); idx = np.sort(rng.choice(n, k, replace=False))
    return np.stack([x[:, i*L:(i+1)*L] for i in idx])   # (k,19,L)

# ---- conservative ICA blink removal (per subject) ------------------------
from sklearn.decomposition import FastICA
FRONTAL = ["Fp1","Fp2","F7","F8","F3","F4","Fz"]; PREF = ["Fp1","Fp2"]
def ica_blink_clean(xf, conservative=True):
    """xf:(19,T) filtered -> blink-cleaned (19,T). Removes only high-confidence
    frontal, low-frequency, high-kurtosis (blink) components."""
    try:
        ica = FastICA(n_components=19, whiten="unit-variance", max_iter=500,
                      tol=1e-4, random_state=0)
        S = ica.fit_transform(xf.T); A = ica.mixing_
    except Exception:
        return xf.copy()
    keep = np.ones(19, bool)
    fi = [CHANNELS.index(c) for c in FRONTAL]; pi = [CHANNELS.index(c) for c in PREF]
    for i in range(19):
        s = S[:, i]; a = np.abs(A[:, i])
        f, P = signal.welch(s, FS, nperseg=256)
        lowf = P[(f>=1)&(f<5)].sum()/ (P[(f>=1)&(f<45)].sum()+1e-12)
        frontal = a[fi].sum()/a.sum(); pref = a[pi].sum()/a.sum()
        k = stats.kurtosis(s, fisher=True)
        blink = (frontal>0.70 and pref>0.45 and lowf>0.5 and k>5) if conservative \
                else (frontal>0.60 and pref>0.35 and lowf>0.5 and k>3)
        if blink: keep[i] = False
    if keep.all():
        return xf.copy()
    Sm = S.copy(); Sm[:, ~keep] = 0
    return (Sm @ A.T + ica.mean_).T

# ---- spectral ------------------------------------------------------------
SPEC_BANDS = {"delta":(1,4),"theta":(4,8),"alpha":(8,13),"beta":(13,30)}
def _welch(x, nperseg=256):
    f, P = signal.welch(x, FS, nperseg=min(nperseg, x.shape[-1]), window="hann",
                        axis=-1, detrend="constant"); return f, P
def spectral_window(w, extra_gamma=False):
    """w:(19,L) -> dict band-> (19,) relative power. total = 1-45 if gamma else 1-30."""
    f, P = _welch(w)
    top = 45 if extra_gamma else 30
    tot = P[..., (f>=1)&(f<top)].sum(-1)+1e-12
    out = {}
    bands = dict(SPEC_BANDS)
    if extra_gamma: bands["gamma"] = (30, 45)
    for b,(lo,hi) in bands.items():
        out[b] = P[..., (f>=lo)&(f<hi)].sum(-1)/tot
    return out

# ---- complexity ----------------------------------------------------------
def complexity_window(w, measures):
    """w:(19,L) -> dict measure-> (19,)."""
    out = {m: np.zeros(19) for m in measures}
    for ch in range(19):
        x = np.ascontiguousarray(w[ch], dtype=np.float64)
        if "permen" in measures: out["permen"][ch] = ant.perm_entropy(x, order=3, normalize=True)
        if "higuchi" in measures: out["higuchi"][ch] = ant.higuchi_fd(x)
        if "lzc" in measures:
            seq = (x >= np.median(x)).astype(int); out["lzc"][ch] = ant.lziv_complexity(seq, normalize=True)
        if "sampen" in measures: out["sampen"][ch] = ant.sample_entropy(x, order=2)
        if "specen" in measures: out["specen"][ch] = ant.spectral_entropy(x, FS, method="welch", nperseg=256, normalize=True)
    for m in out: out[m] = np.nan_to_num(out[m], nan=0.0, posinf=0.0, neginf=0.0)
    return out

# ---- connectivity (phase-based, volume-conduction robust) -----------------
CONN_BANDS = {"theta":(4,8),"alpha":(8,13)}
def _analytic(w, lo, hi):
    b, a = signal.butter(4, [lo/NYQ, hi/NYQ], btype="band")
    wf = signal.filtfilt(b, a, w, axis=-1)
    return signal.hilbert(wf, axis=-1)               # (19,L) complex
def connectivity_window(w):
    """w:(19,L) -> dict band-> dict measure-> (19,19) matrix (wpli, pli, icoh)."""
    res = {}
    for band,(lo,hi) in CONN_BANDS.items():
        z = _analytic(w, lo, hi)                      # (19,L)
        # cross-spectrum S_xy = z_x conj(z_y): (19,19,L)
        Sxy = z[:, None, :] * np.conj(z[None, :, :])
        im = Sxy.imag
        wpli = np.abs(im.mean(-1)) / (np.abs(im).mean(-1) + 1e-12)
        pli = np.abs(np.sign(im).mean(-1))
        pw = (np.abs(z)**2).mean(-1)                  # (19,)
        icoh = np.abs(im.mean(-1)) / (np.sqrt(np.outer(pw, pw)) + 1e-12)
        for M in (wpli, pli, icoh): np.fill_diagonal(M, 0.0)
        res[band] = {"wpli": wpli, "pli": pli, "icoh": icoh}
    return res

def graph_feats_from_matrix(W):
    """W:(19,19) weighted -> node strength (19) + global mean strength (1)."""
    strength = W.sum(1)                                # weighted degree per node
    glob = W[np.triu_indices(19, 1)].mean()
    return strength, glob

# ---- confound features (per subject) -------------------------------------
def confound_features(x_raw, x_lp45):
    """duration/quality features measured as in Phase 2/3 (EMG/transient on 45Hz,
    line/drift on raw). Returns dict."""
    f_r, P_r = _welch(x_raw)
    line = (P_r[..., (f_r>=49)&(f_r<51)].sum(-1) / (P_r[..., (f_r>=1)&(f_r<45)].sum(-1)+1e-12)).mean()
    drift = (P_r[..., (f_r>=0)&(f_r<1)].sum(-1) / (P_r[..., (f_r>=1)&(f_r<45)].sum(-1)+1e-12)).mean()
    f, P = _welch(x_lp45)
    tot = P[..., (f>=1)&(f<45)].sum(-1)+1e-12
    ti = [CHANNELS.index(c) for c in TEMP]
    emg = (P[..., (f>=30)&(f<45)].sum(-1)/tot)[ti].mean()
    med = np.median(x_lp45, axis=-1, keepdims=True); mad = np.median(np.abs(x_lp45-med), axis=-1)+1e-12
    transient = (np.abs(x_lp45-med) > 5*mad[:, None]).mean()
    return dict(duration_s=x_raw.shape[-1]/FS, n_samples=float(x_raw.shape[-1]),
                emg=emg, line=line, transient=transient, drift=drift)
