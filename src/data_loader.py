"""
data_loader.py
==============
Loader for the Nasrabadi "EEG data for ADHD / Control children" dataset
(IEEE DataPort, DOI 10.21227/rzfh-zn36).

Verified against the actual files (audit run 2026-09-26):
  * 121 subjects: 61 ADHD + 60 Control, one MATLAB v5 .mat per subject.
  * Each .mat contains exactly ONE variable, named like the file
    (e.g. v1p.mat -> variable "v1p"). No event/marker variable exists,
    so the task's trial structure CANNOT be reconstructed from the files.
  * On-disk orientation is (n_samples, 19). dtype is mixed across subjects
    (89 int16, 32 float64); we cast everything to float64 on load.
  * Sampling frequency: 128 Hz (from dataset documentation; not stored in file).
  * Channel order is NOT stored in the .mat. It is fixed by the dataset's
    Channel_Labels.docx / Standard-10-20-Cap19new.ced and hard-coded below.
    NOTE: this order differs from the order printed in most papers -- use THIS one.
  * Class label is NOT in the filename (the "p" suffix appears in both groups);
    it is determined solely by the ADHD_* vs Control_* parent folder.

Data integrity confirmed at load time in the audit: 0 NaNs, 0 Infs,
0 flat/near-flat channels across all 121 recordings.

Returned signal orientation
---------------------------
load_subject() returns data as (n_channels, n_samples) = (19, T), which is the
MNE-Python convention (n_channels, n_times). Set orient="samples_channels" to
get the raw on-disk (T, 19) instead.

Typical usage
-------------
    from data_loader import load_dataset, CHANNELS, SFREQ
    subjects = load_dataset("/path/to/My dataset")
    X0 = subjects[0].data            # (19, T0), float64, microvolt-scale
    y  = [s.label for s in subjects] # 1 = ADHD, 0 = Control
"""

from __future__ import annotations

import os
import glob
import re
from dataclasses import dataclass, field

import numpy as np
import scipy.io as sio

# --- Fixed dataset constants (see module docstring for provenance) -----------
SFREQ: float = 128.0

# Column order of the 19 channels INSIDE the .mat matrices, taken from the
# dataset's own Channel_Labels.docx / .ced (NOT from any paper). Modern
# nomenclature; equivalences: T7=T3, T8=T4, P7=T5, P8=T6.
CHANNELS: list[str] = [
    "Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
    "F7", "F8", "T7", "T8", "P7", "P8", "Fz", "Cz", "Pz",
]
N_CHANNELS: int = len(CHANNELS)  # 19

# Sub-paths (relative to the dataset root) that hold the per-subject .mat files,
# with the class each folder implies. The release doubly-nests each part folder.
_GROUP_DIRS: dict[str, int] = {
    "ADHD_part1/ADHD_part1": 1,
    "ADHD_part2/ADHD_part2": 1,
    "Control_part1/Control_part1": 0,
    "Control_part2/Control_part2": 0,
}
_LABEL_NAME = {1: "ADHD", 0: "Control"}


@dataclass
class Subject:
    """One subject's recording plus metadata."""
    subject_id: str            # e.g. "v1p" or "v177"
    label: int                 # 1 = ADHD, 0 = Control
    group: str                 # "ADHD" | "Control"
    data: np.ndarray           # float64, (n_channels, n_samples) by default
    sfreq: float = SFREQ
    channels: list[str] = field(default_factory=lambda: list(CHANNELS))
    source_dtype: str = ""     # original on-disk dtype ("int16"/"float64")
    path: str = ""

    @property
    def n_samples(self) -> int:
        return self.data.shape[-1]

    @property
    def duration_s(self) -> float:
        return self.n_samples / self.sfreq


def load_subject(path: str, orient: str = "channels_samples",
                 check: bool = True) -> Subject:
    """Load a single subject .mat file.

    Parameters
    ----------
    path : str
        Path to a subject file, e.g. ".../ADHD_part1/ADHD_part1/v1p.mat".
        The class label is inferred from whether "ADHD" or "Control" appears
        in the path (raises if neither/both do).
    orient : {"channels_samples", "samples_channels"}
        Output orientation. Default (19, T) matches MNE. "samples_channels"
        returns the raw on-disk (T, 19).
    check : bool
        If True, assert 19 channels and warn (via ValueError) on NaN/Inf.

    Returns
    -------
    Subject
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    up = path.replace("\\", "/")
    is_adhd = "/ADHD_" in up or "ADHD_part" in up
    is_ctrl = "/Control_" in up or "Control_part" in up
    if is_adhd == is_ctrl:  # neither or both -> ambiguous
        raise ValueError(f"Cannot infer class from path (need ADHD_/Control_): {path}")
    label = 1 if is_adhd else 0

    subject_id = os.path.splitext(os.path.basename(path))[0]

    mat = sio.loadmat(path)
    user_vars = [k for k in mat if not k.startswith("__")]
    if len(user_vars) != 1:
        raise ValueError(f"{path}: expected exactly 1 variable, found {user_vars}")
    var = user_vars[0]
    if var != subject_id:
        # Not fatal, but worth surfacing: naming assumption violated.
        raise ValueError(f"{path}: variable '{var}' != filename '{subject_id}'")

    src_dtype = str(np.asarray(mat[var]).dtype)
    X = np.asarray(mat[var]).astype(np.float64)  # cast mixed int16/float64 -> float64

    # Normalize to (n_samples, n_channels) first, using the fact that 19 is a dim.
    if X.ndim != 2:
        raise ValueError(f"{path}: expected 2-D matrix, got shape {X.shape}")
    if X.shape[1] == N_CHANNELS:
        pass                      # already (samples, channels)
    elif X.shape[0] == N_CHANNELS:
        X = X.T                   # (channels, samples) -> (samples, channels)
    else:
        raise ValueError(f"{path}: neither dim is {N_CHANNELS}; shape={X.shape}")

    if check:
        if X.shape[1] != N_CHANNELS:
            raise ValueError(f"{path}: got {X.shape[1]} channels, expected {N_CHANNELS}")
        if np.isnan(X).any() or np.isinf(X).any():
            raise ValueError(f"{path}: contains NaN/Inf")

    if orient == "channels_samples":
        data = np.ascontiguousarray(X.T)     # (19, T)
    elif orient == "samples_channels":
        data = np.ascontiguousarray(X)       # (T, 19)
    else:
        raise ValueError("orient must be 'channels_samples' or 'samples_channels'")

    return Subject(
        subject_id=subject_id, label=label, group=_LABEL_NAME[label],
        data=data, sfreq=SFREQ, channels=list(CHANNELS),
        source_dtype=src_dtype, path=path,
    )


def iter_subject_paths(root: str):
    """Yield (path, label) for every subject file under the dataset root."""
    found = False
    for subdir, label in _GROUP_DIRS.items():
        for p in sorted(glob.glob(os.path.join(root, subdir, "*.mat"))):
            found = True
            yield p, label
    if not found:
        raise FileNotFoundError(
            f"No .mat files found under {root!r}. Expected sub-folders like "
            f"'ADHD_part1/ADHD_part1/'. Point 'root' at the 'My dataset' folder."
        )


def load_dataset(root: str, orient: str = "channels_samples",
                 check: bool = True) -> list[Subject]:
    """Load all 121 subjects. Recordings are variable length, so this returns
    a list of Subject objects (they cannot be stacked into one array without
    segmentation/padding, which is a downstream decision)."""
    subjects = [load_subject(p, orient=orient, check=check)
                for p, _ in iter_subject_paths(root)]
    n_adhd = sum(s.label == 1 for s in subjects)
    if len(subjects) != 121 or n_adhd != 61:
        # Surface any mismatch loudly rather than silently proceeding.
        raise ValueError(
            f"Loaded {len(subjects)} subjects ({n_adhd} ADHD); expected 121 (61 ADHD)."
        )
    return subjects


def to_mne_raw(subject: Subject):
    """Optional: build an MNE RawArray with the correct montage.
    Requires mne installed. Data are assumed to be in microvolts and are
    converted to volts for MNE (which expects SI units)."""
    import mne
    data = subject.data if subject.data.shape[0] == N_CHANNELS else subject.data.T
    info = mne.create_info(ch_names=list(subject.channels),
                           sfreq=subject.sfreq, ch_types="eeg")
    raw = mne.io.RawArray(data * 1e-6, info, verbose="ERROR")  # uV -> V
    raw.set_montage("standard_1020", match_case=False, on_missing="warn")
    return raw


if __name__ == "__main__":
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    subs = load_dataset(root)
    print(f"Loaded {len(subs)} subjects "
          f"({sum(s.label==1 for s in subs)} ADHD / {sum(s.label==0 for s in subs)} Control)")
    s = subs[0]
    print(f"example: {s.subject_id} [{s.group}] data{s.data.shape} "
          f"dtype={s.data.dtype} src={s.source_dtype} dur={s.duration_s:.1f}s")
