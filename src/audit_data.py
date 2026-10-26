"""Step 1: data audit of the raw Nasrabadi release against the previous paper."""
import sys, hashlib, numpy as np, pandas as pd
sys.path.insert(0, "src")
from data_loader import load_dataset, CHANNELS
import sys as _s, os as _o; _s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__))); from paths import NASRABADI_ROOT as RAW
import sys as _s, os as _o; _s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__))); from paths import PREV_CONF as PREV
subs = load_dataset(RAW)
rows = []
for s in subs:
    rows.append(dict(subject=s.subject_id, label=s.label, n_samples=s.n_samples, duration_s=s.duration_s,
                     src_dtype=s.source_dtype, amp_sd=float(np.std(s.data)), max_abs=float(np.abs(s.data).max()),
                     flat_ch=int((s.data.std(1) < 1e-6).sum()),
                     sha256=hashlib.sha256(open(s.path,'rb').read()).hexdigest()))
d = pd.DataFrame(rows)
d.to_csv("derived/data_audit.csv", index=False)
prev = pd.read_csv(PREV)[["subject","label","duration_s","n_samples"]]
m = d.merge(prev, on="subject", suffixes=("", "_prev"), how="outer", indicator=True)
print("subjects:", len(d), "ADHD:", int(d.label.sum()), "Control:", int((d.label==0).sum()))
print("merge:", m["_merge"].value_counts().to_dict())
print("label mismatches:", int((m.label != m.label_prev).sum()))
print("n_samples mismatches:", int((m.n_samples != m.n_samples_prev).sum()))
print("duration range: %.1f-%.1f s" % (d.duration_s.min(), d.duration_s.max()))
print(d.groupby("label").duration_s.agg(["mean","std","min","max"]).round(1))
print("dtypes:", d.src_dtype.value_counts().to_dict(), "flat channels total:", d.flat_ch.sum())
print("min windows (4 s):", int((d.n_samples//512).min()))
print("channels:", CHANNELS)
