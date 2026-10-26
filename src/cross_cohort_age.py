"""Exploratory: cross-cohort transfer restricted to BALLADEER participants aged 7-12 (Nasrabadi age range)."""
import sys, numpy as np, pandas as pd
sys.argv = sys.argv[:1]; sys.path.insert(0, "src")
import cross_cohort as CC
N0, B0, fam = CC.cohorts(); B0 = B0[(B0.age >= 7) & (B0.age <= 12)].reset_index(drop=True)
print("BALLADEER 7-12: n=%d ADHD=%d" % (len(B0), B0.label.sum()))
rows = []
for align in ["none", "cohort_norm"]:
    N, B = CC.prepare(N0, B0, fam, align)
    for name, src, tgt in [("Nasrabadi->BALLADEER7-12", N, B), ("BALLADEER7-12->Nasrabadi", B, N)]:
        aucs, scores = CC.transfer(src, tgt, fam)
        for k in ["E2_soft_vote"] + [f"M_{f}__lr" for f in fam] + ["M_F5_riemann__svm"]:
            lo, hi = CC.boot_ci(tgt.label.values, scores[k]); rows.append(dict(alignment=align, direction=name, model=k, AUC=aucs[k], CI_lo=lo, CI_hi=hi))
T = pd.DataFrame(rows); T.round(3).to_csv("results/cross_cohort_transfer_age7to12.csv", index=False); print(T.round(3).to_string(index=False))
