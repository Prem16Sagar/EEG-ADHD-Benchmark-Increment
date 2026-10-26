"""Paper 2 figures from saved result files (no new analysis). Single-series charts: slot-1 blue, gray reference lines."""
import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
BLUE, INK, MUTED, GRID, SURF = "#2a78d6", "#0b0b0b", "#6b6b68", "#d9d8d4", "#fcfcfb"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": INK,
                     "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF})

# Fig 1: EEG increment over non-EEG baselines (dAUC with 95% bootstrap CI)
r1 = pd.read_csv("results/A3_rq1_table.csv").set_index("run")
pa = pd.read_csv("results/POSTHOC_rq2_age_adjusted_transfer.csv")
r3 = pd.read_csv("results/A3_rq3_device_transfer.csv"); r3 = r3[r3.model.str.startswith("dAUC")]
rows = [("RQ1 primary: +EEG over age, sex, behaviour (L1)", *r1.loc["A3_rq1_primary_est", ["dAUC", "dAUC_CI_lo", "dAUC_CI_hi"]]),
        ("RQ1: same, base also has acquisition proxies (R=1)", *r1.loc["A3_rq1sec_acq_R1", ["dAUC", "dAUC_CI_lo", "dAUC_CI_hi"]]),
        ("RQ1: +EEG over age, sex only (L1, R=1)", *r1.loc["A3_rq1sec_nobeh_R1", ["dAUC", "dAUC_CI_lo", "dAUC_CI_hi"]]),
        ("RQ1 label S2 (recruitment group)", *r1.loc["A3_rq1_S2_est", ["dAUC", "dAUC_CI_lo", "dAUC_CI_hi"]])]
for _, r in pa.iterrows():
    rows.append((f"Level transfer {r.train}→{r.test} (post hoc)", r.dAUC, r.dAUC_CI_lo, r.dAUC_CI_hi))
for _, r in r3.iterrows():
    rows.append((f"Device {r.train.replace('_',' ')}→{r.test.replace('_',' ')}", r.AUC, r.CI_lo, r.CI_hi))
F = pd.DataFrame(rows, columns=["label", "d", "lo", "hi"])[::-1].reset_index(drop=True)
fig, ax = plt.subplots(figsize=(7.2, 0.32 * len(F) + 0.9))
ax.axvline(0, color=MUTED, lw=1); ax.axvline(0.05, color=GRID, lw=1, ls="--")
ax.hlines(F.index, F.lo, F.hi, color=BLUE, lw=2); ax.plot(F.d, F.index, "o", color=BLUE, ms=6, mec=SURF, mew=1.5)
ax.set_yticks(F.index); ax.set_yticklabels(F.label); ax.set_xlabel("ΔAUC from adding EEG (95% bootstrap CI)")
ax.text(0.05, len(F) - 0.4, "0.05 smallest effect of interest", color=MUTED, fontsize=7.5, ha="left", va="bottom")
for s in ["top", "right", "left"]: ax.spines[s].set_visible(False)
ax.tick_params(axis="y", length=0); ax.grid(axis="x", color=GRID, lw=0.5); ax.set_axisbelow(True)
fig.tight_layout(); fig.savefig("figures/fig1_eeg_increment.png", dpi=200); plt.close(fig)

# Fig 2: within-child cross-level reliability by family (median ICC(A,1), 95% CI of median)
R = pd.read_csv("results/A3_rq2a_reliability.csv").sort_values("ICC_A1_median")
names = {"F1_spectral": "Spectral", "F2_aperiodic": "Aperiodic", "F3_connectivity": "wPLI connectivity",
         "F4_complexity": "Complexity", "F5_riemann": "Riemannian covariance"}
fig, ax = plt.subplots(figsize=(6.0, 2.6)); y = np.arange(len(R))
ax.barh(y, R.ICC_A1_median, color=BLUE, height=0.55)
ax.hlines(y, R.ICC_A1_median_CI_lo, R.ICC_A1_median_CI_hi, color=INK, lw=1)
for v in (0.5, 0.75): ax.axvline(v, color=GRID, lw=1, ls="--")
for yi, v, h in zip(y, R.ICC_A1_median, R.ICC_A1_median_CI_hi): ax.text(h + 0.015, yi, f"{v:.2f}", color=INK, fontsize=8, va="center")
ax.set_yticks(y); ax.set_yticklabels([names[f] for f in R.family]); ax.set_xlim(0, 1)
ax.set_xlabel("Median ICC(A,1) across Slackline L1/L6/L11 (n = 110)")
for s in ["top", "right", "left"]: ax.spines[s].set_visible(False)
ax.tick_params(axis="y", length=0); fig.tight_layout(); fig.savefig("figures/fig2_reliability.png", dpi=200); plt.close(fig)
print("saved")
