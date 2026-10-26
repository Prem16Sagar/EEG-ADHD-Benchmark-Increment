"""Scientific Reports display items from saved result files only (no new model fitting).
Outputs: figures/SR_Fig1..SR_Fig4 (.png 300 dpi + .pdf) and results/SR_display_numbers.csv (every plotted number + source).
Run from the paper2 folder: python src/figures_SciRep.py
"""
import sys, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
sys.path.insert(0, "src"); import rq1_A3 as M

BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#6b6b68", "#d9d8d4"
plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans", "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": INK, "axes.linewidth": 0.8, "hatch.linewidth": 0.8,
                     "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white"})
LOG = []

def boot_auc(y, s, nb=2000, seed=0):
    rng = np.random.default_rng(seed); i1, i0 = np.where(y == 1)[0], np.where(y == 0)[0]
    b = [roc_auc_score(y[i], s[i]) for i in (np.r_[rng.choice(i1, len(i1)), rng.choice(i0, len(i0))] for _ in range(nb))]
    return roc_auc_score(y, s), *np.percentile(b, [2.5, 97.5])

def clean(ax, grid_axis="x"):
    for s in ["top", "right"]: ax.spines[s].set_visible(False)
    ax.grid(axis=grid_axis, color=GRID, lw=0.5); ax.set_axisbelow(True)

def save(fig, name):
    fig.savefig(f"figures/{name}.png", dpi=300); fig.savefig(f"figures/{name}.pdf"); plt.close(fig)

# ---------------- Fig 1: age by label, ADHD split by recruitment group (RQ1 sample, n = 121)
D = M.load("primary")
ages = np.arange(6, 19)
ctl = [((D.label == 0) & (D.age == a)).sum() for a in ages]
g1 = [((D.label == 1) & (D.group == 1) & (D.age == a)).sum() for a in ages]
g2 = [((D.label == 1) & (D.group == 2) & (D.age == a)).sum() for a in ages]
fig, ax = plt.subplots(figsize=(6.8, 2.8)); w = 0.38
ax.bar(ages - w / 2 - 0.01, ctl, w, color=BLUE, label=f"No ADHD diagnosis (n = {sum(ctl)})")
ax.bar(ages + w / 2 + 0.01, g1, w, color=ORANGE, label=f"ADHD, clinic-referred group (n = {sum(g1)})")
ax.bar(ages + w / 2 + 0.01, g2, w, bottom=g1, color="white", edgecolor=ORANGE, hatch="////", lw=0.8,
       label=f"ADHD, recorded in control group (n = {sum(g2)})")
ax.set_xticks(ages); ax.set_xlabel("Age (years)"); ax.set_ylabel("Children"); clean(ax, "y")
ax.legend(frameon=False, loc="upper left", fontsize=7.5)
fig.tight_layout(); save(fig, "SR_Fig1_age_by_label")
for a, c, x, z in zip(ages, ctl, g1, g2): LOG.append(("Fig1", f"age {a}", c, x, z, "", "rq1_A3.load('primary')"))

# ---------------- Fig 2: non-neural benchmarks vs EEG models (AUC of repeat-averaged OOF scores, R = 10)
A2 = pd.read_csv("results/balladeer_primary_table_methods.csv").set_index("method")
NA = pd.read_csv("results/table_all_methods.csv").set_index("method")
A1 = pd.read_csv("results/a1_table_methods.csv").set_index("method")
S5 = pd.read_csv("results/A4/S5_nasrabadi_benchmark_decomposition.csv").set_index("benchmark")
def oof(path, col):
    o = pd.read_csv(path); return boot_auc(o.label.values, o[col].values)
def tab(T, m): return tuple(T.loc[m, ["AUC_avgscore", "CI_lo", "CI_hi"]])
bal = [("Non-neural", "Age + sex (6-learner vote)", tab(A2, "B0_demo_vote"), "pre"),
       ("Non-neural", "Acquisition proxies (vote)", tab(A2, "B0_acq_vote"), "pre"),
       ("Non-neural", "Age, sex, L1 behaviour (LR; RQ1 base)", oof("results/A3_rq1_primary_est_oof.csv", "score_base"), "pre"),
       ("Non-neural", "Age bands, sex, behaviour (LR)", oof("results/A4/S2_flexible_base_rq1_oof.csv", "score_base"), "post"),
       ("EEG only", "Spectral power (LR)", tab(A2, "B2_F1_spectral_lr"), "pre"),
       ("EEG only", "Best single model (inner-selected)", tab(A2, "B1_best_single"), "pre"),
       ("EEG only", "Soft vote, 30 models", tab(A2, "E2_soft_vote"), "pre"),
       ("EEG only", "Top-5 soft vote", tab(A2, "E5_topk_soft"), "pre"),
       ("Combined", "RQ1 base + EEG stack", oof("results/A3_rq1_primary_est_oof.csv", "score_full"), "pre"),
       ("Combined", "Age bands base + EEG stack", oof("results/A4/S2_flexible_base_rq1_oof.csv", "score_full"), "post")]
nas = [("Non-neural", "Recording duration only", tuple(S5.loc["duration only", ["AUC_avgscore", "CI_lo", "CI_hi"]]), "post"),
       ("Non-neural", "Recording-condition proxies", tuple(S5.loc["recording-condition proxies", ["AUC_avgscore", "CI_lo", "CI_hi"]]), "post"),
       ("Non-neural", "All six proxies (vote)", tab(NA, "B0_confound_vote"), "pre"),
       ("EEG only", "Riemannian covariance (LR)", tab(NA, "B2_F5_riemann_lr"), "pre"),
       ("EEG only", "Best single model (inner-selected)", tab(NA, "B1_best_single"), "pre"),
       ("EEG only", "Soft vote, 30 models", tab(NA, "E2_soft_vote"), "pre"),
       ("EEG only", "Top-5 soft vote", tab(NA, "E5_topk_soft"), "pre"),
       ("Combined", "Proxies + top-5 vote (stack)", tab(A1, "S_E5"), "pre")]
COL = {"Non-neural": ORANGE, "EEG only": BLUE, "Combined": INK}
fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.6), sharex=True)
for ax, rows, title in zip(axes, [bal, nas], ["BALLADEER, Slackline L1 (n = 121)", "Nasrabadi (n = 121)"]):
    y = np.arange(len(rows))[::-1]
    for yi, (grp, lab, (a, lo, hi), st) in zip(y, rows):
        c = COL[grp]; ax.hlines(yi, lo, hi, color=c, lw=1.4)
        ax.plot(a, yi, "o", ms=5.5, color=c if st == "pre" else "white", mec=c, mew=1.4)
        ax.text(1.03, yi, f"{a:.2f}", fontsize=7, va="center", ha="left", color=INK, transform=ax.get_yaxis_transform(), clip_on=False)
        LOG.append(("Fig2", f"{title} | {lab}", round(a, 3), round(lo, 3), round(hi, 3), st, ""))
    ax.axvline(0.5, color=MUTED, lw=0.8, ls=":")
    ax.set_yticks(y); ax.set_yticklabels([r[1] for r in rows], fontsize=7)
    ax.set_xlim(0.3, 1.0); ax.set_title(title, fontsize=8, loc="left"); ax.set_xlabel("AUC (95% CI)"); clean(ax)
    ax.tick_params(axis="y", length=0)
from matplotlib.lines import Line2D
h = [Line2D([], [], color=COL[k], marker="o", lw=1.4, label=k) for k in COL] + \
    [Line2D([], [], color=MUTED, marker="o", mfc="white", lw=0, label="Post hoc")]
fig.legend(handles=h, ncol=4, frameon=False, loc="lower center", fontsize=7.5)
fig.tight_layout(rect=(0, 0.06, 1, 1)); save(fig, "SR_Fig2_benchmarks")

# ---------------- Fig 3: (a) permutation null of the primary test; (b) RQ1 increment across analyses
T = pd.read_csv("results/A3_rq1_table.csv").set_index("run")
def a4(name):
    import json; s = json.load(open(f"results/A4/{name}_summary.json"))
    return s["dauc"], s["dauc_CI_lo"], s["dauc_CI_hi"]
def t(run): return tuple(T.loc[run, ["dAUC", "dAUC_CI_lo", "dAUC_CI_hi"]])
rows = [("Pre-specified", "Primary (diagnosed label)", t("A3_rq1_primary_est"), "pre"),
        ("Pre-specified", "Acquisition proxies in both models", t("A3_rq1sec_acq_R1"), "pre"),
        ("Pre-specified", "Over age + sex only (no behaviour)", t("A3_rq1sec_nobeh_R1"), "pre"),
        ("Pre-specified", "Clinically verified cases (S1 label)", t("A3_rq1_S1_est"), "pre"),
        ("Pre-specified", "Recruitment group label (S2)", t("A3_rq1_S2_est"), "pre"),
        ("Pre-specified", "Tuned regularisation", t("A3_rq1sec_tunedC_R10"), "pre"),
        ("Post hoc", "Ages 6-17 only", a4("S1_age_le17_rq1"), "post"),
        ("Post hoc", "Age-band base model", a4("S2_flexible_base_rq1"), "post"),
        ("Post hoc", "Level 6 behaviour and EEG", a4("S6_L6_behaviour_rq1"), "post"),
        ("Post hoc", "Aperiodic features re-fitted", a4("S4_F2refit_rq1"), "post"),
        ("Post hoc", "Spectral power only", t("A3_rq1sec_spectral_R10"), "post")]
null = pd.read_csv("results/A3_rq1_primary_test_null.csv").dauc_perm.values
obs = 0.045810729355033164   # results/A3_rq1_primary_test_summary.json (R = 3)
fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 3.3), gridspec_kw={"width_ratios": [1, 1.55]})
a.hist(null, bins=np.arange(-0.17, 0.13, 0.01), color=GRID, edgecolor="white", lw=0.5)
a.axvline(obs, color=BLUE, lw=1.6); a.text(obs + 0.005, a.get_ylim()[1] * 0.92, f"observed\n+{obs:.3f}\np = 0.048", color=INK, fontsize=7, va="top")
a.set_xlabel("ΔAUC under permuted EEG rows"); a.set_ylabel("Permutations (B = 1000)"); clean(a, "y")
a.set_title("a  Primary test (R = 3)", fontsize=8, loc="left")
y = np.arange(len(rows))[::-1]
for yi, (grp, lab, (d, lo, hi), st) in zip(y, rows):
    b.hlines(yi, lo, hi, color=BLUE, lw=1.4); b.plot(d, yi, "o", ms=5.5, color=BLUE if st == "pre" else "white", mec=BLUE, mew=1.4)
    b.text(0.205, yi, (f"{d:+.3f}" if round(d, 3) != 0 else "0.000"), fontsize=7, va="center", ha="right", color=INK)
    LOG.append(("Fig3b", lab, round(d, 3), round(lo, 3), round(hi, 3), st, ""))
b.axvline(0, color=MUTED, lw=0.8); b.axvline(0.05, color=MUTED, lw=0.8, ls="--")
b.axhline(y[5] - 0.5, color=GRID, lw=0.8)
b.set_yticks(y); b.set_yticklabels([r[1] for r in rows], fontsize=7); b.tick_params(axis="y", length=0)
b.set_xlim(-0.11, 0.21); b.set_ylim(y[-1] - 1.1, y[0] + 0.6); b.set_xlabel("ΔAUC from adding EEG (95% CI)"); clean(b)
b.set_title("b  Increment across analyses", fontsize=8, loc="left")
b.text(0.053, y[-1] - 0.75, "SESOI", fontsize=6.5, color=MUTED, va="center")
fig.tight_layout(); save(fig, "SR_Fig3_rq1")
LOG.append(("Fig3a", "null mean/sd", round(null.mean(), 3), round(null.std(ddof=1), 3), "", "", "A3_rq1_primary_test_null.csv"))

# ---------------- Fig 4: (a) cross-level stability; (b) level transfer of the EEG-only stack
R = pd.read_csv("results/A3_rq2a_reliability.csv").sort_values("ICC_A1_median")
names = {"F1_spectral": "Spectral", "F2_aperiodic": "Aperiodic", "F3_connectivity": "wPLI connectivity",
         "F4_complexity": "Complexity", "F5_riemann": "Riemannian covariance"}
L = pd.read_csv("results/A3_rq2b_transfer.csv"); L = L[L.model == "stack"]
fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 2.7), gridspec_kw={"width_ratios": [1.1, 1]})
yy = np.arange(len(R)); a.barh(yy, R.ICC_A1_median, color=BLUE, height=0.55)
a.hlines(yy, R.ICC_A1_median_CI_lo, R.ICC_A1_median_CI_hi, color=INK, lw=1)
for v in (0.5, 0.75): a.axvline(v, color=MUTED, lw=0.8, ls="--")
for yi, v, h in zip(yy, R.ICC_A1_median, R.ICC_A1_median_CI_hi):
    a.text(h + 0.015, yi, f"{v:.2f}", fontsize=7, va="center");
for _, r in R.iterrows(): LOG.append(("Fig4a", names[r.family], r.ICC_A1_median, r.ICC_A1_median_CI_lo, r.ICC_A1_median_CI_hi, "pre", ""))
a.set_yticks(yy); a.set_yticklabels([names[f] for f in R.family]); a.set_xlim(0, 1); a.tick_params(axis="y", length=0)
a.set_xlabel("Median ICC(A,1), levels 1, 6, 11 (n = 110)"); clean(a); a.set_title("a  Cross-level stability", fontsize=8, loc="left")
order = [("L1", "L1"), ("L1", "L6"), ("L1", "L11"), ("L6", "L6"), ("L6", "L1"), ("L11", "L11"), ("L11", "L1")]
yb = np.arange(len(order))[::-1]
for yi, (tr, te) in zip(yb, order):
    r = L[(L.train == tr) & (L.test == te)].iloc[0]; same = tr == te
    b.hlines(yi, r.CI_lo, r.CI_hi, color=BLUE, lw=1.4); b.plot(r.AUC, yi, "o" if same else "D", ms=5, color=BLUE, mec="white", mew=0.8)
    b.text(0.98, yi, f"{r.AUC:.2f}", fontsize=7, va="center", ha="right")
    LOG.append(("Fig4b", f"{tr}->{te}", r.AUC, r.CI_lo, r.CI_hi, "pre", f"p={r.p_perm_stack_uncorrected}"))
b.axvline(0.5, color=MUTED, lw=0.8, ls=":"); b.set_xlim(0.3, 1.0)
b.set_yticks(yb); b.set_yticklabels([f"{tr} → {te}" for tr, te in order]); b.tick_params(axis="y", length=0)
b.set_xlabel("AUC, EEG-only stack (95% CI)"); clean(b); b.set_title("b  Level transfer", fontsize=8, loc="left")
fig.tight_layout(); save(fig, "SR_Fig4_stability_transfer")

pd.DataFrame(LOG, columns=["item", "row", "v1", "v2", "v3", "status", "source"]).to_csv("results/SR_display_numbers.csv", index=False)
print("done", len(LOG))
