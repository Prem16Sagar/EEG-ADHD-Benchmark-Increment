"""Simulation only (no real data): sample size needed for 80% power to detect a true ΔAUC of 0.05.

Added in release v1.0.1 (post hoc; not part of any locked protocol). It extends sim_power_A3.py,
which gave power 0.34-0.50 at the study's size (79 ADHD vs 42 controls), to larger samples.
The data-generating model, effect size, weight factors and test are unchanged:
  - binormal latent model; base AUC 0.72, full AUC 0.77 (true ΔAUC 0.05);
  - EEG weight factor 1.0 (oracle weights) or 0.5 (half the EEG signal recovered);
  - paired two-sided DeLong test at alpha = 0.05 (delong() from analyse.py); 2,000 simulations per cell.
Two class ratios: the study's 79:42 ratio and a balanced 1:1 design.
Because the model ignores the loss from estimating weights inside nested cross-validation, the sample
sizes are optimistic (lower bounds) for a study analysed like the present one.
Run from the repository root: python src/sim_samplesize_A3.py
"""
import sys, numpy as np, pandas as pd
from scipy.stats import norm
from joblib import Parallel, delayed
sys.path.insert(0, "src"); from analyse import delong

N_SIM = 2000
BASE, FULL = 0.72, 0.77
MU_B = np.sqrt(2) * norm.ppf(BASE)
MU_E = np.sqrt((np.sqrt(2) * norm.ppf(FULL)) ** 2 - MU_B ** 2)
RATIOS = {"79:42 (as in the study)": 79 / 121, "1:1 (balanced)": 0.5}
N_GRID = [121, 150, 200, 250, 300, 350, 400, 450, 500, 600, 700, 800, 900, 1000]

def cell(n_total, ratio_name, frac_pos, shrink):
    n1 = int(round(n_total * frac_pos)); n0 = n_total - n1
    y = np.r_[np.ones(n1), np.zeros(n0)].astype(int)
    rng = np.random.default_rng([42, n_total, int(frac_pos * 1000), int(shrink * 10)])
    d, p = np.empty(N_SIM), np.empty(N_SIM)
    for i in range(N_SIM):
        B = rng.normal(MU_B * y, 1); E = rng.normal(MU_E * y, 1)
        full = MU_B * B + shrink * MU_E * E
        a, b, z, pv = delong(y, full, B); d[i] = a - b; p[i] = pv
    return dict(class_ratio=ratio_name, n_total=n_total, n_adhd=n1, n_control=n0, eeg_weight_factor=shrink,
                mean_dAUC=d.mean(), power_delong_a05=(p < 0.05).mean(),
                power_mc_se=np.sqrt((p < 0.05).mean() * (1 - (p < 0.05).mean()) / N_SIM))

if __name__ == "__main__":
    jobs = [(n, r, f, s) for r, f in RATIOS.items() for s in (1.0, 0.5) for n in N_GRID]
    rows = Parallel(n_jobs=2)(delayed(cell)(*j) for j in jobs)
    T = pd.DataFrame(rows).round(4)
    T.to_csv("results/A3_sim_samplesize_simulation_only.csv", index=False)
    print(T.to_string(index=False))
    # N for 80% power: linear interpolation between the two grid points that bracket 0.80
    n80 = []
    for (r, s), g in T.groupby(["class_ratio", "eeg_weight_factor"], sort=False):
        g = g.sort_values("n_total"); n = g.n_total.values; p = g.power_delong_a05.values
        i = int(np.argmax(p >= 0.80))
        est = n[i - 1] + (0.80 - p[i - 1]) * (n[i] - n[i - 1]) / (p[i] - p[i - 1]) if i > 0 else float(n[0])
        n80.append(dict(class_ratio=r, eeg_weight_factor=s, n_for_80pct_power_interpolated=round(est)))
    N80 = pd.DataFrame(n80)
    N80.to_csv("results/A3_sim_samplesize_n80.csv", index=False)
    print("\nSample size for 80% power (interpolated):")
    print(N80.to_string(index=False))
