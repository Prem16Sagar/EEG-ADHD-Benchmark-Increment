"""Simulation only (no real data): precision/power for paired ΔAUC at n=79/42.
Binormal latent model; oracle combination (upper bound: ignores weight-estimation loss in nested CV)."""
import numpy as np, sys
from scipy.stats import norm
sys.path.insert(0, "src"); from analyse import delong
rng = np.random.default_rng(42)
n1, n0 = 79, 42; y = np.r_[np.ones(n1), np.zeros(n0)].astype(int)
mu_b = np.sqrt(2)*norm.ppf(0.72)
out = []
for target in [0.72, 0.75, 0.77, 0.80]:
    mu_e = np.sqrt(max((np.sqrt(2)*norm.ppf(target))**2 - mu_b**2, 0))
    for shrink in [1.0, 0.5]:   # 0.5 = noisy weight estimation (half the EEG signal recovered)
        d, p = [], []
        for _ in range(2000):
            B = rng.normal(mu_b*y, 1); E = rng.normal(mu_e*y, 1)
            full = mu_b*B + shrink*mu_e*E   # LDA-optimal weights (shrink<1: under-weighted EEG)
            a, b, z, pv = delong(y, full, B); d.append(a-b); p.append(pv)
        d = np.array(d); p = np.array(p)
        out.append((target, shrink, d.mean(), np.percentile(d,2.5), np.percentile(d,97.5), (p<0.05).mean()))
import pandas as pd
pd.DataFrame(out, columns=['true_full_AUC','eeg_weight_factor','mean_dAUC','dAUC_2.5','dAUC_97.5','power_delong_a05']).round(3).to_csv('results/A3_sim_power_simulation_only.csv', index=False)
print("true_full_AUC shrink mean_dAUC 2.5% 97.5% power(DeLong,a=.05)")
for r in out: print("%.2f %.1f %.3f %.3f %.3f %.2f" % r)
