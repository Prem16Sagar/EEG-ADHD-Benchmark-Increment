# Changelog

## v1.0.1 (2026-10-26)

No analysis code and no existing result file was changed.

- **Added** `tests/` with automated leakage checks (`python -m pytest tests`), described in the README.
- **Added** `src/sim_samplesize_A3.py`, a post-hoc, simulation-only extension of `sim_power_A3.py` that estimates the sample size needed for 80% power. Its outputs are `results/A3_sim_samplesize_simulation_only.csv` and `results/A3_sim_samplesize_n80.csv` (Supplementary Table S1d).
- **README:** figure and table numbers now follow the submitted manuscript; new sections on the leakage checks and the sample-size simulation.
- **requirements.txt:** `pytest` added.
- **scripts/run_all.sh:** the two new steps added at the end.
- **CITATION.cff:** version 1.0.1.

## v1.0.0 (2026-10-26)

Initial release used for the analyses in the paper (Zenodo: https://doi.org/10.5281/zenodo.23664727).
