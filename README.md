# Benchmarking EEG against non-neural baselines for ADHD classification

Code, locked analysis protocols, derived features and result files for:

> Sagar, P. & Ruban, S. *Benchmarking EEG against non-neural baselines for ADHD classification in children and adolescents: incremental value, cross-condition stability and transfer.* Manuscript submitted to *Scientific Reports* (2026).

Every number, table and figure in the paper can be traced to a file in `results/`. Each file can be regenerated from the raw public datasets with the scripts in `src/`.

## What the analysis does

- **Increment test.** Tests whether task EEG improves discrimination of ADHD beyond age, sex and in-game behaviour. Data: the BALLADEER dataset, Slackline Level 1, n = 121.
- **Method.** Participant-wise nested cross-validation, with a whole-pipeline permutation test.
- **Benchmarks.** Compares EEG models, including voting ensembles, with non-neural benchmarks in BALLADEER and in the Nasrabadi dataset.
- **Stability.** Measures the cross-level stability (ICC) of five EEG feature families.
- **Transfer.** Explores transfer across task levels, recording devices and datasets.

## Data (not redistributed)

| Dataset | Source | Put it in, or set |
|---|---|---|
| BALLADEER | https://doi.org/10.6084/m9.figshare.28676042 | `data/BALLADEER/`, or `BALLADEER_ROOT=<folder with users_demographics.json and UB*/>` |
| Nasrabadi (ADHD/control children) | https://doi.org/10.21227/rzfh-zn36 | `data/Nasrabadi/`, or `NASRABADI_ROOT=<folder>` |

Paths are set in one place: `src/paths.py`.

`derived/` contains participant-level features computed from these datasets, so the statistical analyses can be re-run without the raw data. The intermediate EEG arrays (`derived/balladeer_npz*`) are not included; the conversion scripts regenerate them.

## Installation

```
pip install -r requirements.txt
```

The paper used Python 3.13.16 with the pinned versions. The primary result was also reproduced with numpy 2.1.3, pandas 2.2.3 and scikit-learn 1.6.1 (see `verification/`).

## Reproducing the results

Run all commands from the repository root.

**Quick check, with no raw data needed:**
```
python src/rq1_A3.py run primary 10 0 mycheck
```
- This re-runs the primary estimate from `derived/` in a few minutes.
- Expected output: base AUC 0.6305, full AUC 0.6652, ΔAUC +0.0347.
- Compare `results/A3_rq1_mycheck_oof.csv` with `results/A3_rq1_primary_est_oof.csv`.

**Full pipeline:** `scripts/run_all.sh` lists every step in order, from raw data to figures.

| Paper item | Script(s) | Result file(s) |
|---|---|---|
| Table 1, Fig. 1 | `src/rq1_A3.py` (`load`), `src/figures_SciRep.py` | `results/SR_display_numbers.csv` |
| Fig. 2 (benchmarks) | `run_ensembles.py`, `analyse.py`, `run_a1.py`, `analyse_a1.py`, `run_cohort.py`, `analyse_cohort.py`, `posthoc_A4.py s5` | `results/table_all_methods.csv`, `a1_table_*.csv`, `balladeer_primary_table_*.csv`, `A4/S5_*.csv` |
| Table 2, Fig. 3 (increment) | `rq1_A3.py`, `rq1_A3_sec.py`, `analyse_rq1_A3.py`, `posthoc_A4.py` | `results/A3_rq1_*`, `A3_rq1sec_*`, `A3_rq1_table.csv`, `A4/*`, `SR_A4_dlogloss.csv`, `SR_rq1_secondary_metrics.csv` |
| Fig. 4 (stability, level transfer) | `balladeer_level.py`, `rq2_A3.py`, `rq2_age_posthoc.py` | `results/A3_rq2a_*.csv`, `A3_rq2b_transfer.csv`, `POSTHOC_rq2_age_adjusted_*.csv` |
| Device, wPLI and cross-dataset analyses | `shared10.py`, `rq34_A3.py` | `results/A3_rq3_device_transfer.csv`, `A3_h_wpli.csv`, `A3_rq4_cross_cohort10.csv` |
| Power simulation | `sim_power_A3.py` | `results/A3_sim_power_simulation_only.csv` |

## Protocols

`protocols/` holds the versioned analysis protocols. Each was written before the analyses it governs and locked with a SHA-256 hash:

| File | Scope |
|---|---|
| `protocol.yaml` | Nasrabadi main analysis |
| `protocol_addendum_A1.yaml` | Nasrabadi stacking |
| `protocol_addendum_A2_balladeer.yaml` | BALLADEER Level 1 benchmarks |
| `protocol_addendum_A3.yaml` | Primary increment analysis, stability and transfer (hashes in `A3_lock_sha256.txt`) |
| `protocol_addendum_A4_posthoc.yaml` | Post-hoc sensitivity analyses (hashes in `A4_lock_sha256.txt`) |

The protocols were pre-specified internally, not registered publicly. What had been seen before each lock, and all deviations, are reported in the paper's Methods.

## Notes on this release

The scripts are the ones used for the paper, with three release changes:
1. Raw-data and session paths are read from `src/paths.py`.
2. `balladeer_level.py` reads raw files in place: it finds each recording via `balladeer/A3_file_inventory_121.csv` instead of saved session listings, and it no longer deletes converted files.
3. `PREV_CONF` points to `derived/confounds_from_previous_paper.csv`, which is identical to the file used.

Before release, these were checked to reproduce the stored outputs exactly: the behaviour scoring (Levels 1 and 6) and the primary estimate.

## Use of AI tools

The code was written and run with the assistance of an AI system (Claude, Anthropic) under the direction of the first author, as described in the paper's Methods. The first author independently reproduced the primary analysis (`verification/`).

## Licence and citation

- Code: MIT licence (`LICENSE`).
- Please cite the article; see `CITATION.cff`.
- The datasets keep their original licences and must be cited as given by their authors.

Contact: Prem Sagar (prem.sagar_@sju.edu.in).
