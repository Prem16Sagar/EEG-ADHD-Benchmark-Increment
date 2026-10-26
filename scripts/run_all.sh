#!/bin/bash
# Full pipeline from raw data, in the order used for the paper. Run from the repository root.
# Raw data locations: set BALLADEER_ROOT and NASRABADI_ROOT (see src/paths.py).
# Approximate run times on 2 CPU cores are given in brackets; the permutation tests dominate.
set -e

# --- Nasrabadi (protocol.yaml, addendum A1) ---
python src/extract_features.py                      # derived/features_nasrabadi.csv
python src/run_ensembles.py 10                      # ensemble arm, 10 repeats
python src/analyse.py                               # results/table_all_methods.csv, table_confirmatory_tests.csv
python src/run_a1.py && python src/analyse_a1.py    # results/a1_*

# --- BALLADEER Level 1 (addendum A2) ---
python src/balladeer_convert.py                     # derived/balladeer_npz/ + derived/balladeer_confounds.csv
python src/balladeer_features.py                    # derived/features_balladeer.csv
python src/run_cohort.py primary && python src/analyse_cohort.py primary
python src/run_cohort.py sensitivity && python src/analyse_cohort.py sensitivity

# --- Behaviour (addendum A3) ---
python src/score_behaviour_A3.py                    # derived/behaviour_L1_A3.csv

# --- RQ1: primary analysis and secondary/robustness analyses (addendum A3, locked) ---
python src/rq1_A3.py run primary 10 0 primary_est    # point estimate (minutes)
python src/rq1_A3.py run primary 3 1000 primary_test # permutation test (about 1 h)
python src/rq1_A3_sec.py acq 1 500
python src/rq1_A3_sec.py nobeh 1 500
python src/rq1_A3.py run S1 10 0 S1_est
python src/rq1_A3.py run S2 10 0 S2_est
python src/rq1_A3_sec.py spectral 10 0               # post hoc
python src/rq1_A3_sec.py tunedC 10 0
python src/analyse_rq1_A3.py                         # results/A3_rq1_table.csv

# --- Levels 6 and 11, stability and level transfer (RQ2) ---
for L in 6 11; do python src/balladeer_level.py convert $L; python src/balladeer_level.py features $L; done
python src/rq2_A3.py 500
python src/rq2_age_posthoc.py 500                    # post hoc

# --- Shared 10 channels: device transfer, wPLI hypothesis, cross-dataset (RQ3, RQ4) ---
python src/shared10.py all
python src/rq34_A3.py all

# --- Post-hoc sensitivity analyses (addendum A4) ---
python src/posthoc_A4.py s1 500
python src/posthoc_A4.py s2 500
python src/posthoc_A4.py s3
python src/posthoc_A4.py s4feat
python src/posthoc_A4.py s4
python src/posthoc_A4.py s5
python src/posthoc_A4.py s6score
python src/posthoc_A4.py s6 500

# --- Figures ---
python src/figures_SciRep.py
