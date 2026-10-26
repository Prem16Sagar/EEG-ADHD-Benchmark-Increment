"""Locations of the raw public datasets (not redistributed in this repository).

Set the environment variables BALLADEER_ROOT and NASRABADI_ROOT, or place the datasets in
data/BALLADEER and data/Nasrabadi inside the repository. All scripts are run from the repository root.
"""
import os
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BALLADEER_ROOT = os.environ.get("BALLADEER_ROOT", os.path.join(REPO, "data", "BALLADEER"))     # folder with users_demographics.json and UB*/ folders
NASRABADI_ROOT = os.environ.get("NASRABADI_ROOT", os.path.join(REPO, "data", "Nasrabadi"))     # folder with the ADHD/Control files
PREV_CONF = os.path.join(REPO, "derived", "confounds_from_previous_paper.csv")              # Nasrabadi acquisition proxies (Paper 1)
