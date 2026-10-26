"""Automated leakage checks (added in release v1.0.1).

Run from the repository root:   python -m pytest tests -v
Slow full re-run of the primary estimate (several minutes):   python -m pytest tests -v --runslow

The checks use the released derived features and result files; the raw EEG is not needed.
What they establish:
  1. The unit of analysis is the participant: every feature table has one row per participant.
  2. Labels and identifiers are never used as predictors, and feature-extraction code never reads labels.
  3. Outer cross-validation folds partition the participants, and the released out-of-fold score files
     were produced from exactly these participant-wise folds.
  4. Instrumented runs of the real fitting code (primary analysis, ensemble arm, addendum A1) record every
     participant used to fit any model or preprocessing step, and confirm that none belongs to the test fold.
  5. The permutation test permutes only EEG rows, only within age-band x sex strata, and never the labels.
  6. Pure-noise EEG adds nothing to the base model.
  7. (slow) The primary estimate is reproduced from the released features.
"""
import ast, glob, importlib, json, os, sys
import numpy as np, pandas as pd, pytest
from sklearn.model_selection import StratifiedKFold

OFFSET = 1000.0
BANNED = {"label", "labels", "diagnosed", "group", "subject", "user"}


# ----------------------------------------------------------------------------- helpers
def _import(name):
    """Import an analysis module with sys.argv cleared (some scripts parse argv at import time)."""
    old = sys.argv
    sys.argv = [old[0]]
    try:
        return importlib.import_module(name)
    finally:
        sys.argv = old


def _balladeer(variant="primary"):
    """Analysis set of the BALLADEER ensemble arm; mirrors run_cohort.load() (not imported because it
    rewrites module-level settings of run_ensembles when imported)."""
    D = pd.read_csv("derived/features_balladeer.csv")
    D = D[D.n_bad <= 4].copy()
    if variant == "sensitivity":
        D = D[(D.group == 1) | (D.diagnosed == "no")].copy()
        D["label"] = (D.group == 1).astype(int)
    return D.sort_values("subject").reset_index(drop=True)


def _index_encoded(D, cols):
    """Copy of D in which every predictor column holds the row index + OFFSET, and every 19 x 19
    covariance is diagonal with the same value (so it stays symmetric positive definite).
    Any array passed to a fit() can then be mapped back to the participants it contains."""
    idx = np.arange(len(D), dtype=float) + OFFSET
    new = {c: idx for c in cols}
    for i in range(19):
        for j in range(19):
            new[f"cov_{i}_{j}"] = idx if i == j else np.zeros(len(D))
    E = D.drop(columns=[c for c in new if c in D.columns]).copy()
    return pd.concat([E, pd.DataFrame(new, index=D.index)], axis=1)


def _ids(X):
    if isinstance(X, pd.DataFrame):
        return X.index.to_numpy()
    return np.rint(np.asarray(X, dtype=float)[:, 0] - OFFSET).astype(int)


class _Spy:
    """Wraps a real sklearn pipeline and records which participants each fit() receives."""
    def __init__(self, log, real):
        self.log, self.real = log, real

    def fit(self, X, y=None, **kw):
        self.log.append(_ids(X))
        self.real.fit(X, y, **kw)
        return self

    def predict_proba(self, X):
        return self.real.predict_proba(X)

    def __getattr__(self, name):
        return getattr(self.real, name)


def _spy_on(monkeypatch, module, log):
    real = module.make_pipeline
    monkeypatch.setattr(module, "make_pipeline", lambda *s, **k: _Spy(log, real(*s, **k)))


def _assert_no_test_rows(log, tr, te, min_fits):
    assert len(log) >= min_fits, f"only {len(log)} fits recorded; the spy did not intercept the pipelines"
    tr_set, te_set = set(tr.tolist()), set(te.tolist())
    for ids in log:
        s = set(ids.tolist())
        assert not (s & te_set), f"a model was fitted on {len(s & te_set)} test-fold participants"
        assert s <= tr_set, "a model was fitted on rows outside the outer training set"
    assert any(set(ids.tolist()) == tr_set for ids in log), "no fit used the full outer training set"


# ----------------------------------------------------------------------------- 1. unit of analysis
@pytest.mark.parametrize("path", sorted(glob.glob("derived/features*.csv")) or ["derived/features_balladeer.csv"])
def test_one_row_per_participant(path):
    if not os.path.exists(path):
        pytest.skip(f"{path} not present")
    D = pd.read_csv(path)
    assert D["subject"].is_unique, f"{path}: participants appear in more than one row"


# ----------------------------------------------------------------------------- 2. predictors and features
def _literal_lists(path, names):
    out = {}
    for node in ast.walk(ast.parse(open(path, encoding="utf-8").read())):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.List):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id in names:
                    out[t.id] = [ast.literal_eval(e) for e in node.value.elts]
    return out


def test_labels_and_identifiers_are_never_predictors():
    M = _import("rq1_A3")
    R = _import("run_ensembles")
    D, fam, chan = R.load()
    predictor_sets = {
        "rq1_A3 EEG families": sum(M.FAM.values(), []),
        "rq1_A3 base model": M.DEMO + M.BEH,
        "run_ensembles families": sum(fam.values(), []),
        "run_ensembles channel votes": sum(chan.values(), []),
        "run_ensembles benchmark": list(R.CONF),
    }
    lists = _literal_lists("src/run_cohort.py", {"DEMO", "ACQ"})
    predictor_sets["run_cohort benchmarks"] = lists["DEMO"] + lists["ACQ"]
    for name, cols in predictor_sets.items():
        bad = [c for c in cols if c.lower() in BANNED or c.lower().startswith("label")]
        assert not bad, f"{name} uses {bad} as predictors"


@pytest.mark.parametrize("path,skip_functions", [("src/feat_utils.py", set()),
                                                 ("src/extract_features.py", {"main"})])
def test_feature_extraction_never_reads_labels(path, skip_functions):
    """Feature functions must not refer to labels or group membership (the main() that writes the
    label column next to the features is excluded; it does not compute features from it)."""
    tree = ast.parse(open(path, encoding="utf-8").read())
    for fn in [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name not in skip_functions]:
        used = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name)} | \
               {n.attr for n in ast.walk(fn) if isinstance(n, ast.Attribute)} | \
               {n.value for n in ast.walk(fn) if isinstance(n, ast.Constant) and isinstance(n.value, str)}
        bad = {u for u in used if isinstance(u, str) and u.lower() in BANNED - {"subject", "user"}}
        assert not bad, f"{path}:{fn.name} refers to {bad}"


# ----------------------------------------------------------------------------- 3. folds and released scores
def _outer_folds(D, y, rep):
    return list(StratifiedKFold(5, shuffle=True, random_state=rep).split(D, y))


@pytest.mark.parametrize("which", ["balladeer_rq1", "nasrabadi"])
def test_outer_folds_partition_participants(which):
    if which == "balladeer_rq1":
        D = _import("rq1_A3").load("primary")
    else:
        D = _import("run_ensembles").load()[0]
    y = D.label.values
    assert D.subject.is_unique
    for rep in range(10):
        seen = []
        for tr, te in _outer_folds(D, y, rep):
            assert not set(tr) & set(te)
            assert len(tr) + len(te) == len(D)
            seen.extend(te.tolist())
        assert sorted(seen) == list(range(len(D))), "each participant must be tested exactly once per repeat"


@pytest.mark.parametrize("path,loader", [
    ("results/oof_scores.csv.gz", "nasrabadi"),
    ("results/a1_oof_scores.csv.gz", "nasrabadi"),
    ("results/balladeer_primary_oof_scores.csv.gz", "balladeer_primary"),
    ("results/balladeer_sensitivity_oof_scores.csv.gz", "balladeer_sensitivity"),
])
def test_released_scores_come_from_participant_wise_folds(path, loader):
    if not os.path.exists(path):
        pytest.skip(f"{path} not present")
    S = pd.read_csv(path)
    D = _import("run_ensembles").load()[0] if loader == "nasrabadi" else _balladeer(loader.split("_")[1])
    y = D.label.values
    # each participant is scored exactly once per repeat and method
    counts = S.groupby(["rep", "method", "subject"]).size()
    assert (counts == 1).all(), "a participant was scored more than once within a repeat"
    # the test participants of every (repeat, fold) are exactly those of the re-derived outer split
    for rep, Sr in S.groupby("rep"):
        for k, (tr, te) in enumerate(_outer_folds(D, y, int(rep))):
            got = set(Sr.loc[Sr.fold == k, "subject"])
            assert got == set(D.subject.values[te]), f"rep {rep}, fold {k}: test participants differ"
            assert not got & set(D.subject.values[tr])


@pytest.mark.parametrize("tag", ["primary_est", "primary_test", "S1_est", "S2_est"])
def test_rq1_score_files_have_one_row_per_participant(tag):
    path = f"results/A3_rq1_{tag}_oof.csv"
    if not os.path.exists(path):
        pytest.skip(f"{path} not present")
    S = pd.read_csv(path)
    assert S.subject.is_unique
    D = _import("rq1_A3").load("S1" if tag.startswith("S1") else "S2" if tag.startswith("S2") else "primary")
    assert set(S.subject) == set(D.subject)
    lab = D.set_index("subject").label
    assert (S.set_index("subject").label == lab.loc[S.subject].values).all()


# ----------------------------------------------------------------------------- 4. instrumented fitting
def test_primary_analysis_never_fits_on_test_participants(monkeypatch):
    """Every fit in the primary analysis (base model, five EEG families in the inner loop and refit,
    stacking meta-model) is recorded for all five outer folds of one repeat."""
    M = _import("rq1_A3")
    D = M.load("primary"); y = D.label.values
    E = _index_encoded(D, sum((v for f, v in M.FAM.items() if f != "F5_riemann"), []) + M.DEMO + M.BEH)
    for k, (tr, te) in enumerate(_outer_folds(E, y, 0)):
        log = []
        _spy_on(monkeypatch, M, log)
        pb, pf = M.fold(E, y, tr, te, 0, k)
        assert len(pb) == len(te) and len(pf) == len(te)
        # 1 base + 5 families x (5 inner + 1 refit) + 1 meta-model = 32 fits
        _assert_no_test_rows(log, tr, te, min_fits=32)


def test_ensemble_arm_never_fits_on_test_participants(monkeypatch):
    """Ensemble arm (30 members, channel votes, benchmarks, early fusion) for one outer fold."""
    R = _import("run_ensembles")
    D, fam, chan = R.load(); y = D.label.values
    cols = sum((v for f, v in fam.items() if f != "F5_riemann"), []) + list(R.CONF)
    E = _index_encoded(D, cols)
    tr, te = _outer_folds(E, y, 0)[0]
    log = []
    _spy_on(monkeypatch, R, log)
    rows, stats = R.run_fold(0, 0, tr, te, E, fam, chan)
    assert {r["subject"] for r in rows} == set(E.subject.values[te])
    # 30 members x 6 fits + 19 channel votes + 6 benchmarks + 1 early fusion = 206 fits
    _assert_no_test_rows(log, tr, te, min_fits=206)


def test_addendum_a1_stacking_never_fits_on_test_participants(monkeypatch):
    A = _import("run_a1")
    R = _import("run_ensembles")
    D, fam, chan = R.load(); y = D.label.values
    cols = sum((v for f, v in fam.items() if f != "F5_riemann"), []) + list(A.CONF)
    E = _index_encoded(D, cols)
    tr, te = _outer_folds(E, y, 0)[0]
    log = []
    _spy_on(monkeypatch, A, log)
    rows = A.run_fold(0, 0, tr, te, E, fam)
    assert {r["subject"] for r in rows} == set(E.subject.values[te])
    _assert_no_test_rows(log, tr, te, min_fits=6 * 6 + 2 * 30 * 6)


# ----------------------------------------------------------------------------- 5. permutation test
def test_permutation_stays_within_strata():
    M = _import("rq1_A3")
    D = M.load("primary"); strata = D.stratum.values
    rng = np.random.default_rng(10_000)
    perm = M.strat_perm(strata, rng)
    assert sorted(perm) == list(range(len(D)))
    assert (strata[perm] == strata).all(), "EEG rows were moved across age-band x sex strata"


def test_permutation_moves_only_eeg_rows(monkeypatch):
    M = _import("rq1_A3")
    D = M.load("primary"); y = D.label.values.copy()
    perm = M.strat_perm(D.stratum.values, np.random.default_rng(10_001))
    seen = {}

    def fake_fold(Dp, yp, tr, te, rep, k):
        seen["D"], seen["y"] = Dp, yp
        return np.full(len(te), 0.5), np.full(len(te), 0.5)

    monkeypatch.setattr(M, "fold", fake_fold)
    M.run_once(D, y, 1, eeg_perm=perm)
    Dp = seen["D"]
    other = [c for c in D.columns if c not in M.EEG_COLS]
    pd.testing.assert_frame_equal(Dp[other], D[other])
    assert np.array_equal(Dp[M.EEG_COLS].values, D[M.EEG_COLS].values[perm], equal_nan=True)  # features contain NaN
    assert np.array_equal(seen["y"], D.label.values), "labels must never be permuted"


# ----------------------------------------------------------------------------- 6. noise sanity check
def test_pure_noise_eeg_adds_nothing():
    """Replacing all EEG features by noise must not produce an increment over the base model.
    Three noise data sets with fixed seeds; the expected mean increment is below zero."""
    M = _import("rq1_A3")
    D = M.load("primary"); y = D.label.values
    cols14 = sum((v for f, v in M.FAM.items() if f != "F5_riemann"), [])
    d = []
    for seed in range(3):
        rng = np.random.default_rng(seed)
        N = D.copy()
        N[cols14] = rng.normal(size=(len(D), len(cols14)))
        A = rng.normal(size=(len(D), 19, 19)); C = A @ A.transpose(0, 2, 1) / 19 + np.eye(19)
        N[M.FAM["F5_riemann"]] = C.reshape(len(D), -1)
        d.append(M.run_once(N, y, 1)[0]["dauc"])
    assert np.mean(d) < 0.03, f"noise EEG gave a mean increment of {np.mean(d):+.3f}"


# ----------------------------------------------------------------------------- 7. reproduction (slow)
@pytest.mark.slow
def test_primary_estimate_is_reproduced():
    M = _import("rq1_A3")
    D = M.load("primary"); y = D.label.values
    obs, sb, sf = M.run_once(D, y, 10)
    ref = pd.read_csv("results/A3_rq1_primary_est_oof.csv")
    assert np.array_equal(ref.subject.values, D.subject.values)
    assert np.max(np.abs(ref.score_base.values - sb)) < 1e-6
    assert np.max(np.abs(ref.score_full.values - sf)) < 1e-6
    summ = json.load(open("results/A3_rq1_primary_est_summary.json"))
    assert abs(summ["dauc"] - obs["dauc"]) < 1e-9
