"""Test set-up: run every test from the repository root with src/ importable.

Some analysis scripts read sys.argv when they are imported (for example run_ensembles.py reads the
number of repeats), so sys.argv is cleared while they are imported by the tests.
"""
import os, sys, pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)


@pytest.fixture(autouse=True)
def _repo_root(monkeypatch):
    monkeypatch.chdir(REPO)


def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true", default=False,
                     help="also run slow tests (full re-run of the primary estimate, several minutes)")


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: slow test, run only with --runslow")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        return
    skip = pytest.mark.skip(reason="slow; run with --runslow")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)
