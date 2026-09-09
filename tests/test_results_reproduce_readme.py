"""
Reproducibility test: confirms results/base_results.json and
results/finetuned_results.json, run through this project's scoring code,
produce the README numbers.

This acts as a regression guard, if exact_set_match or categorize_error's
logic changes, this test fails immediately rather than letting the README's 
claims drift out of sync with what the code actually computes. This doesn't 
test the model's actual quality.
"""

import numpy as np
import pytest

from src.evaluate import (
    error_breakdown,
    exact_set_match,
    load_results,
    paired_bootstrap,
)

RESULTS_DIR = "results"


@pytest.fixture(scope="module")
def base_results():
    return load_results(f"{RESULTS_DIR}/base_results.json")


@pytest.fixture(scope="module")
def finetuned_results():
    return load_results(f"{RESULTS_DIR}/finetuned_results.json")


def test_result_files_have_expected_size(base_results, finetuned_results):
    assert len(base_results) == 1034
    assert len(finetuned_results) == 1034


def test_base_accuracy_matches_readme(base_results):
    correct = [exact_set_match(r["gold"], r["generated"]) for r in base_results]
    assert np.mean(correct) == pytest.approx(0.2099, abs=0.001)


def test_finetuned_accuracy_matches_readme(finetuned_results):
    correct = [exact_set_match(r["gold"], r["generated"]) for r in finetuned_results]
    assert np.mean(correct) == pytest.approx(0.5019, abs=0.001)


def test_paired_bootstrap_ci_excludes_zero(base_results, finetuned_results):
    base_correct = [exact_set_match(r["gold"], r["generated"]) for r in base_results]
    ft_correct = [exact_set_match(r["gold"], r["generated"]) for r in finetuned_results]
    mean_diff, ci_low, ci_high = paired_bootstrap(base_correct, ft_correct)

    assert mean_diff == pytest.approx(0.292, abs=0.005)
    # check the 95% CI excludes zero, the improvement is statistically real & not sampling noise.
    assert ci_low > 0
    # Sanity bound: an accuracy difference can never exceed 1.0.
    assert ci_high < 1.0


def test_error_breakdown_matches_readme(base_results, finetuned_results):
    base_breakdown = error_breakdown(base_results)
    ft_breakdown = error_breakdown(finetuned_results)

    # Spot-check the two error-analysis claims from the README:
    # table-selection errors nearly halved, column-selection errors
    # dropped by more than two-thirds. Unaffected by the LIMIT/DISTINCT/
    # HAVING/WHERE-splitting fix, since these categories are checked
    # before those components in categorize_error's priority ladder.
    assert base_breakdown["wrong_tables"] == 507
    assert ft_breakdown["wrong_tables"] == 292
    assert base_breakdown["wrong_columns"] == 163
    assert ft_breakdown["wrong_columns"] == 47
