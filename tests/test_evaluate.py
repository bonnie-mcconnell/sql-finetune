"""
Tests for structural SQL comparison and error categorization.

These four cases were checked by hand against generated model output
during development. The third case (unnecessary self-join through an unrelated 
table, produced by the un-fine-tuned base model) is the main
negative case because it confirms the scorer is able to distinguish different
query structure from superficial formatting differences.
"""

import pytest

from src.evaluate import categorize_error, exact_set_match, get_component_sets, paired_bootstrap

CASES = [
    (
        "SELECT count(*) FROM singer",
        "SELECT COUNT(*) FROM singer",
        True,
    ),
    (
        "SELECT name ,  country ,  age FROM singer ORDER BY age DESC",
        "SELECT Name, Country, Age FROM singer ORDER BY Age DESC",
        True,
    ),
    (
        "SELECT name ,  country ,  age FROM singer ORDER BY age DESC",
        (
            "SELECT T2.Name, T2.Country, T2.Age FROM singer AS T1 "
            "INNER JOIN singer_in_concert AS T3 ON T1.Singer_ID = T3.Singer_ID "
            "INNER JOIN singer AS T2 ON T3.Singer_ID = T2.Singer_ID "
            "ORDER BY T2.Age DESC"
        ),
        False,
    ),
    (
        "SELECT avg(age) ,  min(age) ,  max(age) FROM singer WHERE country  =  'France'",
        "SELECT AVG(Age), MIN(Age), MAX(Age) FROM singer WHERE Country = 'France'",
        True,
    ),
]


@pytest.mark.parametrize("gold,gen,expected", CASES)
def test_exact_set_match(gold, gen, expected):
    assert exact_set_match(gold, gen) is expected


def test_categorize_error_correct():
    assert categorize_error("SELECT * FROM t", "select * from t") == "correct"


def test_categorize_error_wrong_table():
    assert categorize_error("SELECT * FROM singer", "SELECT * FROM concert") == "wrong_tables"


def test_categorize_error_wrong_columns():
    assert categorize_error("SELECT name FROM singer", "SELECT age FROM singer") == "wrong_columns"


def test_categorize_error_unparseable():
    assert categorize_error("SELECT * FROM singer", "not valid sql at all (((") == "unparseable"


def test_exact_set_match_reordered_and_conditions_are_equal():
    # Regression test for earlier bug: get_component_sets
    # originally compared WHERE clauses via Where.flatten() on the wrapper
    # node, which returns the whole clause as one un-split string, so logically-identical, 
    # differently-ordered AND-conjuncts incorrectly scored as not matching
    # Fixed with explicit AND-conjunct splitting (_split_and_conditions).
    assert exact_set_match(
        "SELECT * FROM singer WHERE age > 20 AND country = 'France'",
        "SELECT * FROM singer WHERE country = 'France' AND age > 20",
    ) is True


def test_exact_set_match_distinguishes_comparison_operators():
    # comparisons/operators must be preserved in correct order
    assert exact_set_match(
        "SELECT * FROM t WHERE age > 20", "SELECT * FROM t WHERE age < 20"
    ) is False


def test_exact_set_match_distinguishes_and_from_or():
    # AND and OR must never compare equal, even with identical operands.
    assert exact_set_match(
        "SELECT * FROM t WHERE age > 20 AND country = 'France'",
        "SELECT * FROM t WHERE age > 20 OR country = 'France'",
    ) is False


def test_exact_set_match_distinguishes_limit():
    # Regression test for LIMIT clause (originally ignored)
    assert exact_set_match(
        "SELECT name FROM singer ORDER BY age DESC LIMIT 3",
        "SELECT name FROM singer ORDER BY age DESC",
    ) is False


def test_exact_set_match_distinguishes_distinct():
    assert exact_set_match(
        "SELECT DISTINCT country FROM singer", "SELECT country FROM singer"
    ) is False


def test_exact_set_match_distinguishes_having():
    assert exact_set_match(
        "SELECT country, count(*) FROM singer GROUP BY country HAVING count(*) > 1",
        "SELECT country, count(*) FROM singer GROUP BY country",
    ) is False


def test_exact_set_match_never_raises_on_garbage_input():
    # Malformed SQL must score False, not propagate a parser exception 
    # so evlauation loop never crashes 
    assert exact_set_match("SELECT * FROM t", "((( not sql") is False


def test_get_component_sets_order_independent():
    a = get_component_sets("SELECT name, age FROM singer")
    b = get_component_sets("SELECT age, name FROM singer")
    assert a["select"] == b["select"]


def test_get_component_sets_case_and_whitespace_insensitive():
    a = get_component_sets("SELECT   Name   FROM Singer")
    b = get_component_sets("select name from singer")
    assert a["select"] == b["select"]
    assert a["tables"] == b["tables"]


def test_paired_bootstrap_no_difference_is_centered_on_zero():
    # Identical correctness vectors = true difference is exactly 0, CI should include 0.
    correct = [1, 0, 1, 1, 0, 1, 0, 0, 1, 1]
    mean_diff, ci_low, ci_high = paired_bootstrap(correct, correct, n_boot=2000)
    assert mean_diff == pytest.approx(0.0, abs=1e-9)
    assert ci_low <= 0.0 <= ci_high


def test_paired_bootstrap_large_true_difference_excludes_zero():
    # Fine-tuned always correct, base always wrong = 
    # CI should be tight around 1.0 and clearly exclude zero
    base_correct = [0] * 50
    ft_correct = [1] * 50
    mean_diff, ci_low, ci_high = paired_bootstrap(base_correct, ft_correct, n_boot=2000)
    assert mean_diff == pytest.approx(1.0)
    assert ci_low == pytest.approx(1.0)
    assert ci_high == pytest.approx(1.0)


def test_paired_bootstrap_is_deterministic_given_seed():
    base_correct = [1, 0, 1, 0, 1, 1, 0, 0, 1, 0]
    ft_correct = [1, 1, 1, 0, 1, 1, 1, 0, 1, 1]
    result_a = paired_bootstrap(base_correct, ft_correct, n_boot=500, seed=7)
    result_b = paired_bootstrap(base_correct, ft_correct, n_boot=500, seed=7)
    assert result_a == result_b


def test_paired_bootstrap_ci_bounds_are_ordered():
    base_correct = [1, 0, 0, 1, 0, 1, 1, 0, 1, 0]
    ft_correct = [1, 1, 0, 1, 1, 1, 0, 0, 1, 1]
    mean_diff, ci_low, ci_high = paired_bootstrap(base_correct, ft_correct, n_boot=2000)
    assert ci_low <= mean_diff <= ci_high