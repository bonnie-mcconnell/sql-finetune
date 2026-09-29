"""Test SQL execution comparison. Uses small on-disk SQLite database
created fresh every test."""

import sqlite3
import tempfile
from pathlib import Path

import pytest

from src.execute import execution_match, has_order_by, run_query, score_execution


@pytest.fixture
def db_path():
    path = Path(tempfile.mkdtemp()) / "test.sqlite"
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE singer (name TEXT, age INT, country TEXT)")
    con.executemany(
        "INSERT INTO singer VALUES (?, ?, ?)",
        [("Alice", 30, "France"), ("Bob", 25, "France"), ("Cleo", 40, "Egypt")],
    )
    con.commit()
    con.close()
    return str(path)


def test_has_order_by_true():
    assert has_order_by("SELECT name FROM singer ORDER BY age") is True


def test_has_order_by_false():
    assert has_order_by("SELECT name FROM singer") is False


def test_run_query_success(db_path):
    rows, err = run_query(db_path, "SELECT name FROM singer WHERE country = 'France'")
    assert err is None
    assert sorted(rows) == sorted([("Alice",), ("Bob",)])


def test_run_query_invalid_sql_errors(db_path):
    rows, err = run_query(db_path, "SELECT * FROM nonexistent_table")
    assert rows is None
    assert err is not None


def test_run_query_read_only(db_path):
    rows, err = run_query(db_path, "DELETE FROM singer")
    assert rows is None
    assert err is not None
    con = sqlite3.connect(db_path)
    count = con.execute("SELECT count(*) FROM singer").fetchone()[0]
    con.close()
    assert count == 3


def test_execution_match_true(db_path):
    match, err = execution_match(
        db_path,
        "SELECT name FROM singer WHERE age = (SELECT MIN(age) FROM singer)",
        "SELECT s1.name FROM singer s1 WHERE NOT EXISTS"
        " (SELECT 1 FROM singer s2 WHERE s2.age < s1.age)",
    )
    assert match is True
    assert err is None


def test_execution_match_false(db_path):
    match, _ = execution_match(
        db_path,
        "SELECT name FROM singer WHERE country = 'France'",
        "SELECT name FROM singer WHERE country = 'Egypt'",
    )
    assert match is False


def test_execution_match_string_literal(db_path):
    match, _ = execution_match(
        db_path,
        "SELECT name FROM singer WHERE country = 'France'",
        "SELECT name FROM singer WHERE country = 'FRANCE'",
    )
    assert match is False  # SQLite string comparison is case-sensitive 


def test_execution_match_gen_failure_false(db_path):
    match, err = execution_match(db_path, 
                                 "SELECT name FROM singer", 
                                 "SELECT nonexistent_col FROM singer")
    assert match is False
    assert err is not None


def test_execution_match_gold_failure_none(db_path):
    match, _ = execution_match(db_path, 
                                 "SELECT nonexistent_col FROM singer", 
                                 "SELECT name FROM singer")
    assert match is None


def test_execution_match_order_by(db_path):
    match, _ = execution_match(
        db_path,
        "SELECT name FROM singer ORDER BY age",
        "SELECT name FROM singer ORDER BY age DESC",
    )
    assert match is False # reversed order


def test_score_execution_separates_exec_and_gold_failures(db_path):
    results = [
        {
            "db_id": "test",
            "gold": "SELECT name FROM singer",
            "generated": "SELECT name FROM singer",
        },
        {
            "db_id": "test",
            "gold": "SELECT name FROM singer",
            "generated": "SELECT bad_col FROM singer",
        },
        {
            "db_id": "test",
            "gold": "SELECT bad_col FROM singer",
            "generated": "SELECT name FROM singer",
        },
    ]
    result = score_execution(results, str(Path(db_path).parent))
    assert result["correct"] == 1
    assert result["total"] == 3
    assert len(result["exec_failures"]) == 1
    assert len(result["gold_failures"]) == 1