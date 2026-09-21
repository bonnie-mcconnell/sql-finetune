"""
Execution-based accuracy: runs gold and generated SQL against SQLite databases
and compares returned rows. Complements the structural checking in 
evaluate.py.
"""

from collections import Counter

import sqlglot
from sqlglot import exp


def has_order_by(sql: str, dialect: str = "sqlite") -> bool:
    """True if query has a top-level ORDER BY, meaning row order
    is important and must be preserved when comparing rows."""
    try:
        return sqlglot.parse_one(sql, read=dialect).find(exp.Order) is not None
    except Exception: #noqa: BLE001
        return False


def run_query(db_path: str, sql: str):
    """
    Execute sql against the SQLite database at db_path, read-only.
    Opened with mode=ro so SQLite driver refuses write attempts.
    Progress handler aborts query after ~1000 internal SQLite VM steps.

    Returns (rows, None) on success, where rows is a list of tuples with 
    each row's values sorted by string representation/
    Returns (None, error_message) on any failure.
    """
    import sqlite3

    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    con.set_progress_handler(lambda: 1, 1000)
    try:
        rows = con.execute(sql).fetchall()
        rows = [tuple(sorted(row, key=str)) for row in rows]
        return rows, None
    except Exception as e: # noqa: BLE001
        return None, str(e)
    finally:
        con.close()


def execution_match(db_path: str, gold_sql: str, gen_sql: str):
    """
    Compare gold and generated SQL by executing against db_path and
    comparing returned rows.
    Returns (True, None) if they match, (False, error_or_none) if they 
    don't, and (None, message) if gold failed to execute (to isolate data
    errors vs gen_sql errors).
    Raw order compared when gold uses ORDER BY, otherwise compared as
    multiset (Counter).
    """
    gold_rows, gold_err = run_query(db_path, gold_sql)
    if gold_err is not None:
        return None, f"gold query failed: {gold_err}"

    gen_rows, gen_err = run_query(db_path, gen_sql)
    if gen_err is not None:
        return False, gen_err

    if has_order_by(gold_sql):
        return (gold_rows == gen_rows), None
    return (Counter(gold_rows) == Counter(gen_rows)), None


def score_execution(results: list[dict], db_dir: str) -> dict:
    """
    Execution accuracy across a list of result dicts. Rows where gold
    failed to execute are excluded from correct and total.
    """
    correct = 0
    exec_failures = []
    gold_failures = []
    total = len(results)

    for r in results:
        db_path = f"{db_dir}/{r['db_id']}.sqlite"
        match, err = execution_match(db_path, r["gold"], r["generated"])
        if match is None:
            gold_failures.append({**r, "error": err})
            continue
        if match:
            correct += 1
        elif err is not None:
            exec_failures.append({**r, "error": err})

    return {
        "correct": correct,
        "total": total,
        "accuracy": correct / total,
        "exec_failures": exec_failures,
        "gold_failures": gold_failures,
    }


def structural_execution_disagreement(results: list[dict], db_dir: str, exact_set_match_fn) -> dict:
    """
    Checks where structural and execution scoring disagree using list
    of result dicts.

    struct_correct_exec_wrong: exact_set_match says correct but the queries
    return different rows when executed (e.g self-join column collision)

    exec_correct_struct_wrong: queries return identical rows but exact_set_match
    says they're structurally different.

    Rows where gold failed are excluded from both lists.
    """
    struct_correct_exec_wrong = []
    exec_correct_struct_wrong = []

    for r in results:
        db_path = f"{db_dir}/{r['db_id']}.sqlite"
        struct = exact_set_match_fn(r["gold"], r["generated"])
        match, _ = execution_match(db_path, r["gold"], r["generated"])
        if match is None:
            continue
        if struct and match is False:
            struct_correct_exec_wrong.append(r)
        elif match is True and not struct:
            exec_correct_struct_wrong.append(r)

    return {
        "struct_correct_exec_wrong": struct_correct_exec_wrong,
        "exec_correct_struct_wrong": exec_correct_struct_wrong,
    }