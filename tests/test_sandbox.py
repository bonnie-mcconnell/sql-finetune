import sqlite3
import tempfile
from pathlib import Path

import pytest

from src.sandbox import open_readonly


@pytest.fixture
def db_path():
    path = Path(tempfile.mkdtemp()) / "test.sqlite"
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE t (x INT, y TEXT)")
    con.executemany("INSERT INTO t values (?, ?)", [(1, "a"), (2, "b")])
    con.commit()
    con.close()
    return str(path)


def test_normal_select_allowed(db_path):
    con = open_readonly(db_path)
    assert con.execute("SELECT * FROM t").fetchall() == [(1, "a"), (2, "b")]


def test_join_and_subquery_allowed(db_path):
    con = open_readonly(db_path)
    assert con.execute(
        "SELECT * FROM t WHERE x = (SELECT MIN(x) FROM t)"
    ).fetchall() == [(1, "a")]


@pytest.mark.parametrize("sql", [
    "DELETE FROM t",
    "UPDATE t SET x = 0",
    "INSERT INTO t VALUES (9, 'z')",
    "DROP TABLE t",
    "CREATE TABLE t2 (z)",
    "PRAGMA journal_mode=WAL",
])
def test_writes_and_dangerous_pragmas_blocked(db_path, sql):
    con = open_readonly(db_path)
    with pytest.raises(sqlite3.DatabaseError):
        con.execute(sql)


def test_attach_blocked(db_path):
    # regression test for ATTACH-via-CTE gap
    con = open_readonly(db_path)
    with pytest.raises(sqlite3.DatabaseError):
        con.execute("ATTACH DATABASE '/tmp/pwned.db' AS pwned")


def test_mode_ro_doesnt_stop_writes_via_attach(tmp_path):
    # documents why the authorizer layer is necessary, not just
    # mode=ro: mode=ro restricts the main connection's file, but does
    # nothing to stop ATTACHing a different, separate file and writing
    # to that one. shown here, the attached file gets created.
    main_db = tmp_path / "main.sqlite"
    sqlite3.connect(main_db).execute("CREATE TABLE t (x)")
    target = tmp_path / "attacked.sqlite"
    assert not target.exists()

    con = sqlite3.connect(f"file:{main_db}?mode=ro", uri=True)
    con.executescript(f"ATTACH DATABASE '{target}' AS atk; CREATE TABLE atk.evil (x);")
    assert target.exists()  # mode=ro alone did not prevent this