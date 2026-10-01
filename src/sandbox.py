"""
Read only SQLite access. Used in offline execution (execute.py) and liver serving
(/generalize endpoint), safe to run untrusted SQL against database file.

Layers of enforcement:
1. mode=ro on the connection URI so the OS/VFS layer refuses to open the
main database file for writing
2. SQLite authorizer callback - every individual operation is checked at
execution inside the database engine.

Authorizer exists because sqlglot accepts `ATTACH DATABASE ... AS x` inside a
CTE body as syntactically valid. Authorizer denies ATTACH regardless of what mode
the main connection was opened in.
"""
import sqlite3

_ALLOWED_ACTIONS = frozenset({
    sqlite3.SQLITE_SELECT,
    sqlite3.SQLITE_READ,
    sqlite3.SQLITE_FUNCTION,
    sqlite3.SQLITE_TRANSACTION,
})

# informational pragma allowed for introspection
_SAFE_PRAGMAS = frozenset({
    "table_info", "table_xinfo", "foreign_key_list", "index_list", "index_info",
})

# abort query after ~1000 interval SQLite VM steps
_PROGRESS_STEPS = 1000


def _readonly_authorizer(action, arg1, arg2, dbname, source):
    if action == sqlite3.SQLITE_PRAGMA:
        return sqlite3.SQLITE_OK if arg1 in _SAFE_PRAGMAS else sqlite3.SQLITE_DENY
    return sqlite3.SQLITE_OK if action in _ALLOWED_ACTIONS else sqlite3.SQLITE_DENY


def open_readonly(db_path: str) -> sqlite3.Connection:
    """
    Open db_path with both enforcement layers active. Should be used by
    every call site that runs generated/untrusted SQL against a database.
    """
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    con.set_authorizer(_readonly_authorizer)
    con.set_progress_handler(lambda: 1, _PROGRESS_STEPS)
    return con