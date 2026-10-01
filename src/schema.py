"""
Schema introspection for user-uplaoded SQLite databases. Formats into
"table : col (type), ... | table2 : ..." string style the model was trained on.
"""

import sqlite3

_TYPE_MAP = {
    "INTEGER": "number", "INT": "number", "REAL": "number", "NUMERIC": "number",
    "FLOAT": "number", "DOUBLE": "number", "DECIMAL": "number",
    "TEXT": "text", "VARCHAR": "text", "CHAR": "text", "CLOB": "text",
    "DATE": "text", "DATETIME": "text", "TIMESTAMP": "text",
    "BOOL": "others", "BOOLEAN": "others",
}

MAX_TABLES = 25
MAX_COLUMNS_PER_TABLE = 40


def _normalize_type(declared: str) -> str:
    # take base type name from SQLite and fall back to "others" for anything
    # unrecognizable. match training schema convention of number/text categories
    base = declared.split("(")[0].strip().upper()
    return _TYPE_MAP.get(base, "others")


def introspect_schema(con: sqlite3.Connection) -> str:
    """
    Raises ValueError if database exceeds MAX_TABLE or MAX_COLUMNS_PER_TABLE.
    """
    tables = [
        r[0] for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%'"
        ).fetchall()
    ]
    if len(tables) > MAX_TABLES:
        raise ValueError(f"database has {len(tables)} tables, cap is {MAX_TABLES}")

    parts = []
    for t in tables:
        cols = con.execute(f'PRAGMA table_info("{t}")').fetchall()
        if len(cols) > MAX_COLUMNS_PER_TABLE:
            raise ValueError(
                f"table '{t}' has {len(cols)} columns, cap is {MAX_COLUMNS_PER_TABLE}"
            )
        col_strs = [f"{c[1]} ({_normalize_type(c[2])})" for c in cols]
        parts.append(f"{t} : {' , '.join(col_strs)}")
    return " | ".join(parts)