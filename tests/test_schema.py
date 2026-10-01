import sqlite3

import pytest

from src.schema import MAX_COLUMNS_PER_TABLE, MAX_TABLES, introspect_schema


@pytest.fixture
def con():
    con = sqlite3.connect(":memory:")
    con.executescript(
        """
        CREATE TABLE members (member_id INTEGER, full_name TEXT, monthly_fee REAL);
        CREATE TABLE classes (class_id INTEGER, class_name TEXT);
        """
    )
    return con


def test_introspect_schema_format(con):
    schema = introspect_schema(con)
    assert "members : member_id (number) , full_name (text) , monthly_fee (number)" in schema
    assert "classes : class_id (number) , class_name (text)" in schema
    assert " | " in schema


def test_introspect_schema_excludes_internal_tables(con):
    schema = introspect_schema(con)
    assert "sqlite_" not in schema


def test_introspect_schema_rejects_max_tables():
    con = sqlite3.connect(":memory:")
    for i in range(MAX_TABLES + 1):
        con.execute(f"CREATE TABLE t{i} (x)")
    with pytest.raises(ValueError, match="tables"):
        introspect_schema(con)


def test_introspect_schema_rejects_max_cols():
    con = sqlite3.connect(":memory:")
    cols = ", ".join(f"c{i} INT" for i in range(MAX_COLUMNS_PER_TABLE + 1))
    con.execute(f"CREATE TABLE t ({cols})")
    with pytest.raises(ValueError, match="columns"):
        introspect_schema(con)


def test_introspect_schema_text_type_default():
    con = sqlite3.connect(":memory:")
    con.execute("CREATE TABLE t (x SOME_WEIRD_TYPE)")
    schema = introspect_schema(con)
    assert "x (others)" in schema


def test_bool_maps_to_others():
    con = sqlite3.connect(":memory:")
    con.execute("CREATE TABLE t (flag bool)")
    schema = introspect_schema(con)
    assert "flag (others)" in schema