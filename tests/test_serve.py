"""
Tests for the FastAPI serving layer.

The success-path /generate tests monkeypatch get_model_and_tokenizer and
generate_sql so they never load an actual model or need GPU because CI has
neither. They test the API's request/response contract and the
read-only-SQL guardrail, not model output quality.
"""

import pytest
from fastapi.testclient import TestClient

from src import serve


@pytest.fixture
def client():
    return TestClient(serve.app)


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_generate_rejects_empty_question(client):
    response = client.post("/generate", json={"question": "  ", "db_schema": "t : c (number)"})
    assert response.status_code == 400


def test_generate_rejects_empty_schema(client):
    response = client.post("/generate", json={"question": "how many?", "db_schema": "   "})
    assert response.status_code == 400


def test_generate_rejects_missing_field(client):
    # Pydantic validation (missing required field) returns 422
    response = client.post("/generate", json={"question": "how many?"})
    assert response.status_code == 422


def test_generate_success_path(client, monkeypatch):
    monkeypatch.setattr(serve, "get_model_and_tokenizer", lambda: (None, None))
    monkeypatch.setattr(serve, "generate_sql", lambda *a, **kw: "SELECT COUNT(*) FROM singer")

    response = client.post(
        "/generate",
        json={"question": "How many singers?", "db_schema": "singer : id (number)"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["sql"] == "SELECT COUNT(*) FROM singer"
    assert body["is_read_only"] is True
    assert body["schema_consistent"] is True


def test_generate_flags_destructive_sql(client, monkeypatch):
    monkeypatch.setattr(serve, "get_model_and_tokenizer", lambda: (None, None))
    monkeypatch.setattr(serve, "generate_sql", lambda *a, **kw: "DROP TABLE singer")

    response = client.post(
        "/generate",
        json={"question": "delete everything", "db_schema": "singer : id (number)"},
    )

    assert response.status_code == 200
    assert response.json()["is_read_only"] is False


def test_generate_flags_schema_inconsistent_sql(client, monkeypatch):
    # catch when the generated query names a table absent from the supplied schema
    monkeypatch.setattr(serve, "get_model_and_tokenizer", lambda: (None, None))
    monkeypatch.setattr(serve, "generate_sql", lambda *a, **kw: "SELECT * FROM nonexistent_table")

    response = client.post(
        "/generate",
        json={"question": "how many?", "db_schema": "singer : id (number)"},
    )

    assert response.status_code == 200
    assert response.json()["schema_consistent"] is False


def test_is_read_only_true_for_plain_select():
    assert serve._is_read_only("SELECT * FROM t") is True


def test_is_read_only_false_for_delete():
    assert serve._is_read_only("DELETE FROM t") is False


def test_is_read_only_false_for_update():
    assert serve._is_read_only("UPDATE t SET x = 1") is False


def test_is_read_only_false_for_drop():
    assert serve._is_read_only("DROP TABLE t") is False


def test_is_read_only_true_for_keyword_inside_string_literal():
    # naive keyword-regex check would false-positive because "update"
    # appears inside string literal not as SQL verb. Structural
    # parsing correctly recognizes this is still a plain SELECT.
    sql = "SELECT * FROM t WHERE note = 'please update your info'"
    assert serve._is_read_only(sql) is True


def test_is_read_only_false_for_stacked_destructive_statement():
    # Every ;-separated statement is checked, not just the first.
    assert serve._is_read_only("SELECT * FROM t; DROP TABLE t;") is False


def test_is_read_only_false_for_unparseable_sql():
    assert serve._is_read_only("not valid sql at all (((") is False


def test_is_read_only_false_for_destructive_statement_hidden_in_cte():
    # Regression test: SQLite's grammar allows a CTE body to be a DML statement, 
    # so this parses as a top-level exp.Select despite deleting data.
    # previous logic that only looked at the top-level statement type would have missed it.
    sql = "WITH x AS (DELETE FROM singer RETURNING *) SELECT * FROM x"
    assert serve._is_read_only(sql) is False


def test_is_read_only_false_for_insert_hidden_in_cte():
    sql = "WITH x AS (INSERT INTO singer VALUES (1, 2) RETURNING *) SELECT * FROM x"
    assert serve._is_read_only(sql) is False


def test_is_read_only_false_for_update_hidden_in_cte():
    sql = "WITH x AS (UPDATE singer SET name = 1 RETURNING *) SELECT * FROM x"
    assert serve._is_read_only(sql) is False


def test_is_read_only_true_for_genuine_cte():
    # real & harmless CTE must not be penalized by the above fix
    sql = "WITH x AS (SELECT * FROM singer) SELECT * FROM x"
    assert serve._is_read_only(sql) is True


def test_schema_consistent_true_when_table_present():
    assert serve._schema_consistent(
        "SELECT * FROM singer", "singer : id (number), name (text)"
    ) is True


def test_schema_consistent_false_when_table_absent():
    assert serve._schema_consistent(
        "SELECT * FROM nonexistent_table", "singer : id (number)"
    ) is False


def test_schema_consistent_case_insensitive():
    assert serve._schema_consistent(
        "SELECT * FROM SINGER", "singer : id (number)"
    ) is True


def test_schema_consistent_whole_word_not_substring():
    # "user" must not match because it's a substring of
    # "users_data", \b word boundaries prevent that false positive.
    assert serve._schema_consistent(
        "SELECT * FROM user", "users_data : id (number)"
    ) is False


def test_schema_consistent_all_tables_must_match_for_a_join():
    # One matching table and one missing table = still inconsistent
    sql = "SELECT * FROM singer JOIN nonexistent_table ON singer.id = nonexistent_table.id"
    assert serve._schema_consistent(sql, "singer : id (number)") is False


def test_schema_consistent_none_for_unparseable_sql():
    # "No signal" (can't tell) is distinct from "checked, and found it's wrong"
    # unparseable SQL must return None not False.
    assert serve._schema_consistent("not valid sql at all (((", "singer : id (number)") is None


def test_schema_consistent_none_for_sql_with_no_tables():
    assert serve._schema_consistent("SELECT 1", "singer : id (number)") is None


def test_schema_consistent_ignores_cte_alias_names():
    # Regression test: sqlglot represents a CTE reference as an exp.Table 
    # node identical in shape to a real table reference. Without excluding 
    # CTE alias names, a valid query using a real schema table inside a CTE would be
    # flagged inconsistent because x isn't a real table name.
    sql = "WITH x AS (SELECT * FROM singer) SELECT * FROM x"
    assert serve._schema_consistent(sql, "singer : id (number)") is True