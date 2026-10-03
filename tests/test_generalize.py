import sqlite3
from unittest.mock import patch

from fastapi.testclient import TestClient

from src.serve import app

client = TestClient(app)


def _make_sqlite_bytes(tmp_path):
    path = tmp_path / "test.sqlite"
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE t (x INT, y TEXT)")
    con.executemany("INSERT INTO t VALUES (?, ?)", [(1, "a"), (2, "b")])
    con.commit()
    con.close()
    return path.read_bytes()


@patch("src.serve.get_model_and_tokenizer")
@patch("src.serve.generate_sql")
def test_generalize_executes(mock_generate, mock_get_model, tmp_path):
    mock_get_model.return_value = (None, None)
    mock_generate.return_value = "SELECT * FROM t"

    response = client.post(
        "/generalize",
        data={"question": "show_everything"},
        files={"db_file": ("test.sqlite", _make_sqlite_bytes(tmp_path))},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["is_read_only"] is True
    assert sorted(body["result"]) == [[1, "a"], [2, "b"]]
    assert body["error"] is None


@patch("src.serve.get_model_and_tokenizer")
@patch("src.serve.generate_sql")
def test_generalize_rejects_non_sqlite_upload(mock_generate, mock_get_model):
    mock_get_model.return_value = (None, None)
    response = client.post(
        "/generalize",
        data={"question": "anything"},
        files={"db_file": ("fake.sqlite", b"not a real database")},
    )
    assert response.status_code == 400


@patch("src.serve.get_model_and_tokenizer")
@patch("src.serve.generate_sql")
def test_generalize_doesnt_execute_destructive_sql(mock_generate, mock_get_model, tmp_path):
    mock_get_model.return_value = (None, None)
    mock_generate.return_value = "DELETE FROM t"

    response = client.post(
        "/generalize",
        data={"question": "delete everything"},
        files={"db_file": ("test.sqlite", _make_sqlite_bytes(tmp_path))},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["is_read_only"] is False
    assert body["result"] is None
    assert "not executed" in body["error"]