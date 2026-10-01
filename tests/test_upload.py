import sqlite3

import pytest

from src.upload import MAX_UPLOAD_BYTES, SQLITE_MAGIC, validate_upload


def test_validate_upload_rejects_non_sqlite_file():
    with pytest.raises(ValueError, match="magic header"):
        validate_upload(b"not a real database")


def test_validate_upload_rejects_oversized_file():
    data = SQLITE_MAGIC + b"x" * MAX_UPLOAD_BYTES
    with pytest.raises(ValueError, match="too large"):
        validate_upload(data)


def test_validate_upload_accepts_sqlite(tmp_path):
    db_path = tmp_path / "test.sqlite"
    con = sqlite3.connect(db_path)
    con.execute("CREATE TABLE t (x)")
    con.commit()
    con.close()
    validate_upload(db_path.read_bytes())


def test_validate_upload_max_size():
    data = SQLITE_MAGIC + b"x" * (MAX_UPLOAD_BYTES - len(SQLITE_MAGIC))
    validate_upload(data)