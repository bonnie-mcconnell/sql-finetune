"""
Safety checks for user-uploaded SQLite files.
"""

MAX_UPLOAD_BYTES = 20 * 1024 * 1024 # 20MB
SQLITE_MAGIC = B"SQLite format 3\x00"

def validate_upload(data: bytes) -> None:
    """
    Raises ValueError on anything that's not an SQLite file. 
    Checked before file reaches sqlite3 to fail fast.
    """
    if len(data) > MAX_UPLOAD_BYTES:
        raise ValueError(f"file too large: {len(data)} bytes, cap is {MAX_UPLOAD_BYTES}")
    if not data.startswith(SQLITE_MAGIC):
        raise ValueError("not a valid SQLite file (magic header mismatch)")
