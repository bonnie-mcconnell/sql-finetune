"""
Small FastAPI serving layer for the fine-tuned text-to-SQL model.

Loads the merged model (merged model has less latency than adapter). 
Generates SQL but never executes it. Returned with an `is_read_only` 
flag, executing it against a real database is entirely the caller's decision.

Run: uvicorn src.serve:app --reload
"""

import re

import sqlglot
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sqlglot import exp

from .evaluate import generate_sql

# torch and transformers are imported lazily, which keeps this file 
# usable/testable in CI without ML stack installed

MODEL_PATH = "checkpoints/merged"  # output of model.merge_and_unload().save_pretrained(...)
TOKENIZER_NAME = "Qwen/Qwen2.5-Coder-3B-Instruct"  # the base model repo, not MODEL_PATH

app = FastAPI(title="Text-to-SQL Service")

_model = None
_tokenizer = None


def get_model_and_tokenizer():
    """Lazy-load on first request rather than at import time, so tests
    that don't need the model (e.g. health check) don't pay GPU startup
    cost and the API process starts up quickly."""
    global _model, _tokenizer
    if _model is None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # Tokenizer loaded from the original model repo, not MODEL_PATH.
        # LoRA fine-tuning and merging never touch the tokenizer or its
        # chat template, it's unchanged from the base model.
        # Loading from the canonical source instead of a saved local copy
        # avoids a bug observed in testing: a tokenizer saved by
        # one transformers version can silently lose its chat
        # template when loaded by a much newer one (the library changed
        # where the template is expected to live between versions)
        # loading fresh from the Hub avoids this.
        _tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
        _model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH, device_map="auto", dtype=torch.float16
        )
    return _model, _tokenizer


class GenerateRequest(BaseModel):
    question: str
    db_schema: str  # avoid shadowing built in schema


class GenerateResponse(BaseModel):
    sql: str
    is_read_only: bool
    schema_consistent: bool | None


_UNSAFE_EXPR_TYPES = (
    exp.Insert,
    exp.Update,
    exp.Delete,
    exp.Drop,
    exp.Create,
    exp.Alter,
    exp.TruncateTable,
    exp.Merge,
    exp.Command,  # sqlglot's catch-all for statement types it has no
    # specific class for (e.g. some PRAGMA/EXEC-style statements) included defensively
)


def _is_read_only(sql: str) -> bool:
    """
    True only if every parsed statement in `sql` is a SELECT and none 
    of its nodes are a write of DDL statement (including inside CTE).

    Allowlist. Checks all ;-separated statements (rejects a stacked
    `SELECT ...; DROP TABLE ...;`), and treats unparseable SQL as not
    read-only.

    Tree-wide check exists because `WITH x AS (DELETE FROM t RETURNING *) SELECT * FROM x`
    parses as a top-level `exp.Select`, as SQLite's CTE grammar accepts a
    DML statement as a CTE body, so checking only `isinstance(stmt, exp.Select)` 
    at top level would allow a destructive statement through inside a CTE definition. A
    plain subquery position (`SELECT * FROM (DELETE ...) AS t`) doesn't
    have this problem because SQLite's grammar rejects it so
    sqlglot never parses it in the first place, but a CTE body is
    permissive enough to allow it, so it needs an explicit check.
    """
    try:
        statements = sqlglot.parse(sql, read="sqlite")
    except Exception:  # noqa: BLE001 
        return False
    if not statements:
        return False
    for stmt in statements:
        if stmt is None or not isinstance(stmt, exp.Select):
            return False
        if any(stmt.find_all(*_UNSAFE_EXPR_TYPES)):
            return False
    return True


def _tables_referenced(sql: str, dialect: str = "sqlite") -> frozenset[str]:
    """
    Returns set of every table name sqlglot finds in `sql` lowercased.
    CTE alias names excluded (e.g `x` in `WITH x as (...) SELECT * FROM x`)
    because they will never appear in schema's table list. Uses `tree.find(exp.With)`
    structural search not `tree.args.get("with")` raw dict-key lookup for version
    control. Raises on unparseable SQL, callers handle this.
    """
    tree = sqlglot.parse_one(sql, read=dialect)
    with_node = tree.find(exp.With)
    cte_names = (
        {cte.alias.lower() for cte in with_node.expressions if cte.alias} if with_node else set()
    )
    return frozenset(
        t.name.lower()
        for t in tree.find_all(exp.Table)
        if t.name and t.name.lower() not in cte_names
    )


def _schema_consistent(sql: str, db_schema: str) -> bool | None:
    """
    Best-effort check that every table `sql` references appears in `db_schema`.
    Format agnostic, rather than assuming a specific delimiter convention and 
    parsing stucturally, which would break when caller's schema string is shaped 
    differently, this does a whole-word case-insensitive substring search per 
    referenced table name. `\\b` word boundraries avoid table name amtching as 
    a substring of an unrelated word.

    This only catches obvious errors: table name appearing in `sql` doesn't
    guarantee it was a table in the schema, could be a column name. A structured
    per-database schema would make this a precise check not a heuristic one.

    Returns None for unparseable SQL or SQL that references no tables.
    """
    try:
        tables = _tables_referenced(sql)
    except Exception: # noqa: BLE001
        return None
    if not tables:
        return None
    schema_lower = db_schema.lower()
    return all(re.search(rf"\b{re.escape(t)}\b", schema_lower) for t in tables)


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    if not req.question.strip() or not req.db_schema.strip():
        raise HTTPException(status_code=400, detail="question and db_schema must be non-empty")

    model, tokenizer = get_model_and_tokenizer()
    sql = generate_sql(model, tokenizer, req.question, req.db_schema)
    return GenerateResponse(
        sql=sql, 
        is_read_only=_is_read_only(sql),
        schema_consistent=_schema_consistent(sql, req.db_schema),
    )


@app.get("/health")
def health():
    return {"status": "ok"}
