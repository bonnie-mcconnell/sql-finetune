"""
Evaluation utilities: SQL generation, cleaning, structural exact-match
scoring, error categorization, and paired-bootstrap statistical comparison.
"""

import json
import re
from collections import Counter

import numpy as np
import sqlglot
from sqlglot import exp

from .prompts import build_messages

# torch is imported lazily, inside generate_sql() only so 
# everything else can be tested in CI without it installed


def clean_sql(raw_output: str) -> str:
    """
    Strips a markdown code fence (```sql ... ``` or ``` ... ```) if present.

    Instruction-tuned models often wrap code in markdown fences from
    pretraining, regardless of the explicit "no explanation" system
    prompt. This is applied to every generation, not conditionally, 
    since the fence is invalid SQL syntax that would break any downstream parser.
    """
    match = re.search(r"```(?:sql)?\s*(.*?)```", raw_output, re.DOTALL)
    if match:
        raw_output = match.group(1)
    return raw_output.strip().rstrip(";").strip()


def generate_sql(model, tokenizer, question: str, schema: str, max_new_tokens: int = 128) -> str:
    """
    Greedy-decode a SQL query for a given question + schema.

    Greedy (do_sample=False) deterministic decoding is used for
    the paired base-vs-fine-tuned statistical comparison which requires
    reproducible outputs on identical inputs. Sampling-based decoding
    would undermine this with unrelated randomness.
    """
    import torch  # lazy import

    messages = build_messages(question, schema)
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    generated_ids = output_ids[0][inputs["input_ids"].shape[1] :]
    raw = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return clean_sql(raw)


def _normalize_expr(e, dialect: str = "sqlite") -> str:
    return e.sql(dialect=dialect).lower().replace(" ", "")


def _split_and_conditions(condition: exp.Expression) -> list[exp.Expression]:
    """
    Recursively split a WHERE/HAVING condition into its top-level
    AND-conjuncts, treating each conjunct (including any nested OR
    expression within it) as one atomic unsplit unit.

    AND is commutative so split on AND into an unordered set. OR is
    not decomposed because "A OR B" is different condition from "A AND B" 
    (either-one vs. must-satisfy-both), so an OR expression is kept whole 
    rather than risking it being confused with an AND of the same two conditions.
    """
    if isinstance(condition, exp.And):
        return _split_and_conditions(condition.this) + _split_and_conditions(
            condition.expression
        )
    return [condition]


def get_component_sets(sql: str, dialect: str = "sqlite") -> dict:
    """
    Parse SQL into structural component sets: select columns, DISTINCT,
    tables, WHERE conditions, GROUP BY, HAVING, ORDER BY, and LIMIT, each
    an unordered normalized set (or single normalized value for
    DISTINCT/LIMIT), so two queries that differ only in column order,
    case, or whitespace compare as identical. Uses sqlglot's SQL
    parser rather than text/regex splitting, which would break on commas 
    inside string literals, function calls, and subqueries.

    WHERE and HAVING conditions are split on top-level, only reordered 
    AND-conjuncts compare equal, reordered operands of an OR do not 
    (OR is kept as one atomic unit, not decomposed), and a changed comparison 
    operator (> vs < vs =) always produces a different set, since operators 
    are never stripped away.

    Raises on malformed or non-SELECT SQL. `exact_set_match` and
    `categorize_error` catch these and turn them into a `False`/`"unparseable"` 
    result. Call this function directly only when you need a malformed input to 
    raise rather than be silently treated as "not a match".
    """
    tree = sqlglot.parse_one(sql, read=dialect)

    select = tree.find(exp.Select)
    select_exprs = frozenset(_normalize_expr(e, dialect) for e in select.expressions)
    distinct = select.args.get("distinct") is not None

    tables = frozenset(_normalize_expr(t, dialect) for t in tree.find_all(exp.Table))

    where = tree.find(exp.Where)
    where_conds = (
        frozenset(_normalize_expr(c, dialect) for c in _split_and_conditions(where.this))
        if where
        else frozenset()
    )

    group = tree.args.get("group")
    group_cols = (
        frozenset(_normalize_expr(g, dialect) for g in group.expressions) if group else frozenset()
    )

    having = tree.find(exp.Having)
    having_conds = (
        frozenset(_normalize_expr(c, dialect) for c in _split_and_conditions(having.this))
        if having
        else frozenset()
    )

    order = tree.args.get("order")
    order_cols = (
        frozenset(_normalize_expr(o, dialect) for o in order.expressions) if order else frozenset()
    )

    limit = tree.args.get("limit")
    limit_val = _normalize_expr(limit, dialect) if limit else None

    return {
        "select": select_exprs,
        "distinct": distinct,
        "tables": tables,
        "where": where_conds,
        "group": group_cols,
        "having": having_conds,
        "order": order_cols,
        "limit": limit_val,
    }


def exact_set_match(gold_sql: str, gen_sql: str) -> bool:
    """
    Structural SQL comparison: True if gold and generated SQL have the same
    SELECT columns, DISTINCT usage, tables, WHERE conditions, GROUP BY,
    HAVING conditions, ORDER BY, and LIMIT value, each compared as an
    unordered, case/whitespace-insensitive set (DISTINCT/LIMIT as single
    normalized values, since they aren't multi-item clauses). Returns
    False (never raises) if either SQL string fails to parse.

    Note: this measures structural/set equivalence, not execution
    equivalence against a real database.
    """
    try:
        return get_component_sets(gold_sql) == get_component_sets(gen_sql)
    except Exception:  # noqa: BLE001 
        # malformed SQL can raise sqlglot's own parse errors or AttributeError 
        # (e.g. select.expressions on a None .find(exp.Select) result for SQL
        # with no SELECT at all)
        return False


def categorize_error(gold_sql: str, gen_sql: str) -> str:
    """
    Classify a (gold, generated) pair into one primary failure category.

    Checked in order from most to least structurally fundamental (tables
    -> columns -> distinct -> where -> group -> having -> order -> limit).
    A generated query can differ from gold in multiple components
    simultaneously, this reports only the first, most severe mismatch
    found. See README for the resulting caveat on interpreting category 
    shifts between two models.
    """
    try:
        gold_c = get_component_sets(gold_sql)
        gen_c = get_component_sets(gen_sql)
    except Exception:  # noqa: BLE001 
        # same multiple error catching as above
        return "unparseable"

    if gold_c == gen_c:
        return "correct"
    if gold_c["tables"] != gen_c["tables"]:
        return "wrong_tables"
    if gold_c["select"] != gen_c["select"]:
        return "wrong_columns"
    if gold_c["distinct"] != gen_c["distinct"]:
        return "wrong_distinct"
    if gold_c["where"] != gen_c["where"]:
        return "wrong_conditions"
    if gold_c["group"] != gen_c["group"]:
        return "wrong_grouping"
    if gold_c["having"] != gen_c["having"]:
        return "wrong_having"
    if gold_c["order"] != gen_c["order"]:
        return "wrong_ordering"
    if gold_c["limit"] != gen_c["limit"]:
        return "wrong_limit"
    return "other"


def error_breakdown(results: list[dict]) -> Counter:
    """Category counts across a list of {"gold", "generated"} result dicts."""
    return Counter(categorize_error(r["gold"], r["generated"]) for r in results)


def paired_bootstrap(base_correct, ft_correct, n_boot: int = 10000, seed: int = 42):
    """
    Paired-bootstrap 95% confidence interval on the accuracy difference
    (fine-tuned - base).

    Resamples question indices with replacement n_boot times, preserving 
    the pairing (both models are scored on the same simulated resample of 
    questions), which is more statistically powerful than an unpaired comparison 
    since it cancels out question-level difficulty variance shared by both models. 
    Returns (mean_diff, ci_low, ci_high) from the resulting distribution's 2.5th
    and 97.5th percentiles.
    """
    base_correct = np.array(base_correct)
    ft_correct = np.array(ft_correct)
    n = len(base_correct)
    rng = np.random.default_rng(seed)

    diffs = np.array(
        [
            ft_correct[idx].mean() - base_correct[idx].mean()
            for idx in (rng.integers(0, n, n) for _ in range(n_boot))
        ]
    )
    ci_low, ci_high = np.percentile(diffs, [2.5, 97.5])
    return float(diffs.mean()), float(ci_low), float(ci_high)


def load_results(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)
