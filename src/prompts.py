"""
Prompt construction, dependency-free (no torch/datasets imports) so
the pure SQL-scoring logic in evaluate.py can be imported and tested
without requiring the full ML stack to be installed.
"""

SYSTEM_PROMPT = (
    "You are a SQL expert. Given a database schema and a question, "
    "write a single SQL query that answers the question. "
    "Output only the SQL query, with no explanation."
)


def build_messages(question: str, schema: str, gold_sql: str | None = None) -> list[dict]:
    """Build a ChatML messages list. Omit gold_sql to build a generation prompt."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Schema: {schema}\nQuestion: {question}"},
    ]
    if gold_sql is not None:
        messages.append({"role": "assistant", "content": gold_sql})
    return messages
