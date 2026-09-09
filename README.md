# Text-to-SQL Fine-Tuning (LoRA, Qwen2.5-Coder-3B)

![CI](https://github.com/bonnie-mcconnell/sql-finetune/actions/workflows/ci.yml/badge.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Fine-tunes `Qwen2.5-Coder-3B-Instruct` on the [Spider](https://yale-lily.github.io/spider)
text-to-SQL benchmark, with a statistically-backed before/after evaluation
and a minimal serving layer, built to demonstrate the full LLM engineering
stack (data pipeline → training → rigorous evaluation → deployment).

## Results

| | Base model (zero-shot) | Fine-tuned (LoRA, epoch 1) |
|---|---|---|
| Exact-set-match accuracy | 21.0% | 50.2% |
| n (full validation set) | 1,034 | 1,034 |

**Improvement: +29.2 points, 95% bootstrap CI [26.0, 32.3]:** this interval excludes zero, so this is a statistically real effect on this evaluation set.

## Table of contents

- [What this measures](#what-this-measures)
- [Error analysis](#error-analysis)
- [Key engineering decisions](#key-engineering-decisions)
- [Efficiency: adapter vs. merged](#efficiency-adapter-vs-merged)
- [Known limitations](#known-limitations)
- [Architecture](#architecture)
- [Running it](#running-it)
- [Repo layout](#repo-layout)

## What this measures

Both models were given the identical zero-shot prompt format (system
instruction + serialized schema + question, see `src/data.py`) and scored
with **exact-set-match**: generated and gold SQL are each parsed into
structural components (SELECT columns, DISTINCT usage, tables, WHERE
conditions, GROUP BY, HAVING conditions, ORDER BY, and LIMIT) via a real
SQL parser (`sqlglot`), and compared as unordered, case/whitespace-
insensitive sets, so `SELECT name, age` and `SELECT age, name` count as
equivalent, but an actually different query does not.

Under this specific zero-shot prompt
and schema format, fine-tuning improved exact-match accuracy from 21% to
50%. This project does not claim that 21% represents the base model's best possible
zero-shot performance - a different prompt format (few-shot examples,
standard `CREATE TABLE` schema serialization) could plausibly raise the
baseline independent of fine-tuning. The comparison between the two numbers
is internally valid (identical conditions, isolating fine-tuning's effect) but
the baseline's absolute value should not be read as a ceiling.

## Error analysis

Every generated/gold pair is also classified into a single primary failure
category (`src/evaluate.py::categorize_error`), checked in order from most
to least structurally fundamental (tables → columns → distinct → where →
group → having → order → limit), reporting only the first mismatch found:

| Category | Base | Fine-tuned |
|---|---|---|
| correct | 21.0% | 50.2% |
| wrong_tables | 49.0% | 28.2% |
| wrong_columns | 15.8% | 4.5% |
| wrong_conditions | 9.2% | 10.3% |
| wrong_distinct | 1.9% | 1.9% |
| wrong_ordering | 1.4% | 1.8% |
| wrong_grouping | 0.9% | 2.3% |
| unparseable | 0.5% | 0.3% |
| wrong_having | 0.4% | 0.3% |

Fine-tuning substantially improved schema grounding (table
selection errors nearly halved, column selection errors dropped by more
than two-thirds) but had essentially no effect on WHERE-clause condition
correctness (9.2% → 10.3%, within noise) or DISTINCT usage (1.9% → 1.9%,
unchanged). This makes sense as schema-paired training
examples directly teach "which table/column for this kind of question",
while getting filter logic, deduplication, and other query-semantic
details right is a different and harder skill that this fine tuning data didn't improve.

**Methodological caveat:** because only the first mismatch is reported per
example, a category's count can shift simply because fewer examples are
now being caught earlier by a table/column mismatch, which surfaces
previously-masked mismatches further down the ladder for the first time, but not necessarily because that specific error type got objectively worse in the data.
Read the wrong_grouping/wrong_ordering upticks with that in mind.

## Key engineering decisions

**QLoRA → plain LoRA, mid-project, backed by measurement.** Initially
implemented QLoRA (4-bit NF4 quantization + LoRA adapters), which is the standard
approach for fine-tuning on constrained hardware. Direct profiling showed
the quantized model used only ~2.0GB of the training GPU's 15GB, so memory constraints weren't actually an issue on this hardware. Quantization was also the root cause of a
recurring `bitsandbytes`/mixed-precision dtype instability during
training (a stray internal `bfloat16` tensor crashing PyTorch's fp16
gradient scaler, regardless of every model- and trainer-level dtype
setting tried). Switching to plain LoRA on an unquantized fp16 model
(`transformers.AutoModelForCausalLM`, no `BitsAndBytesConfig`) resolved
the crash and cut per-step training time roughly 5-6x by
removing per-step dequantization overhead (~55-63s/step quantized →
~10.3s/step unquantized, measured on the same hardware and batch config).

**Checkpoint selection via validation loss, not the last epoch.** Training
ran 3 epochs with per-epoch evaluation:

| Epoch | Train loss | Validation loss |
|---|---|---|
| 1 | 0.093 | 0.228 |
| 2 | 0.039 | 0.289 |
| 3 | 0.014 | 0.361 |

Training loss falls monotonically, validation loss rises after epoch 1, which is textbook overfitting. The epoch-1 checkpoint was selected as the final model, not the epoch-3 (last) checkpoint. All reported results use the epoch-1 checkpoint.

**Train/validation database overlap: 0/166.** Verified directly
(`len(set(train_db_ids) & set(val_db_ids)) == 0`). The validation set
tests generalization to entirely unseen databases, not just unseen
questions on familiar schemas.

**Loss masking.** Only the assistant's SQL response tokens contribute to
the training loss (`labels` set to `-100` for every prompt/schema token,
following PyTorch's `CrossEntropyLoss` `ignore_index` convention). The
model is graded on generating correct SQL, not on reproducing the schema
text it was given.

**SQL parsing bugs in scoring:** The original `get_component_sets` implementation compared WHERE
conditions by calling `.flatten()` on the whole `Where` node rather than
on its condition (`.this`). `.flatten` returns the entire clause as an
un-split opaque string, meaning WHERE clauses were never actually
compared as an unordered set of conditions despite the code and docs
claiming exactly that (it happened to still catch operator/connective
differences, since those produce different literal text, but silently
failed the one case it was meant to handle: `age > 20 AND country = 'X'`
vs. the logically-identical `country = 'X' AND age > 20` scored as not
a match). Separately, the scorer had no concept of `LIMIT`, `DISTINCT`, or
`HAVING` at all. Fixed (explicit AND-conjunct
splitting that preserves comparison operators and never decomposes OR and
added `distinct`/`having`/`limit` as scored components) and re-run against
the real, saved 1,034-example results in `results/`: base accuracy moved
21.95%→21.0%, fine-tuned 51.26%→50.2% — both down by roughly the same
~1 point, consistent with correcting a symmetric measurement bias rather
than something that happened to favor one model. The statistically
significant improvement survives the correction unchanged. The numbers
throughout this README are post-fix.
`tests/test_results_reproduce_readme.py` pins them against regression.

**Tokenizer versioning:** Every
automated test for `src/serve.py` mocks the model and tokenizer to avoid needing a GPU in CI, which means they verify the
API's request/response contract, but can never catch a problem in the
actual model-loading code. Running an
actual server against the real merged model showed that `tokenizer.apply_chat_template` failed
with `chat_template is not set`, despite every mocked test passing. The error was that the tokenizer had
been saved by an older `transformers` version, and a newer one (installed
via `pip install -U` with only floor-pinned versions in
`requirements.txt`) expects the chat template stored differently. Fixed by loading the
tokenizer from the original model repo rather than the saved local copy
(the chat template is a property of the base model, untouched by LoRA
fine-tuning or merging, so no need to load it from
a local save in the first place), and by switching `requirements.txt` to
exact-pinned versions (`requirements-dev.txt` for test/lint tools).

## Efficiency: adapter vs. merged

Single-example generation latency, measured on the same GPU, same 50-example sample (first 5 discarded as warmup), greedy decoding:

| Model | Latency (s/example) |
|---|---|
| Base (no adapter) | 2.439 |
| Fine-tuned, unmerged LoRA adapter | 5.408 (+121.8%) |
| Fine-tuned, merged (`merge_and_unload()`) | 2.583 |

An unmerged LoRA adapter **more than doubles** per-example latency versus
the base model on this hardware, because every adapted layer pays for a frozen base-weight matmul *and* a separate small adapter matmul, on every forward pass, at every generation step. Merging (`model.merge_and_unload()`, which computes `W_base + (B @ A * scaling)` once and folds it into ordinary weights) recovers 95.2% of that overhead, landing within ~6% of the base model's latency. Always merge the model before deployment to decrease unneccessary latency.
`src/serve.py` loads the merged model for this reason.

**The read-only-SQL guardrail as an allowlist checking the whole tree:** `_is_read_only`
parses generated SQL with `sqlglot` and checks that every statement is
structurally a `SELECT`, rather than scanning for a list of "dangerous"
keywords (`DROP`, `DELETE`, ...). A keyword denylist is only as safe as the list is complete, and it false-positives on
harmless SQL where a banned word appears inside a string literal (e.g.
`WHERE note = 'please update your info'`). Checking statement type
structurally avoids both of these issues and correctly rejects stacked
`SELECT ...; DROP TABLE ...;` statements too. However this top-level-only
structural checking also has a gap where SQLite's grammar accepts a DML
statement as a CTE body, so
`WITH x AS (DELETE FROM t RETURNING *) SELECT * FROM x` parses as a
top-level `exp.Select` despite deleting data. `_is_read_only` also walks
the full tree (`find_all` across `Insert`/`Update`/`Delete`/`Drop`/
`Create`/`Alter`/`Merge`/etc.) so a write hidden inside a CTE is caught
too. A plain subquery position (`SELECT * FROM (DELETE ...) AS t`)
doesn't need this, since SQLite's grammar rejects it/sqlglot never parses it, but a CTE body is permissive enough to need an
explicit check.

**Weak schema consistency check:**
`/generate` also returns `schema_consistent`: whether every table the
generated SQL references appears (as a whole word) somewhere in the
`db_schema` string the caller supplied. This catches the simplest hallucinations: a generated query naming a table that
was never mentioned in the schema at all. `db_schema` is a free-form string with no fixed
serialization enforced here, so the check is a whole-word,
case-insensitive substring search per table name, not a structural
schema parse. A table name appearing
somewhere in the text doesn't guarantee it was declared there as a table vs as a column, and it returns `null` rather than `false`
when the SQL doesn't parse or references no tables at all, since "no
signal" and "inconsistent" should be treated differently by the caller. A real per-database schema (structured
table/column lists, not a string) would let this be a precise check
instead of a heuristic one. CTE alias names (e.g. the `x` above) are
excluded from the referenced-tables set because sqlglot
represents a CTE reference identically to a real table reference.

## Known limitations

- **`get_component_sets`'s "tables" set doesn't distinguish a CTE alias
  from a real table.** sqlglot represents a CTE reference (e.g. the `x`
  in `WITH x AS (...) SELECT * FROM x`) as an `exp.Table` node
  structurally identical to a real table reference, so a generated query
  that wraps a correct answer in a CTE would currently score as
  `wrong_tables` against a gold query that doesn't use one, even though
  the two are semantically identical. Not currently affecting the
  numbers because none of the 2,068 committed generations (base +
  fine-tuned) use a CTE, but it's a latent gap in the scorer. `src/serve.py`'s
  `_schema_consistent` (a separate function) excludes CTE aliases
  correctly but `get_component_sets` doesn't yet.
- **No execution-based accuracy.** The official Spider release includes
  real SQLite databases enabling execution-match scoring (run the SQL,
  compare returned rows) in addition to structural matching. Adding execution accuracy on a
  subset of databases (sourced from an alternate, reliably-hosted mirror) is a next step.
- **Possible benchmark contamination.** Spider is a long-public dataset.
  Qwen2.5-Coder's pretraining corpus may include some of it, meaning
  baseline zero-shot performance could reflect memorization rather
  than pure schema-grounded reasoning. This doesn't invalidate the
  before/after comparison (both models are equally exposed to whatever
  contamination exists), but it does mean the baseline number shouldn't be
  read as a clean measure of "true" zero-shot ability.
- **Evaluated with greedy decoding only.** Deliberate, for reproducibility
  of the paired statistical comparison. Haven't tested with
  beam search or sampling-based decoding yet.
- **`categorize_error`'s first-match-wins ordering** (see Error analysis)
    means category-level shifts should be read as directional signal, not
  as an actual count of every error type present.
- **Checked for generation truncation, found no evidence of it.**
  `max_new_tokens=128` is a hard cap on `generate_sql`. A generation cut
  off mid-query would fail the SQL scoring structurally rather than
  raising anything. Character-length distribution of the committed
  `results/` generations: max 437 chars (base) / 431 chars (fine-tuned),
  p99 ~310-316 chars, which is within these 128 tokens.

## Architecture

```
Spider (xlangai/spider)          spider-schema (richardr1126/spider-schema)
   question + gold SQL  ⟶ join by db_id ⟵  table/column schema
              │
              ▼
   ChatML prompt (system + schema + question) -> tokenize -> mask prompt tokens
              │
              ▼
   Qwen2.5-Coder-3B-Instruct (fp16) + LoRA adapters (r=16, 0.96% trainable)
              │
              ▼
   SFTTrainer, 3 epochs, checkpoint selected by validation loss (epoch 1)
              │
        ┌─────┴─────┐
        ▼           ▼
  generate_sql   generate_sql
  (base model)   (fine-tuned)
        │           │
        └─────┬─────┘
              ▼
   exact_set_match + categorize_error (sqlglot-based structural comparison)
              │
              ▼
   paired bootstrap 95% CI on accuracy delta
              │
              ▼
   merge_and_unload() → merged model → latency benchmark (base vs. adapter
                                        vs. merged) → FastAPI /generate endpoint
```

## Running it

```bash
pip install -r requirements.txt

# Train (writes checkpoints/checkpoint-<step>/ per epoch + checkpoints/final/)
python -m src.train

# Serve (expects a merged model at checkpoints/merged/, produced via
# model.merge_and_unload().save_pretrained("checkpoints/merged"))
uvicorn src.serve:app --reload
# POST /generate {"question": "...", "db_schema": "table : col (type), ..."}

# or containerized (mounting merged model into the container):
docker build -t sql-finetune .
docker run -p 8000:8000 -v $(pwd)/checkpoints:/app/checkpoints sql-finetune

# Tests (no GPU required, tests structural SQL comparison, prompt/masking logic, 
# the FastAPI routing/validation contract, and a regression check that the
# real results in results/ reproduce every number claimed in this README)
pip install -r requirements-ci.txt
pip install -r requirements-dev.txt
pytest tests/ -v

# Lint (matches CI)
ruff check .
```

**Dependency files** `requirements.txt`
is the full project (training + serving). `requirements-serve.txt` is
what the Docker image installs - everything `src/serve.py`
needs to run, and nothing `src/train.py`-only. `requirements-ci.txt` is what CI installs, for tests instead of Docker, plus `datasets` (needed
transitively because `test_data.py` imports `src.data`) but never torch
(tests import it lazily).
`requirements-core.txt` is the package list common to the other two, so
a shared version only needs to change in one place.

## Repo layout

```
src/
  data.py       # Spider + schema loading, prompt formatting, loss masking
  train.py      # LoRA config, SFTTrainer setup and training loop
  evaluate.py   # generation, cleaning, exact-set-match, error categorization,
                # paired bootstrap
  serve.py      # FastAPI serving layer (merged model, read-only-SQL guardrail)
results/
  base_results.json, finetuned_results.json  # full 1,034-example generations
                            
tests/
  test_data.py                    # prompt formatting + masking-boundary correctness
  test_evaluate.py                # exact-set-match + error categorization, incl. the
                                   # real over-join example that motivated structural
                                   # (not text) comparison
  test_serve.py                   # API contract + read-only-SQL guardrail
  test_results_reproduce_readme.py  # regression guard: real results -> exact
                                     # numbers claimed in this README
Dockerfile          # containerizes src/serve.py
LICENSE             # MIT
pyproject.toml      # ruff lint config (enforced in CI)
results/
  base_results.json, finetuned_results.json  # full 1,034-example generations
```
