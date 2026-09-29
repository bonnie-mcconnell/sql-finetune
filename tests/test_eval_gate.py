"""
Eval-CI gate: a threshold based regression guard, seperate from
test_results_reproduce_readme.py exact value pinning, which fails on
any drift, intentional or not. This file tests that accuracy on a fixed, 
seeded, representative subset doesn't fall below a floor. A regression fails,
improving the scorer doesn't.

This catches regressions in scoring code against already-saved generations.
It cannot catch a prompt-format regression which requires regenerating predictions
with the real model. If src/prompts.py changes regenerate results/ and re-verify
accuracy locally before merging.
"""

import random

from src.evaluate import exact_set_match, load_results

HELD_OUT_SEED = 42
HELD_OUT_SIZE = 100

# measured accuracy on this subset is 55% (full set = 59%)
# floor set below this so variance never trips it but regression does
ACCURACY_FLOOR = 0.40


def test_ft_accuracy_above_floor():
    results = load_results("results/finetuned_results.json")
    subset = random.Random(HELD_OUT_SEED).sample(results, HELD_OUT_SIZE)
    correct = [exact_set_match(r["gold"], r["generated"]) for r in subset]
    accuracy = sum(correct) / len(correct)

    assert accuracy >= ACCURACY_FLOOR, (
        f"Fine-tuned accuracy on the held-out subset dropped to {accuracy:.1%}, "
        f"below the {ACCURACY_FLOOR:.0%} floor (real baseline: 55%). This indicates "
        f" a change to scoring logic introduced a regression."
    )