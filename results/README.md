# Results

`base_results.json` and `finetuned_results.json` are the full
1,034-example validation-set generations from the base and fine-tuned
models respectively. Each is a JSON list of
`{"db_id", "question", "gold", "generated"}` records.

`tests/test_results_reproduce_readme.py` runs these through the actual
scoring code in `src/evaluate.py` and asserts the results match every
headline number in the top-level README (accuracy, confidence interval,
error breakdown), acting as a regression guard, not a model-quality test: if
`exact_set_match` or `categorize_error`'s logic ever changes, this test
fails immediately rather than letting the README silently drift out of
sync with what the code actually computes.

To reproduce the numbers yourself:

```python
from src.evaluate import load_results, exact_set_match, error_breakdown, paired_bootstrap
import numpy as np

base_results = load_results("results/base_results.json")
finetuned_results = load_results("results/finetuned_results.json")

base_correct = [exact_set_match(r["gold"], r["generated"]) for r in base_results]
ft_correct = [exact_set_match(r["gold"], r["generated"]) for r in finetuned_results]

print("Base accuracy:", np.mean(base_correct))
print("Fine-tuned accuracy:", np.mean(ft_correct))
print("Diff, 95% CI:", paired_bootstrap(base_correct, ft_correct))
print(error_breakdown(base_results))
print(error_breakdown(finetuned_results))
```
