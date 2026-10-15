# Evaluations

`aiforge.evals` scores module outputs against a test set, so you can see
what a prompt, model or provider change does before you ship it. It runs
any AI Forge module (or any Python function) over a dataset with bounded
concurrency. Each output is scored, and you get pass rates, cost, tokens
and latency. A new run can be compared with a saved baseline to flag
regressions. The `aiforge eval run` command exits non-zero on failure, so
it can gate CI.

It covers the core of tools like Promptfoo, LangSmith evals and DeepEval,
but stays small: no new dependencies, no server, and plain JSON reports.

---

## Quick start

```python
from aiforge import Classifier
from aiforge.evals import evaluate, label_match

cases = [
    {"input": "The app crashes when I upload a PNG", "expected": "bug"},
    {"input": "Could you add dark mode?", "expected": "feature"},
    {"input": "How do I reset my password?", "expected": "question"},
]

report = evaluate(
    Classifier(labels=["bug", "feature", "question"]),
    cases,
    scorers=[label_match()],
    min_pass_rate=0.9,
)

report.print()                   # rich table + failing cases
report.save("evals/report.json") # JSON for CI artifacts / baselines
assert report.passed
```

`evaluate()` is synchronous and safe to call from inside a running event
loop. Use `await aevaluate(...)` in async code.

---

## Datasets

A case has an `input`, an optional `expected` value, optional per-case
`kwargs` (forwarded to the target) and free-form `metadata`. `id` is
optional and defaults to the row index. Give cases stable ids if you
compare reports over time.

| Source | How |
|---|---|
| Python | `Dataset.from_list([{"input": ..., "expected": ...}])`, or pass the list straight to `evaluate` |
| JSONL | One object per line, blank lines skipped |
| JSON | A list of cases, or `{"cases": [...]}` |
| YAML | A list of cases, or `cases:` mapping |
| CSV | An `input` column, plus optional `id`, `expected`, `kwargs`, `metadata` |

`Dataset.load(path)` (or passing a path to `evaluate`) picks the parser
from the extension.

```yaml
# evals/tickets.yaml
cases:
  - id: crash-png
    input: The app crashes when I upload a PNG
    expected: bug
    metadata: {area: uploads}
  - id: multi
    input: Dark mode is broken and I'd like a high-contrast theme
    expected: [bug, feature]          # multi-label
```

In CSV files, `expected` values starting with `[` or `{` are parsed as
JSON. `kwargs` and `metadata` cells hold JSON objects. Any other column is
copied into `metadata`.

---

## Targets

| Target | What is called |
|---|---|
| `Classifier` | `aclassify(input, **kwargs)` |
| `Extractor` | `aextract` |
| `Summarizer` | `asummarize` |
| `Generator` | `agenerate` |
| `Moderator` | `acheck` |
| `Translator` | `atranslate` |
| `Vision` | `aanalyze` |
| `DocQA` | `aask` |
| `SemanticSearch` | `aquery` |
| `ChatAgent` | `achat` |
| Other `BaseModule` | `arun` |
| Any callable | `fn(input, **kwargs)`. Async functions are awaited; sync ones run in a worker thread |

Pass `method="sentiment"` (or `--method sentiment`) to call another
method. Its `a`-prefixed async twin is used when it exists.

For every case the report records:

- the output
- the error, if any (as `"Type: message"`)
- `cost`, `input_tokens` and `output_tokens`, read from the
  `ModuleResponse`, or from a returned object or dict that has those fields
- the wall-clock latency

A target error never aborts the run. The case fails every scorer with
reason `target error: ...`.

```python
report = evaluate(
    Extractor(output_schema=Invoice),
    "evals/invoices.jsonl",
    scorers=[json_match(), numeric_close(field="total", tolerance=0.01)],
    concurrency=8,          # cases in flight at once
    timeout=30,             # per-case timeout, seconds
    call_kwargs={},         # sent on every call; per-case kwargs win
)
```

!!! note
    `ChatAgent` keeps conversation history between calls. Use
    `concurrency=1` and call `agent.reset()` in a wrapper function if each
    case should start fresh.

---

## Scorers

Each scorer returns a `ScoreResult(name, score, passed, reason, cost)`,
where `score` is between 0 and 1. A scorer that compares against a
reference takes an optional explicit value. Without one, it uses the
case's `expected`. Every factory accepts `name=`, so you can use the same
scorer twice in one run.

| Scorer | Passes when | Score |
|---|---|---|
| `exact_match(expected=None, ignore_case=False, normalize_whitespace=True)` | output text equals expected (or any item of a list) | 1/0 |
| `contains(value=None, ignore_case=False, require_all=True)` | substring(s) present | fraction found |
| `not_contains(value=None, ignore_case=False)` | none of the substrings present | fraction absent |
| `regex(pattern=None, flags=0, full_match=False)` | `re.search` (or `fullmatch`) matches | 1/0 |
| `json_match(expected=None, subset=True, fields=None, ignore_case=False)` | every compared field matches (nested dicts and lists recurse) | fraction of fields matching |
| `numeric_close(expected=None, tolerance=1e-6, relative=False, field=None)` | the absolute (or relative) difference is within the tolerance | 1/0 |
| `label_match(expected=None, ignore_case=True)` | top label equals expected. With a list, the label sets are equal | 1/0, or Jaccard overlap |
| `embedding_similarity(expected=None, threshold=0.8, embedder=None, provider=None, model=None)` | cosine similarity ≥ threshold | similarity |
| `llm_judge(rubric, llm=None, provider=None, model=None, threshold=None)` | the judge says so (or score ≥ threshold) | judge score |

Scorers read the output in one of two ways:

- **Text** (`exact_match`, `contains`, `not_contains`, `regex`): uses
  `ModuleResponse.to_text()`. That is the label for `ClassifyResponse`,
  the summary for `SummarizeResponse`, and so on.
- **Structured data** (`json_match`, `numeric_close(field=...)`): uses
  `ExtractResponse.data`, dicts, Pydantic models (without the usage
  fields) or JSON strings.

### LLM judge

```python
from aiforge.evals import llm_judge

polite = llm_judge(
    "The reply is polite, answers the question and does not invent facts.",
    provider="anthropic",          # judge separately from the target
    model="claude-haiku-4-5",
    threshold=0.7,                 # optional; default uses the judge's passed flag
)
```

The judge sees the rubric, the case input, the expected reference (if
any) and the output. It must answer with a fixed schema:
`JudgeVerdict(score: float, passed: bool, reason: str)`.

- When the provider supports native structured output, the call uses
  `response_format=JudgeVerdict`.
- If that call fails with a non-transient error, or the reply cannot be
  parsed, it retries once without `response_format` and parses the JSON
  from the text.
- A reply that still cannot be parsed scores 0, with the raw text as the
  reason.
- Rate-limit, timeout and connection errors are not swallowed. They are
  recorded on the case as a scorer error.

The judge's cost is reported separately as `report.scoring_cost`.

### Custom scorers

A plain function `fn(output, case)` (sync or async) is a scorer. It may
return:

- a `ScoreResult`
- a `bool`
- a `float` (passes at ≥ 0.5)
- a `(score, reason)` tuple
- a `{"score", "passed", "reason"}` dict

```python
from aiforge.evals import scorer, output_text

def under_100_words(output, case):
    return len(output_text(output).split()) <= 100

@scorer(name="mentions_customer", threshold=1.0)
def mentions_customer(output, case):
    name = case.metadata["customer"]
    return (1.0, "named") if name in output_text(output) else (0.0, f"missing {name}")

report = evaluate(summarizer, "evals/emails.yaml",
                  scorers=[under_100_words, mentions_customer])
```

For more control, subclass `Scorer`, set `name`, and implement
`async def score(self, output, case) -> ScoreResult`. A scorer that raises
fails that case with reason `scorer error: ...`. The run continues.

---

## Reports

```python
report.pass_rate               # cases where the target ran and every scorer passed
report.scorer_stats()          # {name: ScorerStats(mean, pass_rate, passed_count, min_pass_rate, passed)}
report.total_cost              # target spend, USD
report.scoring_cost            # judge spend, USD
report.total_tokens
report.latency_percentiles()   # {"mean", "p50", "p90", "p95", "p99", "max"} in ms
report.failed_cases
report.passed                  # every scorer meets its minimum pass rate

report.to_json()               # includes a computed "summary" block
report.save("report.json")
EvalReport.load("report.json")
report.print()                 # rich tables
```

### Thresholds

Each scorer must reach its minimum pass rate for `report.passed` to be
true. `min_pass_rate` sets the default (1.0, which is strict).
`thresholds={"llm_judge": 0.8}` overrides it per scorer. A report with no
cases never passes.

### Regressions

```python
baseline = EvalReport.load("evals/baseline.json")
cmp = report.compare(baseline, tolerance=0.02)
cmp.regressed          # True if any scorer's pass rate or mean dropped by > tolerance
cmp.regressions        # [ScorerDelta(name, pass_rate_delta, mean_delta, ...)]
cmp.newly_failing      # case ids that passed in the baseline and fail now
cmp.print()
```

Scorers that are only in one of the two reports are listed as `added` or
`removed`. They are not counted as regressions.

---

## Use in pytest

```python
from aiforge import Classifier
from aiforge.evals import assert_eval, evaluate, label_match

def test_ticket_classifier():
    report = evaluate(
        Classifier(labels=["bug", "feature", "question"]),
        "evals/tickets.yaml",
        scorers=[label_match()],
        min_pass_rate=0.9,
    )
    assert_eval(report, baseline="evals/baseline.json")
```

`assert_eval` raises an `AssertionError` that lists each scorer below its
threshold, each regression and the first failing cases with their
reasons.

---

## CLI

```bash
aiforge eval run evals/tickets.yaml \
  --module classify \
  --module-args '{"labels": ["bug", "feature", "question"]}' \
  --scorer label_match \
  --scorer 'llm_judge:{"rubric": "The label fits the ticket"}' \
  --judge-model gpt-4o-mini \
  --min-pass-rate 0.9 \
  --threshold llm_judge=0.8 \
  --output report.json \
  --baseline evals/baseline.json
```

| Option | Meaning |
|---|---|
| `--module/-m NAME` | Registered module (`classify`, `extract`, `summarize`, …, or one you registered) |
| `--module-args JSON` | Constructor kwargs |
| `--target pkg.mod:func` | Evaluate a callable instead of a module. `--module-args` are passed to it if given |
| `--method NAME` | Call a different method |
| `--call-args JSON` | Kwargs for every call |
| `--scorer/-s SPEC` | `name` or `name:{json params}`. Repeatable |
| `--min-pass-rate F` | Default minimum per scorer (default 1.0) |
| `--threshold name=F` | Per-scorer minimum. Repeatable |
| `--concurrency/-c N` | Cases in flight (default 5) |
| `--timeout S` | Per-case timeout |
| `--judge-provider`, `--judge-model` | Defaults for any `llm_judge` scorer |
| `--output/-o PATH` | Write the JSON report |
| `--baseline/-b PATH` | Compare with a saved report |
| `--tolerance F` | Allowed drop before a regression is flagged |
| `--name NAME` | Report name (default: the dataset file stem) |
| `--max-failures N` | Failing cases to print (default 10) |

`aiforge eval compare current.json baseline.json` compares two saved
reports.

**Exit codes**

| Code | Meaning |
|---|---|
| 0 | All scorers met their thresholds, and there was no regression |
| 1 | At least one scorer is below its minimum pass rate |
| 2 | Bad usage (unknown module or scorer, invalid JSON, …) |
| 3 | Thresholds met, but the run regressed against the baseline |

---

## CI example

```yaml
# .github/workflows/evals.yml
name: evals
on:
  pull_request:
    paths: ["prompts/**", "aiforge.yaml", "evals/**"]

jobs:
  evals:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: {python-version: "3.12"}
      - run: pip install aiforge-kit
      - name: Run evals
        env:
          AIFORGE_OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
        run: |
          aiforge eval run evals/tickets.yaml \
            -m classify --module-args '{"labels": ["bug", "feature", "question"]}' \
            -s label_match --min-pass-rate 0.9 \
            --baseline evals/baseline.json --tolerance 0.02 \
            --output eval-report.json
      - uses: actions/upload-artifact@v4
        if: always()
        with: {name: eval-report, path: eval-report.json}
```

To refresh the baseline after an intended change, commit the new
`eval-report.json` as `evals/baseline.json`.

---

## See also

- [Cost Tracking](cost-tracking.md): how per-call cost is computed
- [Providers](providers.md): structured output support per provider
- [CLI Reference](cli.md)
