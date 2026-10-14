# Cost Tracking & Pricing

AIForge automatically tracks the USD cost and token usage of every LLM and
embedding call. Cost data is returned per-call, aggregated per-session, and
exposed via the REST API — giving you complete visibility into spend without
any extra instrumentation.

---

## Table of Contents

- [Why Cost Tracking?](#why-cost-tracking)
- [How It Works](#how-it-works)
- [Per-Call Cost](#per-call-cost)
- [Session-Level Aggregation](#session-level-aggregation)
  - [CostTracker API](#costtracker-api)
  - [Properties](#properties)
  - [Summary](#summary)
  - [Reset](#reset)
- [Cost Warning Threshold](#cost-warning-threshold)
- [REST API — GET /stats](#rest-api--get-stats)
- [Built-in Pricing Table](#built-in-pricing-table)
  - [OpenAI Models](#openai-models)
  - [Anthropic Models](#anthropic-models)
  - [Voyage Embeddings](#voyage-embeddings)
  - [Ollama / Local Models](#ollama--local-models)
- [Context Windows](#context-windows)
- [Custom Pricing Overrides](#custom-pricing-overrides)
- [Custom Context Windows](#custom-context-windows)
- [Fuzzy Model Matching](#fuzzy-model-matching)
- [Cost Calculation Formula](#cost-calculation-formula)
- [Prompt Caching — Cost Savings](#prompt-caching--cost-savings)
- [Request Cache — Cost Savings](#request-cache--cost-savings)
- [Rate Limiting — Budget Protection](#rate-limiting--budget-protection)
- [Batch Operations](#batch-operations)
- [Cost Estimation (Monthly, Per Developer)](#cost-estimation-monthly-per-developer)
- [Integration Examples](#integration-examples)
  - [Python SDK](#python-sdk)
  - [REST API (curl)](#rest-api-curl)
  - [Node.js](#nodejs)
  - [Go](#go)
  - [Java](#java)
- [Troubleshooting](#troubleshooting)
- [Configuration Reference](#configuration-reference)
- [See Also](#see-also)

---

## Why Cost Tracking?

| Without tracking | With AIForge cost tracking |
|-----------------|---------------------------|
| Discover surprise bills weeks later | See per-call cost in every `ModuleResponse` |
| No breakdown by feature or model | Aggregation by module **and** by model |
| No budget guardrails | `cost_warn_threshold` logs a WARNING for expensive calls |
| Manual token counting | Input/output tokens tracked automatically |
| No caching awareness | Cache hits return `cost=0`, and provider prompt-cache reads are billed at the discounted rate |

Cost tracking is **enabled by default**. Set `AIFORGE_COST_TRACKING_ENABLED=false` to disable.

---

## How It Works

```
Module.run(text)
    │
    ├── RateLimiter.acquire()          ← block if RPM/TPM exceeded
    │
    ├── RequestCache.get(key)          ← return cached result (cost=0) if hit
    │
    ├── Provider.complete(messages)    ← actual LLM call
    │       returns CompletionResult { text, input_tokens, output_tokens, model,
    │                                  cache_read_tokens, cache_write_tokens }
    │
    ├── CompletionResult.cost
    │       → calculate_cost(model, input_tokens, output_tokens,
    │                        cache_read_tokens=..., cache_write_tokens=...)
    │       looks up PRICING table → USD amount
    │
    ├── CostTracker.record(module, model, tokens, cost)
    │       accumulates session-level stats
    │
    └── return ModuleResponse { text, cost, input_tokens, output_tokens, model }
```

Every module (`Summarizer`, `Classifier`, `DocQA`, etc.) follows this exact
flow via the `BaseModule._call_llm()` / `_acall_llm()` methods. No manual
instrumentation is required.

---

## Per-Call Cost

Every `ModuleResponse` includes cost and token fields:

```python
from aiforge import Summarizer

s = Summarizer()
result = s.run("Long document text here...")

print(result.cost)            # 0.00285  (USD)
print(result.input_tokens)    # 1200
print(result.output_tokens)   # 150
print(result.model)           # "gpt-4o"
print(result.latency_ms)      # 820.5
```

**`ModuleResponse` cost fields:**

| Field | Type | Description |
|-------|------|-------------|
| `cost` | `float` | USD cost of this call (0.0 for cache hits or unknown models) |
| `input_tokens` | `int` | Tokens sent to the model |
| `output_tokens` | `int` | Tokens received from the model |
| `model` | `str` | Model name used for the call |
| `latency_ms` | `float` | Wall-clock time in milliseconds |
| `trace_id` | `str` | Trace ID (if tracing enabled) |

---

## Session-Level Aggregation

The `CostTracker` singleton accumulates costs across **all** module calls in the
current process. It is thread-safe and requires no setup.

### CostTracker API

```python
from aiforge.infra.cost_tracker import CostTracker

# Get the process-level singleton
tracker = CostTracker.get_session()
```

| Method | Signature | Description |
|--------|-----------|-------------|
| `get_session()` | `@classmethod (warn_threshold=1.0) → CostTracker` | Return or create the process-level singleton |
| `reset_session()` | `@classmethod () → None` | Destroy the singleton (next `get_session()` creates a new one) |
| `record()` | `(*, module, model, input_tokens, output_tokens, cost) → None` | Record a completed LLM call (called automatically by modules) |
| `summary()` | `() → dict` | Return a snapshot of all tracked metrics |
| `reset()` | `() → None` | Reset all counters on the current instance |

### Properties

All properties are thread-safe (acquire an internal lock before reading):

```python
tracker = CostTracker.get_session()

tracker.total_cost    # float — cumulative USD cost
tracker.total_tokens  # int   — cumulative input + output tokens
tracker.call_count    # int   — number of LLM/embedding calls
tracker.by_module     # dict[str, float] — cost grouped by module name
tracker.by_model      # dict[str, float] — cost grouped by model name
```

### Summary

`summary()` returns all metrics in a single dict — ideal for logging,
dashboards, or API responses:

```python
tracker.summary()
# {
#   "total_cost_usd": 0.04215,
#   "total_tokens": 8421,
#   "call_count": 12,
#   "by_module": {"Summarizer": 0.012, "DocQA": 0.025, "Classifier": 0.005},
#   "by_model": {"gpt-4o": 0.040, "text-embedding-3-small": 0.002}
# }
```

### Reset

```python
# Reset counters on the current instance (keeps the singleton alive)
tracker.reset()

# OR fully destroy and recreate
CostTracker.reset_session()
tracker = CostTracker.get_session()
```

---

## Cost Warning Threshold

When a **single** LLM call exceeds the configured threshold, a structured
WARNING log is emitted:

```yaml
# aiforge.yaml
cost_tracking_enabled: true
cost_warn_threshold: 1.00    # USD — warn when any single call exceeds $1
```

```bash
# Or via environment variable
export AIFORGE_COST_WARN_THRESHOLD=0.50
```

The warning includes full context for investigation:

```json
{
  "event": "High-cost LLM call",
  "cost_usd": "$1.2500",
  "threshold_usd": "$1.0000",
  "module": "Summarizer",
  "model": "gpt-4-turbo",
  "input_tokens": 95000,
  "output_tokens": 2000
}
```

The threshold is a **warning only** — calls are never blocked. Adjust the
threshold to match your budget: `0.10` for strict monitoring, `5.00` for
batch workloads.

---

## REST API — GET /stats

The API server exposes session-level cost metrics at the `/stats` endpoint:

```bash
curl http://localhost:8000/stats
```

```json
{
  "total_cost_usd": 0.04215,
  "total_tokens": 8421,
  "call_count": 12,
  "by_module": {
    "Summarizer": 0.012,
    "DocQA": 0.025,
    "Classifier": 0.005
  },
  "by_model": {
    "gpt-4o": 0.040,
    "text-embedding-3-small": 0.002
  }
}
```

This endpoint reads from `CostTracker.get_session()` — the same singleton
used by all modules. No authentication is required for `/stats` (it is a
public path by default).

---

## Built-in Pricing Table

All prices are in **USD per 1,000,000 tokens** (1M tokens). The table is
defined in `aiforge/core/pricing.py` and kept as a best-effort snapshot.

### OpenAI Models

| Model | Input ($/1M) | Output ($/1M) | Context Window |
|-------|-------------|---------------|---------------|
| `gpt-4o` | 2.50 | 10.00 | 128,000 |
| `gpt-4o-mini` | 0.15 | 0.60 | 128,000 |
| `gpt-4-turbo` | 10.00 | 30.00 | 128,000 |
| `gpt-4.5` / `gpt-4.5-preview` | 75.00 | 150.00 | 128,000 |
| `gpt-3.5-turbo` | 0.50 | 1.50 | 16,385 |
| `o1` | 15.00 | 60.00 | 200,000 |
| `o1-preview` | 15.00 | 60.00 | 128,000 |
| `o1-mini` | 3.00 | 12.00 | 128,000 |
| `o3` | 10.00 | 40.00 | 200,000 |
| `o3-mini` | 1.10 | 4.40 | 200,000 |
| `text-embedding-3-small` | 0.02 | — | — |
| `text-embedding-3-large` | 0.13 | — | — |
| `text-embedding-ada-002` | 0.10 | — | — |

### Anthropic Models

| Model | Input ($/1M) | Output ($/1M) | Context Window |
|-------|-------------|---------------|---------------|
| `claude-fable-5-1` | 10.00 | 50.00 | 1,000,000 |
| `claude-opus-5-5` | 4.00 | 20.00 | 1,000,000 |
| `claude-opus-5` | 5.00 | 25.00 | 1,000,000 |
| `claude-opus-4-8` / `4-7` / `4-6` | 5.00 | 25.00 | 1,000,000 |
| `claude-sonnet-5-5` | 2.00 | 10.00 | 1,000,000 |
| `claude-sonnet-5` | 2.00 | 10.00 | 1,000,000 |
| `claude-sonnet-4-6` | 3.00 | 15.00 | 1,000,000 |
| `claude-haiku-5-5` | 0.10 | 0.50 | 1,000,000 |
| `claude-haiku-4-5` | 1.00 | 5.00 | 200,000 |

`claude-haiku-5-5` is listed at its rate for prompts up to 100K tokens. Retired Claude 3.x
models stay in the table so historical cost reports still resolve.

### Voyage Embeddings

Used with the Anthropic provider for embedding calls:

| Model | Input ($/1M) | Output ($/1M) |
|-------|-------------|---------------|
| `voyage-3` | 0.06 | — |
| `voyage-3-lite` | 0.02 | — |
| `voyage-large-2` | 0.12 | — |
| `voyage-code-2` | 0.12 | — |

### Ollama / Local Models

All local models are priced at **$0.00** (no cost):

`llama3`, `llama3.1`, `llama3.2`, `llama3.3`, `mistral`, `mistral-nemo`, `codellama`, `phi-3`, `phi-4`, `gemma2`, `qwen2.5`, `nomic-embed-text`, `mxbai-embed-large`, `all-minilm`

---

## Context Windows

Context windows determine the maximum token length per call. The Summarizer
uses this to decide between `stuff` (single-shot) and `map_reduce` (chunked)
strategies.

Default fallback for unknown models: **8,192** tokens.

See the pricing tables above for per-model context windows. Override with
`custom_context_windows` in your config.

---

## Custom Pricing Overrides

Override built-in prices or add pricing for new/fine-tuned models:

```yaml
# aiforge.yaml
custom_pricing:
  gpt-4o:                    # override built-in price
    input: 2.50
    output: 10.00
  my-finetuned-model:        # add new model
    input: 6.00
    output: 18.00
  claude-sonnet-5-5:         # optional explicit prompt-cache prices
    input: 2.00
    output: 10.00
    cache_read: 0.20
    cache_write: 2.50
```

`cache_read` and `cache_write` are optional. When they are left out, the
default multipliers in [Prompt Caching](#prompt-caching--cost-savings) apply.

Overrides are applied at startup via `apply_pricing_overrides()` and take
precedence over the built-in `PRICING` table. The lookup order is:

1. `_pricing_overrides` (from `custom_pricing` config) — **checked first**
2. `PRICING` (built-in table) — **fallback**
3. Not found → cost = `0.0` + WARNING log

You can also apply overrides at runtime:

```python
from aiforge.core.pricing import apply_pricing_overrides

apply_pricing_overrides({
    "my-new-model": {"input": 3.00, "output": 12.00}
})
```

---

## Custom Context Windows

```yaml
# aiforge.yaml
custom_context_windows:
  my-finetuned-model: 128000
  custom-llama: 32768
```

Or at runtime:

```python
from aiforge.core.pricing import apply_context_window_overrides

apply_context_window_overrides({"my-new-model": 64000})
```

---

## Fuzzy Model Matching

When looking up pricing or context windows, AIForge uses fuzzy matching to
handle date-suffixed model variants:

```
claude-3-7-sonnet-20250219  →  strips "-20250219"  →  matches "claude-3-7-sonnet"
gpt-4o-2024-11-20           →  strips "-20"        →  matches "gpt-4o-2024-11"
```

The algorithm: if the exact name isn't found, split on the last `-` and retry
with the prefix. This handles the common pattern of providers appending release
dates to model names.

---

## Cost Calculation Formula

```
uncached = input_tokens − cache_read_tokens − cache_write_tokens

cost = (input_price  × uncached
      + cache_read_price  × cache_read_tokens
      + cache_write_price × cache_write_tokens
      + output_price × output_tokens) / 1,000,000
```

Without prompt caching both cache counts are zero, so this reduces to
`input_price × input_tokens + output_price × output_tokens`.

**Example:** A `gpt-4o` call with 2,000 input tokens and 500 output tokens:

$$
\text{cost} = \frac{2.50 \times 2{,}000}{1{,}000{,}000} + \frac{10.00 \times 500}{1{,}000{,}000} = 0.005 + 0.005 = \$0.01
$$

**API function:**

```python
from aiforge.core.pricing import calculate_cost

cost = calculate_cost("gpt-4o", input_tokens=2000, output_tokens=500)
# 0.01

# With prompt caching: 1,800 of the 2,000 input tokens were cache reads
cost = calculate_cost("gpt-4o", 2000, 500, cache_read_tokens=1800)
# 0.00775  (2.50 × 200 + 1.25 × 1,800 + 10.00 × 500) / 1M
```

`CompletionResult.cost` applies this formula to a provider result directly.

---

## Prompt Caching — Cost Savings

Providers can reuse a long, repeated prompt prefix (system prompt, tool
definitions, a document) across calls and bill those input tokens at a
discount. AIForge reports them on every `CompletionResult`:

| Field | Meaning |
|-------|---------|
| `cache_read_tokens` | Input tokens served from the provider's prompt cache |
| `cache_write_tokens` | Input tokens written to the cache on this call (Anthropic) |
| `cost` | USD cost with cache reads and writes priced at their own rates |

Default cache prices, as multiples of the model's input price:

| Provider | Cache read | Cache write |
|----------|-----------|-------------|
| Anthropic (Claude) | 0.1× | 1.25× |
| OpenAI / Azure / others | 0.5× | 1.0× (no surcharge) |

OpenAI and Azure cache long prompts automatically. For Anthropic, caching is
controlled by `prompt_cache` (`system` by default, or `auto` / `off`); see
[Configuration](configuration.md) and [Providers → Prompt caching](providers.md#prompt-caching).

Example from a live run of `claude-haiku-5-5` with a ~13.8K-token system
prompt sent twice:

| Call | Cache write | Cache read | Cost |
|------|------------|-----------|------|
| 1st | 13,821 | 0 | $0.00173 |
| 2nd | 0 | 13,821 | $0.00014 |

---

## Request Cache — Cost Savings

The request cache eliminates cost for repeated identical calls. Cache hits
return the stored result with `cost=0.0`, `input_tokens=0`, `output_tokens=0`.

```yaml
# aiforge.yaml
cache_enabled: true
cache_ttl: 3600           # 1 hour
cache_backend: memory     # or "redis" for shared caching
cache_max_entries: 1000   # LRU eviction for memory backend
```

Cache keys are SHA-256 hashes of: `model + messages + temperature + max_tokens + module + provider`. Two identical calls within the TTL window return the same result at zero cost.

Check cache effectiveness:

```python
from aiforge.infra.cache import RequestCache

cache = RequestCache.get_instance()
print(cache.stats())
# {"enabled": true, "size": 42, "hits": 120, "misses": 38, "hit_rate": 0.76, "ttl_seconds": 3600}
```

A high hit rate directly translates to cost savings.

---

## Rate Limiting — Budget Protection

Rate limiting prevents runaway costs by capping throughput:

```yaml
# aiforge.yaml
rate_limit_rpm: 500       # requests per minute
rate_limit_tpm: 150000    # tokens per minute
```

When limits are hit, requests **block** (sleep) until capacity is available —
they do not fail with an error. This provides automatic backpressure without
losing work.

```python
from aiforge.infra.rate_limiter import RateLimiter

# Update limits at runtime
RateLimiter.get_instance().update_limits(rpm=100, tpm=50000)
```

Combined with cost tracking, rate limiting ensures you stay within both
API provider limits and your budget.

---

## Batch Operations

`BatchResult` aggregates cost across multiple calls:

```python
from aiforge import Classifier

c = Classifier(categories=["positive", "negative", "neutral"])
batch = c.run_batch(["Great product!", "Terrible service.", "It's okay."])

print(batch.total_cost)    # 0.0045 (sum of all calls)
print(batch.total_tokens)  # 450
print(len(batch.results))  # 3
print(len(batch.failed))   # 0

for r in batch.results:
    print(f"{r.text}: ${r.cost:.4f}")
```

**`BatchResult` fields:**

| Field | Type | Description |
|-------|------|-------------|
| `results` | `list[ModuleResponse]` | Successful call results |
| `total_cost` | `float` | Sum of all `result.cost` values |
| `total_tokens` | `int` | Sum of all input + output tokens |
| `failed` | `list[dict]` | Failed items with error details |

---

## Cost Estimation (Monthly, Per Developer)

| Component | Unit Cost | Estimated Dev Usage | Monthly Cost |
|-----------|----------|-------------------|-------------|
| OpenAI GPT-4o | $2.50/1M input, $10/1M output | ~2M input, ~500K output | ~$10.00 |
| OpenAI text-embedding-3-small | $0.02/1M tokens | ~5M tokens | ~$0.10 |
| Anthropic Claude Sonnet 5.5 | $2/1M input, $10/1M output | ~1M input, ~250K output | ~$4.50 |
| Ollama (local) | Free | Unlimited | $0.00 |
| ChromaDB | Free (local) | Unlimited | $0.00 |
| **Total (OpenAI only)** | | | **~$10.10/mo** |
| **Total (mixed: Ollama dev + cloud prod)** | | | **~$5.00/mo** |

> **Tip:** Use Ollama during development (`provider: ollama`) and switch to
> cloud providers for staging/production. All Ollama models are priced at $0.

---

## Integration Examples

### Python SDK

```python
from aiforge import Summarizer, Classifier, DocQA
from aiforge.infra.cost_tracker import CostTracker

# Make some calls
s = Summarizer()
c = Classifier(categories=["bug", "feature", "question"])
qa = DocQA()

summary = s.run("Long document text...")
label = c.run("The app crashes on startup")
qa.ingest(["doc1.pdf", "doc2.md"])
answer = qa.ask("How does authentication work?")

# Check per-call costs
print(f"Summary cost: ${summary.cost:.4f}")
print(f"Classify cost: ${label.cost:.4f}")
print(f"QA cost: ${answer.cost:.4f}")

# Session-level totals
tracker = CostTracker.get_session()
metrics = tracker.summary()
print(f"Total spend: ${metrics['total_cost_usd']:.4f}")
print(f"Total tokens: {metrics['total_tokens']:,}")
print(f"Calls: {metrics['call_count']}")
print(f"By module: {metrics['by_module']}")
print(f"By model: {metrics['by_model']}")
```

### REST API (curl)

```bash
# Make an API call — cost is in the response
curl -s http://localhost:8000/summarize \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $AIFORGE_API_KEY" \
  -d '{"text": "Long document text..."}' | jq '{cost, input_tokens, output_tokens, model}'
# {
#   "cost": 0.00285,
#   "input_tokens": 1200,
#   "output_tokens": 150,
#   "model": "gpt-4o"
# }

# Check session-level stats
curl -s http://localhost:8000/stats | jq .
# {
#   "total_cost_usd": 0.04215,
#   "total_tokens": 8421,
#   "call_count": 12,
#   "by_module": { "Summarizer": 0.012, "DocQA": 0.025, "Classifier": 0.005 },
#   "by_model": { "gpt-4o": 0.040, "text-embedding-3-small": 0.002 }
# }
```

### Node.js

```javascript
const BASE = "http://localhost:8000";
const HEADERS = {
  "Content-Type": "application/json",
  Authorization: `Bearer ${process.env.AIFORGE_API_KEY}`,
};

// Make a call and read cost from the response
async function summarizeWithCost(text) {
  const res = await fetch(`${BASE}/summarize`, {
    method: "POST",
    headers: HEADERS,
    body: JSON.stringify({ text }),
  });
  const data = await res.json();
  console.log(`Cost: $${data.cost.toFixed(4)}`);
  console.log(`Tokens: ${data.input_tokens} in / ${data.output_tokens} out`);
  return data;
}

// Poll session stats
async function getStats() {
  const res = await fetch(`${BASE}/stats`);
  const stats = await res.json();
  console.log(`Total spend: $${stats.total_cost_usd.toFixed(4)}`);
  console.log(`Calls: ${stats.call_count}`);
  console.log("By module:", stats.by_module);
  console.log("By model:", stats.by_model);
  return stats;
}
```

### Go

```go
package main

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
)

type ModuleResponse struct {
	Text         string  `json:"text"`
	Cost         float64 `json:"cost"`
	InputTokens  int     `json:"input_tokens"`
	OutputTokens int     `json:"output_tokens"`
	Model        string  `json:"model"`
}

type Stats struct {
	TotalCostUSD float64            `json:"total_cost_usd"`
	TotalTokens  int                `json:"total_tokens"`
	CallCount    int                `json:"call_count"`
	ByModule     map[string]float64 `json:"by_module"`
	ByModel      map[string]float64 `json:"by_model"`
}

func summarize(text string) (*ModuleResponse, error) {
	body := fmt.Sprintf(`{"text": %q}`, text)
	req, _ := http.NewRequest("POST", "http://localhost:8000/summarize",
		strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+apiKey)

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var result ModuleResponse
	json.NewDecoder(resp.Body).Decode(&result)
	fmt.Printf("Cost: $%.4f | Tokens: %d in, %d out\n",
		result.Cost, result.InputTokens, result.OutputTokens)
	return &result, nil
}

func getStats() (*Stats, error) {
	resp, err := http.Get("http://localhost:8000/stats")
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var stats Stats
	json.NewDecoder(resp.Body).Decode(&stats)
	fmt.Printf("Total: $%.4f across %d calls\n", stats.TotalCostUSD, stats.CallCount)
	return &stats, nil
}
```

### Java

```java
import com.fasterxml.jackson.databind.ObjectMapper;
import java.net.URI;
import java.net.http.*;
import java.util.Map;

public class AiForgeCostExample {
    private static final String BASE = "http://localhost:8000";
    private static final String API_KEY = System.getenv("AIFORGE_API_KEY");
    private static final ObjectMapper mapper = new ObjectMapper();
    private static final HttpClient client = HttpClient.newHttpClient();

    /** Call /summarize and print cost info. */
    public static Map<String, Object> summarize(String text) throws Exception {
        String json = mapper.writeValueAsString(Map.of("text", text));
        HttpRequest req = HttpRequest.newBuilder()
            .uri(URI.create(BASE + "/summarize"))
            .header("Content-Type", "application/json")
            .header("Authorization", "Bearer " + API_KEY)
            .POST(HttpRequest.BodyPublishers.ofString(json))
            .build();
        HttpResponse<String> resp = client.send(req, HttpResponse.BodyHandlers.ofString());
        Map<String, Object> result = mapper.readValue(resp.body(), Map.class);
        System.out.printf("Cost: $%.4f | Tokens: %s in, %s out%n",
            ((Number) result.get("cost")).doubleValue(),
            result.get("input_tokens"),
            result.get("output_tokens"));
        return result;
    }

    /** Fetch session-level cost stats. */
    public static Map<String, Object> getStats() throws Exception {
        HttpRequest req = HttpRequest.newBuilder()
            .uri(URI.create(BASE + "/stats"))
            .GET()
            .build();
        HttpResponse<String> resp = client.send(req, HttpResponse.BodyHandlers.ofString());
        Map<String, Object> stats = mapper.readValue(resp.body(), Map.class);
        System.out.printf("Total: $%.4f across %s calls%n",
            ((Number) stats.get("total_cost_usd")).doubleValue(),
            stats.get("call_count"));
        return stats;
    }
}
```

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `cost=0.0` for a known model | Model name has an unexpected suffix | Check `get_model_pricing("your-model")` — if `None`, add to `custom_pricing` |
| `cost=0.0` for all calls | `COST_TRACKING_ENABLED=false` | Set `AIFORGE_COST_TRACKING_ENABLED=true` |
| WARNING: "Unknown model X" | Model not in built-in table | Add to `custom_pricing` in `aiforge.yaml` |
| `by_module` is empty | No calls have been made yet | Make at least one module call, then check `summary()` |
| `/stats` returns all zeros | Server just started / session was reset | Stats accumulate from first call; they reset when the process restarts |
| Warn threshold fires too often | Threshold too low for your workload | Increase `AIFORGE_COST_WARN_THRESHOLD` (e.g., `5.0` for batch jobs) |
| Cache hits not showing in stats | Cache is disabled | Set `AIFORGE_CACHE_ENABLED=true` — cache hits bypass cost recording entirely |
| Rate limiter blocking too long | RPM/TPM caps too low | Increase `rate_limit_rpm` / `rate_limit_tpm` in config |

---

## Configuration Reference

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `AIFORGE_COST_TRACKING_ENABLED` | `bool` | `true` | Enable/disable cost tracking |
| `AIFORGE_COST_WARN_THRESHOLD` | `float` | `1.0` | USD threshold for per-call WARNING log |
| `AIFORGE_CUSTOM_PRICING` | `dict` | `{}` | Override or add model pricing (USD per 1M tokens) |
| `AIFORGE_CUSTOM_CONTEXT_WINDOWS` | `dict` | `{}` | Override model context window sizes |
| `AIFORGE_CACHE_ENABLED` | `bool` | `true` | Enable request cache (cache hits = $0) |
| `AIFORGE_CACHE_TTL` | `int` | `3600` | Cache time-to-live in seconds |
| `AIFORGE_CACHE_BACKEND` | `str` | `memory` | Cache backend: `memory` or `redis` |
| `AIFORGE_CACHE_MAX_ENTRIES` | `int` | `1000` | Max in-memory cache entries (LRU eviction) |
| `AIFORGE_CACHE_REDIS_URL` | `str` | `redis://localhost:6379/0` | Redis connection URL |
| `AIFORGE_RATE_LIMIT_RPM` | `int` | `500` | Requests per minute cap |
| `AIFORGE_RATE_LIMIT_TPM` | `int` | `150000` | Tokens per minute cap |

---

## See Also

- [Configuration](configuration.md) — Full configuration reference and settings hierarchy
- [API Server](api-server.md) — REST API endpoints including `/stats`
- [API Authentication](api-auth.md) — Securing the API server
- [Providers](providers.md) — Provider-specific details and model lists
- [Modules](modules/) — Per-module SDK documentation
