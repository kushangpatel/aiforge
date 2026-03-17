# Observability & Cost Tracking

Every AIForge module call returns cost information and supports tracing for full observability.

---

## Cost Tracking

### Per-Call Cost

Every module call returns cost as part of the response:

```python
result = summarizer.run(text)
print(result.cost)             # 0.003 (USD)
print(result.input_tokens)     # 1,200
print(result.output_tokens)    # 150
```

### Session-Level Aggregation

```python
from aiforge import CostTracker

tracker = CostTracker.get_session()
print(tracker.total_cost)      # $0.042
print(tracker.total_tokens)    # 8,421
print(tracker.by_module)       # {"summarize": $0.012, "qa": $0.025, ...}
print(tracker.by_model)        # {"gpt-4o": $0.040, "text-embedding-3-small": $0.002}
```

### Cost Warning Threshold

Configure a per-call cost warning:

```yaml
# aiforge.yaml
cost_tracking:
  enabled: true
  warn_threshold: 1.00          # warn if single call > $1
```

Or via environment variable:

```bash
export AIFORGE_COST_WARN_THRESHOLD=1.00
```

---

## Pricing Tables

Built-in pricing for supported models:

```python
# aiforge/core/pricing.py
PRICING = {
    "gpt-4o": {"input": 5.00, "output": 15.00},           # per 1M tokens
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "claude-3-5-sonnet": {"input": 3.00, "output": 15.00},
    "claude-3-haiku": {"input": 0.25, "output": 1.25},
    "text-embedding-3-small": {"input": 0.02},
    "text-embedding-3-large": {"input": 0.13},
    # Ollama models: cost = 0 (local)
}
```

### Pricing Maintenance

The built-in pricing table is a best-effort snapshot at release time. If provider prices change:

1. **Config override** (immediate): Override any model price in `aiforge.yaml`:
    ```yaml
    pricing:
      gpt-4o: {input: 2.50, output: 10.00}  # override built-in
    ```
2. **Patch release**: Pricing updates are included in PATCH releases (no breaking change).
3. **Unknown models**: If a model isn't in the table and no override is set, cost is reported as `0.0` with a WARNING log: `"No pricing data for model 'X'. Cost tracking will be inaccurate."`

---

## Cost Estimation (Monthly, Per Developer)

| Component | Unit Cost | Estimated Dev Usage | Monthly Cost |
|-----------|----------|-------------------|-------------|
| OpenAI GPT-4o | $5/1M input, $15/1M output | ~2M input, ~500K output | ~$17.50 |
| OpenAI text-embedding-3-small | $0.02/1M tokens | ~5M tokens | ~$0.10 |
| Anthropic Claude 3.5 Sonnet | $3/1M input, $15/1M output | ~1M input, ~250K output | ~$6.75 |
| Ollama (local) | Free | Unlimited | $0.00 |
| ChromaDB | Free (local) | Unlimited | $0.00 |
| **Total (OpenAI only)** | | | **~$17.60/mo** |
| **Total (Ollama dev + OpenAI prod)** | | | **~$5.00/mo** |

---

## Tracing

### LangSmith Tracing

```yaml
# aiforge.yaml
tracing:
  enabled: true
  provider: langsmith
```

Or via environment variables:

```bash
export AIFORGE_TRACING_ENABLED=true
export AIFORGE_LANGSMITH_API_KEY=ls-...
```

Every call gets a `trace_id`:

```python
result = qa.ask("How does auth work?")
print(result.trace_id)  # "run-abc-123-def-456"
# View in LangSmith UI: https://smith.langchain.com/runs/abc-123-def-456
```

### OpenTelemetry Tracing

```yaml
tracing:
  enabled: true
  provider: otel
```

### Console Tracing (Development)

```yaml
tracing:
  enabled: true
  provider: console
```

---

## Structured Logging

All operations produce structured log entries via `structlog`:

```json
{
    "event": "llm_call",
    "module": "summarize",
    "provider": "openai",
    "model": "gpt-4o",
    "input_tokens": 1200,
    "output_tokens": 150,
    "cost": 0.003,
    "latency_ms": 820,
    "trace_id": "abc-123",
    "timestamp": "2026-03-06T10:30:00Z"
}
```

### Security

API keys never appear in logs — `structlog` filter strips `*_API_KEY` fields automatically.

---

## Configuration Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `AIFORGE_TRACING_ENABLED` | `false` | Enable tracing |
| `AIFORGE_TRACING_PROVIDER` | `langsmith` | Tracing backend: `langsmith`, `otel`, or `console` |
| `AIFORGE_LANGSMITH_API_KEY` | — | LangSmith API key |
| `AIFORGE_COST_TRACKING_ENABLED` | `true` | Enable cost tracking |
| `AIFORGE_COST_WARN_THRESHOLD` | `1.00` | Warn if single call exceeds this USD amount |

---

## See Also

- [Configuration](configuration.md) — Full configuration reference
- [Providers](providers.md) — Provider-specific pricing
