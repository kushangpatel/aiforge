# Guardrails

`aiforge.guardrails` adds composable input and output checks that wrap any module or callable. They cover PII and secret redaction, prompt-injection detection, length limits, banned terms, schema validation and system-prompt leaks, with no extra dependencies.

```python
from aiforge import Summarizer
from aiforge.guardrails import (
    Guard, PIIGuard, PromptInjectionGuard, MaxLengthGuard, SystemPromptLeakGuard,
)

guard = Guard(
    input=[PIIGuard(), PromptInjectionGuard(), MaxLengthGuard(max_tokens=8000)],
    output=[PIIGuard()],
    on_fail="redact",
)

safe = guard.wrap(Summarizer())
result = safe.run("Notes from jane@acme.io: ...")   # the model sees [EMAIL_1]
```

---

## Guard

```python
Guard(input=[...], output=[...], on_fail="block" | "redact" | "warn")
```

| Method | Returns |
|---|---|
| `check_input(text)` / `acheck_input(text)` | `GuardResult` |
| `check_output(text)` / `acheck_output(text)` | `GuardResult` |
| `wrap(target, rehydrate=True)` | `GuardedCallable` |

### `on_fail` modes

| Mode | Behaviour |
|---|---|
| `block` | The result has `passed=False`. `wrap()` raises `GuardrailViolation`. |
| `redact` | The check repairs the text: PII is redacted, banned terms and leaks are masked, over-long text is truncated. Checks that cannot repair (prompt injection, schema, a missing `must_match`) block instead. |
| `warn` | Logs a `guardrails.violation` warning and lets the text through unchanged. |

Every check also accepts its own `on_fail=`, which overrides the Guard's. For example, `Guard([PIIGuard(on_fail="redact"), PromptInjectionGuard()], on_fail="block")` redacts PII but blocks injection. Checks run in order and stop after the first block.

### GuardResult

| Field | Meaning |
|---|---|
| `passed` | `False` only if a violation blocked. Redacted or warned violations still pass. |
| `violations` | `Violation(type, subtype, message, span, severity, guard, action)` |
| `sanitized_text` | The text after redactions. |
| `original_text` | The text as received. |
| `mapping` | Placeholder → original value for PII redacted in this stage. |
| `raise_for_violation()` | Raises `GuardrailViolation` if blocked. |

`span` is a `(start, end)` offset into the text that the check saw. Violations deliberately leave out the matched value, so logging them does not leak PII.

### `wrap()`

`guard.wrap(target)` accepts a module with `run`/`arun`, a sync function or an async function. It finds the input text in this order:

1. the first positional `str`
2. a `text=`, `prompt=`, `query=`, `question=`, `input=` or `message=` keyword
3. the `user` turns of an OpenAI-style `messages` list (the caller's list is not mutated)

Then the wrapper:

1. sanitises the input text
2. calls through
3. checks the output (a `str`, or any pydantic model through `to_text()`/`.text`, repaired field by field so the type is kept)
4. raises `GuardrailViolation` (an `AIForgeError` with `.stage`, `.result` and `.violations`) when it blocks

With placeholder redaction and `rehydrate=True`, the input's placeholders are restored in the output. Your users get their own data back, but the model never saw it. PII that the *model* introduces stays redacted.

```python
async def call_llm(prompt: str) -> str: ...

safe = guard.wrap(call_llm)
await safe("Email a@b.com the report")          # async target -> awaitable
guard.wrap(Summarizer()).run(doc)               # module: run / arun
```

---

## Checks

| Check | Flags | `redact` does |
|---|---|---|
| `PIIGuard(types=None, style=None, custom_patterns=None, llm=False)` | PII and secrets | Redacts (mask / placeholder / hash) |
| `PromptInjectionGuard(llm=False, moderator=None, threshold=0.5, extra_patterns=())` | Injection and jailbreak phrasing | Blocks |
| `MaxLengthGuard(max_chars=None, max_tokens=None, encoding="cl100k_base")` | Too long | Truncates |
| `BannedTermsGuard(terms, topics=None, regex=False, case_sensitive=False)` | Whole-word terms, or `{topic: [keywords]}` | Replaces with `[REDACTED]` |
| `RegexGuard(must_match=(), must_not_match=())` | Missing or forbidden patterns | Removes forbidden matches; a missing required pattern blocks |
| `SchemaGuard(PydanticModel \| json_schema)` | Output that is not valid JSON for the schema (code fences are tolerated) | Blocks |
| `SystemPromptLeakGuard(system_prompt=None, canary=None, ngram=8)` | A canary token, or ≥ `ngram` consecutive words from the system prompt | Removes the leaked spans |

**Prompt injection.** Regex heuristics run first and take microseconds. If they find nothing and `llm=True` (or a `moderator=` is given), the text goes to the Moderator's `prompt_injection` LLM check, and is flagged when its confidence is at least `threshold`.

**System prompt leaks.** Call the check's `.protect(system_prompt)` to append a random canary marker. A reply that contains the marker has leaked the prompt.

```python
leak = SystemPromptLeakGuard(SYSTEM_PROMPT, canary=True)
system = leak.protect(SYSTEM_PROMPT)
guard = Guard(output=[leak], on_fail="block")
```

Custom checks subclass `Check` and implement `validate(text, ctx) -> list[Violation]`, plus `fix()` if they can repair the text.

---

## PII detection and redaction

```python
from aiforge.guardrails import PIIDetector, redact_pii, rehydrate

PIIDetector().detect("card 4111 1111 1111 1111")  # [PIIEntity(type="credit_card", start=5, ...)]

res = redact_pii("Mail a@b.com or c@d.org", style="placeholder")
res.text        # "Mail [EMAIL_1] or [EMAIL_2]"
res.mapping     # {"[EMAIL_1]": "a@b.com", "[EMAIL_2]": "c@d.org"}
rehydrate("Replied to [EMAIL_1]", res.mapping)
```

| Type | Validation |
|---|---|
| `email` | Pattern only |
| `phone` | Either a `+CC` international number (8–15 digits) or a national number with an area code (10–15 digits). Bare digit runs, ISBNs and dates are ignored. |
| `credit_card` | Card-network prefix and Luhn checksum |
| `iban` | ISO 13616 mod-97 checksum |
| `ssn` | US format, excluding the `000`, `666`, `9xx`, `00` and `0000` ranges |
| `ip_address` | IPv4 and IPv6 via `ipaddress`. Dotted quads after `v`, `version`, `==` or `@` are treated as versions and skipped. |
| `private_key` | A full `BEGIN … PRIVATE KEY` block |
| `aws_access_key`, `aws_secret_key` | `AKIA…` IDs, and `aws_secret_access_key=` values |
| `github_token` | `ghp_` / `gho_` / `ghu_` / `ghs_` / `ghr_` and `github_pat_` |
| `api_key` | OpenAI- and Anthropic-style `sk-…`, `sk-proj-…`, `sk-ant-…` |
| `google_api_key`, `slack_token`, `jwt` | Pattern only |
| `bearer_token`, `generic_secret` | `Bearer …` and `api_key=` / `token:` / `password=` values, accepted only above a Shannon-entropy threshold |

`types=` accepts any of these names or a group: `"pii"`, `"secrets"` or `"all"` (the default). When matches overlap, secrets win over PII and custom patterns win over everything.

**Redaction styles**

- `placeholder` (the default): `[EMAIL_1]`, reversible. Pass a shared `PIIVault` to keep numbering consistent across calls.
- `mask`: `*******`
- `hash`: `[EMAIL:9f86d081884c]`, a salted SHA-256 prefix that is stable but not reversible.

**Custom patterns**

```python
from aiforge.guardrails import PIIDetector, PIIPattern

PIIDetector(custom_patterns={"employee_id": r"\bEMP-\d{6}\b"})
PIIDetector(custom_patterns=[
    PIIPattern("ticket", r"T-(?P<value>\d{4})", validator=lambda v, text, start: v != "0000",
               severity="low"),
])
```

When the pattern has a named group `value`, only that group is reported and redacted.

**LLM-assisted detection (off by default).** Regexes cannot find names or postal addresses. `LLMPIIDetector` asks the configured provider for them in one call. It uses native structured output (`response_format`) when the provider supports it and falls back to parsing JSON otherwise. Cost is recorded in the session `CostTracker` under `guardrails`.

```python
PIIGuard(llm=True)                                   # settings' provider
PIIGuard(llm=LLMPIIDetector(provider=my_provider, types=["person_name", "address"]))
```

---

## Moderator integration

`Moderator(pii_types=...)` now also accepts any guardrails type or group. Defaults are unchanged:

```python
Moderator(checks=["pii"], pii_types=["email", "iban", "secrets"]).redact(text)
```

---

## Settings

| Setting | Default | Used when |
|---|---|---|
| `AIFORGE_GUARDRAILS_PII_REDACTION` | `placeholder` | `PIIGuard` or `redact_pii` is called without `style=` |
| `AIFORGE_GUARDRAILS_ON_FAIL` | `block` | `Guard` is created without `on_fail=` |
