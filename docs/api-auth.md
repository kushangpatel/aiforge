# API Key Authentication

Protect your AIForge REST API with Bearer token authentication.

---

## How It Works

When `AIFORGE_API_KEY` is set (non-empty), the `AuthMiddleware` intercepts every
request and validates the `Authorization: Bearer <token>` header.  When the key is
**not set** (the default), authentication is completely disabled — all endpoints are
open.  This is the intended mode for local development.

```
Request ──► AuthMiddleware
               │
               ├─ API_KEY not set?  → pass through (no auth)
               ├─ Public path?      → pass through
               ├─ Valid Bearer?     → pass through
               └─ Otherwise         → 401 Unauthorized
```

---

## Configuration

Set the API key via `.env` or environment variable:

```bash
# .env
AIFORGE_API_KEY=your-secret-key
```

Or at launch time:

```bash
AIFORGE_API_KEY=your-secret-key aiforge serve
```

Or in `aiforge.yaml` (not recommended — prefer `.env` for secrets):

```yaml
# aiforge.yaml
api_key: your-secret-key
```

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `AIFORGE_API_KEY` | `str` | `""` (disabled) | Bearer token required on protected endpoints |

---

## Making Authenticated Requests

Include the `Authorization` header with every request:

```bash
curl -X POST http://localhost:8000/classify \
  -H "Authorization: Bearer your-secret-key" \
  -H "Content-Type: application/json" \
  -d '{"text": "Login is broken", "labels": ["bug", "feature", "question"]}'
```

### Python (requests)

```python
import requests

API_URL = "http://localhost:8000"
API_KEY = "your-secret-key"

resp = requests.post(
    f"{API_URL}/summarize",
    headers={"Authorization": f"Bearer {API_KEY}"},
    json={"text": "Long text to summarize..."},
)
print(resp.json())
```

### Python (httpx — async)

```python
import httpx

async with httpx.AsyncClient(
    base_url="http://localhost:8000",
    headers={"Authorization": "Bearer your-secret-key"},
) as client:
    resp = await client.post("/classify", json={
        "text": "Payment failing for all customers",
        "labels": ["urgent", "normal", "low"],
    })
    print(resp.json())
```

---

## Error Responses

### Missing or malformed header

```
HTTP 401 Unauthorized
{"detail": "Missing or invalid Authorization header"}
```

Returned when:
- No `Authorization` header is present
- The header does not start with `Bearer `

### Invalid token

```
HTTP 401 Unauthorized
{"detail": "Invalid API key"}
```

Returned when:
- The Bearer token does not match the configured `AIFORGE_API_KEY`
- The token is empty (`Authorization: Bearer `)

---

## Public Paths

The following paths **never** require authentication, even when an API key is
configured.  This ensures health checks, load-balancer probes, and API docs
remain accessible.

| Path | Purpose |
|------|---------|
| `/health` | Health check / status |
| `/docs` | Swagger UI |
| `/openapi.json` | OpenAPI spec (JSON) |
| `/redoc` | ReDoc UI |

All other paths — including `/` — require a valid Bearer token when the key is set.

---

## Security Properties

| Property | Detail |
|----------|--------|
| **Constant-time comparison** | `secrets.compare_digest()` prevents timing attacks |
| **No key echo** | The API key is never returned in any response |
| **No key logging** | structlog's `_strip_secrets` processor redacts `api_key`, `token`, and `authorization` fields |
| **Auth failure audit trail** | Every rejected request is logged with `path`, `method`, and `client` IP |
| **Disabled by default** | Empty `AIFORGE_API_KEY` = auth middleware is a no-op (local dev friendly) |

---

## Docker / Production

Pass the key as an environment variable — never bake it into the image:

```bash
docker run -p 8000:8000 \
  -e AIFORGE_OPENAI_API_KEY=sk-... \
  -e AIFORGE_API_KEY=your-secret-key \
  aiforge/server
```

Docker Compose:

```yaml
services:
  api:
    build: .
    ports:
      - "8000:8000"
    env_file:
      - .env          # contains AIFORGE_API_KEY + provider keys
    command: aiforge serve
```

---

## See Also

- [Configuration](configuration.md) — full settings reference and layering rules
- [API Server](api-server.md) — endpoints, CORS, async jobs, Docker deployment
