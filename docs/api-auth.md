# API Key Authentication

Protect your AIForge REST API with Bearer-token authentication. This guide
covers **why** you need it, **how** it works internally, **how** to configure
it, and complete integration examples in **Python**, **curl**, **Node.js**,
**Go**, and **Java**.

---

## Table of Contents

- [Why Authenticate?](#why-authenticate)
- [How It Works](#how-it-works)
- [Generating a Key](#generating-a-key)
- [Configuration](#configuration)
- [Public Paths (No Auth Required)](#public-paths-no-auth-required)
- [Making Authenticated Requests](#making-authenticated-requests)
  - [curl](#curl)
  - [Python (requests)](#python-requests)
  - [Python (httpx — async)](#python-httpx--async)
  - [Node.js / TypeScript](#nodejs--typescript)
  - [Go](#go)
  - [Java](#java)
- [Error Responses](#error-responses)
- [Security Properties](#security-properties)
- [Middleware Architecture](#middleware-architecture)
- [Request Logging & Tracing](#request-logging--tracing)
- [Webhook SSRF Protection](#webhook-ssrf-protection)
- [CORS Configuration](#cors-configuration)
- [Docker / Production](#docker--production)
- [Kubernetes](#kubernetes)
- [Troubleshooting](#troubleshooting)
- [See Also](#see-also)

---

## Why Authenticate?

| Without Auth | With Auth |
|--------------|-----------|
| Any process on the network (or internet) can call your AI endpoints | Only clients with the secret key can make requests |
| No accountability — impossible to trace who consumed tokens | Auth failures logged with path, method, and client IP |
| Fine for local development and experimentation | Required for staging, production, or any shared deployment |
| Cost exposure — anyone can trigger expensive LLM calls | Token gate prevents unauthorised spend |

**Recommendation:** leave auth disabled for local dev (`AIFORGE_API_KEY` unset).
Enable it wherever the server is reachable by untrusted clients.

---

## How It Works

AIForge uses a single shared Bearer token (sometimes called a "pre-shared key"
or "static API key"). There is no user management, OAuth, or session system —
the same key is used by all authorised clients.

```
Client Request
      │
      ▼
┌─────────────────────────┐
│     AuthMiddleware       │
│                          │
│  AIFORGE_API_KEY set?    │──── No ──► pass through (all requests allowed)
│          │               │
│         Yes              │
│          │               │
│  Public path?            │──── Yes ─► pass through
│  (/health, /docs, etc.)  │
│          │               │
│         No               │
│          │               │
│  Authorization header    │
│  starts with "Bearer "?  │──── No ──► 401 "Missing or invalid Authorization header"
│          │               │
│         Yes              │
│          │               │
│  Token matches key?      │──── No ──► 401 "Invalid API key"
│  (constant-time compare) │
│          │               │
│         Yes              │
│          ▼               │
│    Forward to route      │
└─────────────────────────┘
```

The middleware is implemented in `aiforge/server/middleware.py` as a Starlette
`BaseHTTPMiddleware`. It is registered in the FastAPI app factory
(`create_app()`) and runs on **every** request before the route handler.

---

## Generating a Key

AIForge accepts any non-empty string as the API key — there is no built-in key
management service. Use a cryptographically random value:

```bash
# Python (recommended — 32 URL-safe bytes → 43 characters)
python -c "import secrets; print(secrets.token_urlsafe(32))"
# Example output: Ks8Jv3FqLm2RtNzWxYp1AhBcDe4GiUoQ5SnTfVg6Xkw

# OpenSSL
openssl rand -base64 32
# Example output: k7R+3Fq2Lm8Rt/NzWx=Yp1Ah+BcDeGi5UoQSnTfVgXk=

# Node.js
node -e "console.log(require('crypto').randomBytes(32).toString('base64url'))"
```

Store the output as `AIFORGE_API_KEY`. Treat it like a password — **never**
commit it to version control.

---

## Configuration

The API key can be set through any layer of AIForge's configuration system.
Layers are listed from **lowest** to **highest** priority — higher layers
override lower ones:

| Priority | Source | Example |
|----------|--------|---------|
| 1 (lowest) | Hardcoded default | `API_KEY: str | None = None` (disabled) |
| 2 | `aiforge.yaml` | `api_key: your-secret-key` |
| 3 | `.env` file | `AIFORGE_API_KEY=your-secret-key` |
| 4 | Environment variable | `export AIFORGE_API_KEY=your-secret-key` |
| 5 (highest) | Constructor kwarg | `AIForgeSettings(API_KEY="your-secret-key")` |

### Recommended: `.env` file (secrets only)

```bash
# .env  — NEVER commit this file
AIFORGE_API_KEY=your-secret-key
AIFORGE_OPENAI_API_KEY=sk-...
```

### Inline at launch time

```bash
AIFORGE_API_KEY=your-secret-key aiforge serve
```

### In aiforge.yaml (not recommended for secrets)

```yaml
# aiforge.yaml
api_key: your-secret-key   # prefer .env for secrets
api_host: "0.0.0.0"
api_port: 8000
```

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `AIFORGE_API_KEY` | `str \| None` | `None` (disabled) | Bearer token required on all protected endpoints. When empty or unset, auth is completely disabled. |

---

## Public Paths (No Auth Required)

The following paths **never** require authentication, even when an API key is
configured. This ensures health checks, load-balancer probes, and API
documentation remain accessible without credentials.

| Path | Purpose | When to Use |
|------|---------|-------------|
| `GET /health` | Liveness probe | Kubernetes `livenessProbe`, Docker `HEALTHCHECK` |
| `GET /health/ready` | Readiness probe (validates LLM provider connectivity) | Kubernetes `readinessProbe`, load-balancer health checks |
| `GET /docs` | Swagger UI | Browser-based API exploration |
| `GET /openapi.json` | OpenAPI 3.1 spec (JSON) | Client SDK generation, CI validation |
| `GET /redoc` | ReDoc UI | Alternative API docs viewer |

These paths are defined as a `frozenset` in `_PUBLIC_PATHS` and checked via
set membership — no regex or prefix matching, so `/health/foo` would still
require auth.

All other paths — including the root `/` — require a valid Bearer token when
the key is set.

---

## Making Authenticated Requests

Every request to a non-public endpoint must include an `Authorization` header
with the scheme `Bearer` followed by the API key.

### curl

```bash
# Classify text
curl -X POST http://localhost:8000/classify \
  -H "Authorization: Bearer your-secret-key" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Login button is broken on mobile",
    "labels": ["bug", "feature_request", "question"]
  }'

# Summarise text
curl -X POST http://localhost:8000/summarize \
  -H "Authorization: Bearer your-secret-key" \
  -H "Content-Type: application/json" \
  -d '{"text": "Long article text...", "style": "concise"}'

# Health check (no auth needed)
curl http://localhost:8000/health
```

### Python (requests)

```python
import requests

API_URL = "http://localhost:8000"
API_KEY = "your-secret-key"

headers = {"Authorization": f"Bearer {API_KEY}"}

# Classify
resp = requests.post(
    f"{API_URL}/classify",
    headers=headers,
    json={
        "text": "Payment failing for all customers",
        "labels": ["urgent", "normal", "low"],
    },
)
resp.raise_for_status()
result = resp.json()
print(f"Label: {result['label']} ({result['scores'][0]['confidence']})")

# Summarise
resp = requests.post(
    f"{API_URL}/summarize",
    headers=headers,
    json={"text": "Long text to summarize...", "style": "bullet_points"},
)
print(resp.json()["summary"])

# Extract
resp = requests.post(
    f"{API_URL}/extract",
    headers=headers,
    json={
        "text": "John Smith, CTO of Acme Corp. john@acme.com",
        "output_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "title": {"type": "string"},
                "email": {"type": "string"},
            },
            "required": ["name"],
        },
    },
)
print(resp.json()["data"])
```

### Python (httpx — async)

```python
import httpx

async with httpx.AsyncClient(
    base_url="http://localhost:8000",
    headers={"Authorization": "Bearer your-secret-key"},
    timeout=30.0,
) as client:
    # Classify
    resp = await client.post("/classify", json={
        "text": "Payment failing for all customers",
        "labels": ["urgent", "normal", "low"],
    })
    print(resp.json())

    # Chat (multi-turn)
    r1 = await client.post("/chat", json={"message": "What is Docker?"})
    conv_id = r1.json()["conversation_id"]

    r2 = await client.post("/chat", json={
        "message": "How does it differ from VMs?",
        "conversation_id": conv_id,
    })
    print(r2.json()["message"])

    # Moderate + redact
    resp = await client.post("/moderate", json={
        "text": "Email me at user@example.com",
        "checks": ["pii"],
        "action": "redact",
    })
    print(resp.json()["redacted_text"])
```

### Node.js / TypeScript

```typescript
const BASE_URL = process.env.AIFORGE_URL ?? "http://localhost:8000";
const API_KEY  = process.env.AIFORGE_API_KEY ?? "";

async function aiforge<T>(path: string, body?: unknown): Promise<T> {
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
  };
  if (API_KEY) {
    headers["Authorization"] = `Bearer ${API_KEY}`;
  }

  const res = await fetch(`${BASE_URL}${path}`, {
    method: body ? "POST" : "GET",
    headers,
    body: body ? JSON.stringify(body) : undefined,
  });

  if (!res.ok) {
    const detail = await res.text();
    throw new Error(`AIForge ${res.status}: ${detail}`);
  }
  return res.json() as Promise<T>;
}

// ── Classify ──
const cls = await aiforge<{ label: string; scores: { label: string; confidence: number }[] }>(
  "/classify",
  { text: "App crashes on startup", labels: ["bug", "feature_request", "question"] },
);
console.log(`Label: ${cls.label}`);

// ── Summarise ──
const sum = await aiforge<{ summary: string; cost_usd: number }>(
  "/summarize",
  { text: "Long article...", style: "bullet_points", max_length: 300 },
);
console.log(sum.summary);

// ── Chat (multi-turn) ──
const t1 = await aiforge<{ message: string; conversation_id: string }>(
  "/chat",
  { message: "What is Kubernetes?" },
);
const t2 = await aiforge<{ message: string }>(
  "/chat",
  { message: "How does it handle scaling?", conversation_id: t1.conversation_id },
);
console.log(t2.message);

// ── Extract ──
const ext = await aiforge<{ data: Record<string, unknown> }>(
  "/extract",
  {
    text: "Jane Doe, VP Engineering at TechCorp, jane@techcorp.io",
    output_schema: {
      type: "object",
      properties: {
        name: { type: "string" },
        title: { type: "string" },
        email: { type: "string" },
      },
      required: ["name"],
    },
  },
);
console.log(ext.data);

// ── Moderate ──
const mod = await aiforge<{ flagged: boolean; redacted_text: string | null }>(
  "/moderate",
  { text: "My SSN is 123-45-6789", checks: ["pii"], action: "redact" },
);
console.log(mod.redacted_text);

// ── Search — index + query ──
await aiforge("/search/index", {
  documents: [
    { id: "1", text: "AIForge is an AI toolkit.", metadata: { source: "readme" } },
  ],
  collection: "kb",
});

const search = await aiforge<{ hits: { id: string; text: string; score: number }[] }>(
  "/search/query",
  { query: "AI toolkit", collection: "kb", top_k: 3 },
);
search.hits.forEach((h) => console.log(`${h.score.toFixed(2)} — ${h.text}`));

// ── Generate ──
const gen = await aiforge<{ text: string }>(
  "/generate",
  { prompt: "Write a haiku about TypeScript", temperature: 0.9 },
);
console.log(gen.text);

// ── Document Q&A ──
await aiforge("/qa/ingest", { source: "./docs/", collection: "project" });

const qa = await aiforge<{ text: string; sources: string[] }>(
  "/qa/ask",
  { query: "How does auth work?", collection: "project" },
);
console.log(qa.text, qa.sources);

// ── Health (no auth needed) ──
const health = await aiforge<{ status: string }>("GET /health" as any);
```

### Go

```go
package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
)

var (
	baseURL = envOr("AIFORGE_URL", "http://localhost:8000")
	apiKey  = os.Getenv("AIFORGE_API_KEY")
)

func envOr(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

func aiforge(method, path string, body any, dest any) error {
	var reqBody io.Reader
	if body != nil {
		b, _ := json.Marshal(body)
		reqBody = bytes.NewReader(b)
	}

	req, _ := http.NewRequest(method, baseURL+path, reqBody)
	req.Header.Set("Content-Type", "application/json")
	if apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+apiKey)
	}

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 400 {
		detail, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("aiforge %d: %s", resp.StatusCode, detail)
	}

	if dest != nil {
		return json.NewDecoder(resp.Body).Decode(dest)
	}
	return nil
}

func main() {
	// Classify
	var cls struct {
		Label  string `json:"label"`
		Scores []struct {
			Label      string  `json:"label"`
			Confidence float64 `json:"confidence"`
		} `json:"scores"`
	}
	_ = aiforge("POST", "/classify", map[string]any{
		"text":   "App crashes on startup",
		"labels": []string{"bug", "feature_request", "question"},
	}, &cls)
	fmt.Printf("Label: %s (%.2f)\n", cls.Label, cls.Scores[0].Confidence)

	// Summarise
	var sum struct {
		Summary string  `json:"summary"`
		CostUSD float64 `json:"cost_usd"`
	}
	_ = aiforge("POST", "/summarize", map[string]any{
		"text":  "Long article...",
		"style": "bullet_points",
	}, &sum)
	fmt.Println("Summary:", sum.Summary)

	// Chat (multi-turn)
	var t1 struct {
		Message        string `json:"message"`
		ConversationID string `json:"conversation_id"`
	}
	_ = aiforge("POST", "/chat", map[string]any{
		"message": "What is Docker?",
	}, &t1)
	fmt.Println("Turn 1:", t1.Message)

	var t2 struct{ Message string `json:"message"` }
	_ = aiforge("POST", "/chat", map[string]any{
		"message":         "How does it differ from VMs?",
		"conversation_id": t1.ConversationID,
	}, &t2)
	fmt.Println("Turn 2:", t2.Message)

	// Moderate
	var mod struct {
		Flagged      bool   `json:"flagged"`
		RedactedText string `json:"redacted_text"`
	}
	_ = aiforge("POST", "/moderate", map[string]any{
		"text":   "My email is user@example.com",
		"checks": []string{"pii"},
		"action": "redact",
	}, &mod)
	fmt.Println("Redacted:", mod.RedactedText)

	// Generate
	var gen struct{ Text string `json:"text"` }
	_ = aiforge("POST", "/generate", map[string]any{
		"prompt":      "Write a haiku about Go",
		"temperature": 0.9,
	}, &gen)
	fmt.Println("Generated:", gen.Text)
}
```

### Java

```java
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;

/**
 * AIForge authenticated client for Java 11+.
 * Requires: com.fasterxml.jackson.core:jackson-databind:2.17+
 */
public class AIForgeAuthClient {

    private static final String BASE_URL =
            System.getenv().getOrDefault("AIFORGE_URL", "http://localhost:8000");
    private static final String API_KEY =
            System.getenv().getOrDefault("AIFORGE_API_KEY", "");

    private static final HttpClient HTTP = HttpClient.newHttpClient();
    private static final ObjectMapper JSON = new ObjectMapper();

    static JsonNode request(String method, String path, Object body) throws Exception {
        HttpRequest.Builder builder = HttpRequest.newBuilder()
                .uri(URI.create(BASE_URL + path))
                .header("Content-Type", "application/json");

        if (!API_KEY.isEmpty()) {
            builder.header("Authorization", "Bearer " + API_KEY);
        }

        if (body != null) {
            builder.method(method, HttpRequest.BodyPublishers.ofString(
                    JSON.writeValueAsString(body)));
        } else {
            builder.method(method, HttpRequest.BodyPublishers.noBody());
        }

        HttpResponse<String> resp = HTTP.send(builder.build(),
                HttpResponse.BodyHandlers.ofString());

        if (resp.statusCode() >= 400) {
            throw new RuntimeException(
                    "AIForge " + resp.statusCode() + ": " + resp.body());
        }
        return JSON.readTree(resp.body());
    }

    static JsonNode post(String path, Object body) throws Exception {
        return request("POST", path, body);
    }

    static JsonNode get(String path) throws Exception {
        return request("GET", path, null);
    }

    public static void main(String[] args) throws Exception {
        // Health (no auth needed)
        JsonNode health = get("/health");
        System.out.println("Status: " + health.get("status").asText());

        // Classify
        ObjectNode clsBody = JSON.createObjectNode()
                .put("text", "The login page returns a 500 error");
        clsBody.putArray("labels").add("bug").add("feature_request").add("question");

        JsonNode cls = post("/classify", clsBody);
        System.out.printf("Label: %s (%.2f)%n",
                cls.get("label").asText(),
                cls.get("scores").get(0).get("confidence").asDouble());

        // Summarise
        ObjectNode sumBody = JSON.createObjectNode()
                .put("text", "Long article about AI...")
                .put("style", "bullet_points")
                .put("max_length", 300);

        JsonNode sum = post("/summarize", sumBody);
        System.out.println("Summary: " + sum.get("summary").asText());
        System.out.printf("Cost: $%.4f%n", sum.get("cost_usd").asDouble());

        // Chat (multi-turn)
        ObjectNode chat1 = JSON.createObjectNode()
                .put("message", "What is Kubernetes?");
        JsonNode t1 = post("/chat", chat1);
        String convId = t1.get("conversation_id").asText();
        System.out.println("Turn 1: " + t1.get("message").asText());

        ObjectNode chat2 = JSON.createObjectNode()
                .put("message", "How does it handle auto-scaling?")
                .put("conversation_id", convId);
        JsonNode t2 = post("/chat", chat2);
        System.out.println("Turn 2: " + t2.get("message").asText());

        // Extract
        ObjectNode schema = JSON.createObjectNode().put("type", "object");
        schema.putObject("properties")
                .putObject("name").put("type", "string");
        schema.putArray("required").add("name");

        ObjectNode extBody = JSON.createObjectNode()
                .put("text", "Jane Doe, VP Engineering at TechCorp");
        extBody.set("output_schema", schema);

        JsonNode ext = post("/extract", extBody);
        System.out.println("Extracted: " + ext.get("data"));

        // Moderate
        ObjectNode modBody = JSON.createObjectNode()
                .put("text", "My SSN is 123-45-6789")
                .put("action", "redact");
        modBody.putArray("checks").add("pii");

        JsonNode mod = post("/moderate", modBody);
        System.out.println("Redacted: " + mod.get("redacted_text").asText());

        // Generate
        ObjectNode genBody = JSON.createObjectNode()
                .put("prompt", "Write a haiku about Java")
                .put("temperature", 0.9);
        JsonNode gen = post("/generate", genBody);
        System.out.println("Generated: " + gen.get("text").asText());
    }
}
```

---

## Error Responses

All authentication errors return HTTP `401 Unauthorized` with a JSON body.

### Missing or Malformed Header

```http
HTTP/1.1 401 Unauthorized
Content-Type: application/json

{"detail": "Missing or invalid Authorization header"}
```

**Returned when:**
- No `Authorization` header is present
- The header value does not start with `Bearer ` (case-sensitive)
- The header is present but empty (`Authorization:`)

### Invalid Token

```http
HTTP/1.1 401 Unauthorized
Content-Type: application/json

{"detail": "Invalid API key"}
```

**Returned when:**
- The Bearer token does not match the configured `AIFORGE_API_KEY`
- The token portion is empty (`Authorization: Bearer `)
- The token has leading/trailing whitespace (strict match — no trimming)

### Not an Auth Error

| Status | Body | Cause |
|--------|------|-------|
| `422` | `{"detail": [...]}` | Request body validation failed (wrong types, missing fields) |
| `429` | `{"detail": "Rate limit exceeded"}` | Too many requests to the LLM provider |
| `500` | `{"detail": "Internal server error"}` | Unhandled exception (never leaks stack traces) |
| `503` | `{"detail": "..."}` | LLM provider unreachable |

---

## Security Properties

| Property | Implementation Detail |
|----------|---------------------|
| **Constant-time comparison** | `secrets.compare_digest(token, self._api_key)` prevents timing side-channel attacks |
| **No key echo** | The API key is never returned in any response body or header |
| **No key in logs** | structlog's `_strip_secrets` processor redacts fields named `api_key`, `token`, and `authorization` |
| **Auth failure audit trail** | Every 401 is logged with `path`, `method`, and `client` IP via structlog |
| **Disabled by default** | `AIFORGE_API_KEY = None` → AuthMiddleware is a no-op (local dev friendly) |
| **Frozen public paths** | `_PUBLIC_PATHS` is a `frozenset[str]` — immutable, O(1) set-membership check, no regex ambiguity |
| **No session state** | Stateless auth — every request is independently validated |
| **Global error handler** | Unhandled exceptions return `500` with a generic `"Internal server error"` message — stack traces, file paths, and secrets are never leaked |

---

## Middleware Architecture

The FastAPI app registers three middlewares in order. Starlette processes them
**bottom-up** — the last middleware registered runs first:

```
Incoming request
      │
      ▼
┌─────────────────────────┐
│   AuthMiddleware         │  ← runs first (registered last)
│   (validates Bearer)     │
└────────────┬────────────┘
             │
┌────────────▼────────────┐
│   RequestLoggingMiddleware│  ← runs second
│   (injects X-Request-Id, │
│    logs method/path/     │
│    status/latency)       │
└────────────┬────────────┘
             │
┌────────────▼────────────┐
│   CORSMiddleware         │  ← runs third (registered first)
│   (adds CORS headers)    │
└────────────┬────────────┘
             │
             ▼
      Route handler
```

The `AuthMiddleware` receives the API key from `settings.API_KEY` (resolved via
`get_settings()`) at app startup. The key is stored once and reused for the
lifetime of the process.

---

## Request Logging & Tracing

Every HTTP request — whether authenticated or not — is logged by the
`RequestLoggingMiddleware`:

```json
{
  "event": "HTTP request",
  "method": "POST",
  "path": "/classify",
  "status": 200,
  "latency_ms": 1234.5,
  "request_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
}
```

The `request_id` UUID is:
- Generated fresh for every request
- Attached to the response as the `X-Request-Id` header
- Useful for correlating client-side errors with server-side logs

Auth failures are logged separately at `WARNING` level with the client IP:

```json
{
  "event": "Auth failure: invalid API key",
  "path": "/classify",
  "method": "POST",
  "client": "192.168.1.42"
}
```

---

## Webhook SSRF Protection

When async endpoints (`/qa/ingest`, `/search/index`) accept a `webhook_url`,
the URL is validated **before** the job is created to prevent Server-Side
Request Forgery (SSRF):

| Check | Behaviour |
|-------|-----------|
| **Scheme** | `https://` required in production; `http://` allowed only if `webhook_allow_insecure: true` in `aiforge.yaml` |
| **Localhost** | `localhost`, `127.0.0.1`, `::1`, `0.0.0.0` → rejected |
| **Private IPs** | `10.0.0.0/8`, `172.16.0.0/12`, `192.168.0.0/16`, `127.0.0.0/8`, `::1/128` → rejected |
| **DNS resolution** | Public hostnames are resolved; if any resolved IP falls in a private range → rejected |

```yaml
# aiforge.yaml — webhook SSRF settings
webhook_enabled: true
webhook_timeout: 10.0          # seconds per delivery attempt
webhook_allow_insecure: false   # allow http:// URLs (dev only!)
```

Webhook delivery uses **3 retries** with exponential backoff (1s → 2s → 4s).

---

## CORS Configuration

CORS is configured via the `AIFORGE_CORS_ORIGINS` environment variable
(comma-separated list of allowed origins):

```bash
# Single production origin
export AIFORGE_CORS_ORIGINS=https://app.example.com

# Multiple origins
export AIFORGE_CORS_ORIGINS=https://app.example.com,https://admin.example.com

# Allow all (development only)
export AIFORGE_CORS_ORIGINS=*
```

| Setting | `allow_credentials` | Notes |
|---------|-------------------|-------|
| Specific origins | `true` | Cookies / auth headers forwarded |
| `*` (wildcard) | `false` | Browsers block credentials with wildcard CORS |

The default (`*`) allows any origin — suitable for local dev. Tighten this
for any deployment accessible on a network.

---

## Docker / Production

Pass the key as an environment variable — **never** bake it into the image:

```bash
docker run -p 8000:8000 \
  -e AIFORGE_OPENAI_API_KEY=sk-... \
  -e AIFORGE_API_KEY=your-secret-key \
  -v aiforge_data:/app/aiforge_data \
  aiforge/server
```

### Docker Compose

```yaml
services:
  api:
    build: .
    ports:
      - "8000:8000"
    env_file:
      - .env          # contains AIFORGE_API_KEY + provider keys
    command: aiforge serve --host 0.0.0.0 --port 8000
    volumes:
      - aiforge_data:/app/aiforge_data

  playground:
    build: .
    ports:
      - "8501:8501"
    env_file:
      - .env
    environment:
      - AIFORGE_API_BASE=http://api:8000
    command: aiforge playground --port 8501
    depends_on:
      - api

volumes:
  aiforge_data:
```

---

## Kubernetes

Use a Kubernetes `Secret` for the API key and mount it as an environment
variable:

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: aiforge-secrets
type: Opaque
stringData:
  AIFORGE_API_KEY: "your-secret-key"
  AIFORGE_OPENAI_API_KEY: "sk-..."
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: aiforge-api
spec:
  replicas: 2
  selector:
    matchLabels:
      app: aiforge-api
  template:
    metadata:
      labels:
        app: aiforge-api
    spec:
      containers:
        - name: aiforge
          image: aiforge/server:latest
          ports:
            - containerPort: 8000
          envFrom:
            - secretRef:
                name: aiforge-secrets
          livenessProbe:
            httpGet:
              path: /health
              port: 8000
            periodSeconds: 30
          readinessProbe:
            httpGet:
              path: /health/ready
              port: 8000
            periodSeconds: 10
```

---

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| All requests return `401` | `AIFORGE_API_KEY` is set but client isn't sending the header | Add `Authorization: Bearer <key>` to every request |
| `401` even with correct key | Trailing newline or whitespace in `.env` value | Re-generate the key; ensure no extra whitespace |
| Health check returns `401` | Health endpoint path has a typo (e.g. `/healthz`) | Use exactly `/health` or `/health/ready` |
| Auth works locally but fails in Docker | Key not passed to container | Add `-e AIFORGE_API_KEY=...` or `env_file` in compose |
| `422` instead of `401` | Auth passed but request body is malformed | Fix the JSON payload; `422` is a validation error, not auth |
| Webhook URL rejected | URL targets a private IP or uses `http://` | Use an `https://` URL with a public hostname |
| CORS preflight fails | `AIFORGE_CORS_ORIGINS` doesn't include the requesting origin | Add the origin to the comma-separated list |

---

## See Also

- [API Server](api-server.md) — Full endpoint reference with request/response schemas
- [Configuration](configuration.md) — Full settings reference and layering rules
- [Modules](modules/) — Detailed SDK documentation for each module
- [Quick Start](quickstart.md) — Getting started
- [API Server](api-server.md) — endpoints, CORS, async jobs, Docker deployment
