# Changelog

All notable changes to AI Forge are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
While the version is 0.x, minor releases may contain breaking changes; they are
called out under **Changed** or **Removed**.

## [Unreleased]

### Added

- CONTRIBUTING, SECURITY and CODE_OF_CONDUCT files, plus issue and pull
  request templates, for the public repository.

### Fixed

- README links now point at absolute GitHub URLs, so they work on PyPI.
- Build & Publish guide: the GitHub Packages instructions are removed, since
  GitHub Packages has no Python registry.

## [0.2.2] - 2026-10-15

### Changed

- Updated the README and the documentation site for the 0.2 features: installing
  from PyPI as `aiforge-kit` with the `gemini`, `bedrock` and `mistral` extras,
  Guardrails, Evals (`aiforge eval run` / `compare`), hybrid retrieval with
  reranking and chunking strategies, the new settings, Gemini/Bedrock/Mistral
  pricing tables and the architecture diagrams.
- The documentation site now uses GitHub-style heading anchors, so in-page
  links work on GitHub and on the site alike, and builds cleanly with
  `mkdocs build --strict`.

## [0.2.1] - 2026-10-15

### Changed

- Updated the package documentation link to the hosted AI Forge documentation
  at [kushangpatel.com/docs/aiforge](https://kushangpatel.com/docs/aiforge/).

### Removed

- Removed internal development guides, planning documents and the product
  requirements document from the public repository. User-facing documentation
  remains available in `docs/` and on the hosted documentation site.

## [0.2.0] - 2026-10-15

First release published to PyPI, as `aiforge-kit` (`pip install aiforge-kit`;
the import name is still `aiforge`).

### Added

- Evaluations (`aiforge.evals`): datasets from lists, JSONL, CSV or YAML;
  built-in scorers (exact match, contains, regex, JSON match, numeric, label,
  embedding similarity, LLM-as-judge) plus custom scorers; reports with pass
  rates, cost and latency, and run-to-run comparison; and `aiforge eval run` /
  `aiforge eval compare`, which exit non-zero on a failed threshold or a
  regression, so they can gate CI.
- Guardrails (`aiforge.guardrails`): a `Guard` with input and output checks for
  PII and secrets (detection, masking, reversible placeholders, hashing),
  prompt injection, length, banned terms, regex, schema and system-prompt
  leaks; `Guard.wrap()` guards any module or callable. The Moderator's `pii`
  check uses the new detector and accepts the new PII types.
- Hybrid retrieval for DocQA and SemanticSearch: `retrieval_mode` of
  `vector` (default), `keyword` (BM25) or `hybrid` (reciprocal rank fusion).
- Reranking: an `LLMReranker` and a hook for your own scorer, such as a
  cross-encoder (`rerank`, `rerank_top_n`, `fetch_k`).
- Chunking strategies `recursive` (default), `markdown` (keeps the heading
  path) and `token`; DocQA citations carry source, page and heading path.
- Google Gemini, Amazon Bedrock and Mistral providers (LLM and embeddings), via
  the `[gemini]`, `[bedrock]` and `[mistral]` extras, with current pricing.
- Native tool calling: providers accept `tools=` and return structured
  `ToolCall` objects. `ChatAgent` uses it on providers that support it and
  keeps the prompted JSON protocol for custom providers.
- Native structured output: providers accept `response_format=` (Pydantic model
  or JSON schema) and use each provider's `json_schema` mode, falling back to a
  plain call when a model or schema is not supported. `Extractor`, `Classifier`
  (labels constrained to the taxonomy) and `Moderator` use it.
- Multimodal messages: messages can carry content blocks (images, files),
  assistant tool calls and tool results, with builders in
  `aiforge.core.messages`.
- Prompt caching: cache reads and writes are reported on `CompletionResult` and
  priced at the discounted rates; the system prompt and tools are cached on
  Anthropic by default (`PROMPT_CACHE=system|auto|off`), with
  `cache_read`/`cache_write` pricing overrides.
- MCP Streamable HTTP transport (`streamable_http`, alias `http`) with optional
  headers.
- `[anthropic]` extra (`[anthropicai]` is kept as an alias).
- Current Claude models (Opus, Sonnet and Haiku 5.5) in the model router, with
  pricing and context windows.
- `uv.lock` with the exact versions CI tests against; Dependabot updates for the
  lockfile and GitHub Actions.
- CI: ruff lint and format check, `mypy --strict`, unit tests on Python 3.12 and
  3.13, a job on the lowest allowed dependency versions, and a job that builds
  the wheel and installs it into a clean environment.
- Release workflow publishing to PyPI with Trusted Publishing.
- `scripts/smoke_live.py`, a live smoke test against the OpenAI and Anthropic
  APIs.
- Incremental DocQA indexing hardening and vector-store resilience.

### Changed

- Providers share one LangChain-backed implementation (`LangChainLLMProvider`).
- Reply text is read with `.text`, so Claude replies that include thinking
  blocks no longer come back as a stringified block list.
- `temperature`, `top_p` and `top_k` are dropped for models that reject them
  (Claude Opus 4.7+, Claude 5.x, OpenAI o-series and GPT-5+).
- Dependency floors are the oldest versions verified in CI, and AI-stack
  packages are capped below their next major. Notably `typer>=0.15.4`,
  `chromadb>=1.3`, `langchain-anthropic>=1.4`, `qdrant-client>=1.10`,
  `mcp>=2.0`.
- The `[pinecone]` extra installs `pinecone` instead of the deprecated
  `pinecone-client`; PDF and DOCX loading uses `pypdf` and `python-docx`
  directly.
- The `sse` MCP transport still works but emits a `DeprecationWarning`.
- Providers reject `tools=` combined with `response_format=`.
- The package passes `mypy --strict` and is formatted with `ruff format`.
- The version is defined once, in `aiforge/__version__.py`.
- Usage docs updated for all of the above.

### Removed

- Retired Claude 3.x models and `gpt-3.5-turbo` from the model router.

### Fixed

- MCP tools failed on mcp 2.x, which renamed `Tool.inputSchema` to
  `input_schema`.
- `QdrantVectorStore.query` called `QdrantClient.search`, which qdrant-client
  1.16 removed; it now uses `query_points`.
- `PineconeVectorStore.list_all_metadata` broke on current Pinecone SDKs, whose
  `list()` yields pages rather than items.
- The YAML settings source called a method that does not exist.
- `jsonschema` was imported by `Extractor` but not declared as a dependency.
- GPT-4o pricing.
- Latency is measured with `perf_counter`, so fast calls no longer report
  `latency_ms=0.0` on Windows.
- Custom pricing overrides passed as module kwargs were ignored.
- The wheel's `LICENSE` file was empty.

## [0.1.0]

Initial development version. Not published to PyPI.

[Unreleased]: https://github.com/kushangpatel/aiforge/compare/v0.2.2...HEAD
[0.2.2]: https://github.com/kushangpatel/aiforge/releases/tag/v0.2.2
[0.2.1]: https://github.com/kushangpatel/aiforge/releases/tag/v0.2.1
[0.2.0]: https://github.com/kushangpatel/aiforge/releases/tag/v0.2.0
