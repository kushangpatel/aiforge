# Security Policy

## Supported versions

AI Forge is in 0.x. Security fixes go into the latest release only; please
upgrade to it before reporting.

## Reporting a vulnerability

Please do not open a public issue for a security problem.

Report it privately through GitHub instead:
[Report a vulnerability](https://github.com/kushangpatel/aiforge/security/advisories/new)
(the **Security** tab of this repository, then **Report a vulnerability**).

Include what an attacker can do, the affected version, and the steps or code
to reproduce it. You should get a first response within a week. Once a fix is
released, the advisory is published with credit to you, unless you prefer
not to be named.

## Scope

In scope: the `aiforge` package published on PyPI as `aiforge-kit`, including
the optional REST server, playground and MCP client.

Out of scope: vulnerabilities in LLM providers or third-party dependencies
themselves (report those upstream), and prompt-injection results from a model
that the documented guardrails do not claim to stop.
