# Documentation Site

AIForge uses [MkDocs](https://www.mkdocs.org/) with the
[Material for MkDocs](https://squidfunk.github.io/mkdocs-material/) theme
to render the `docs/` directory as a polished documentation website with
navigation, full-text search, dark mode, and code-copy buttons.

## Prerequisites

Install the `site` extra (MkDocs + Material theme):

```bash
pip install -e ".[site]"
# or via Make
make install-all
```

## Local Development (Live-Reload)

Start a local server with hot-reload — every time you save a `.md` file the
browser refreshes automatically:

```bash
make docs-serve
# → http://127.0.0.1:8000
```

Or directly:

```bash
mkdocs serve                        # default port 8000
mkdocs serve -a 127.0.0.1:8100      # custom port
```

## Build Static Site

Generate the HTML into the `site/` directory:

```bash
make docs
# or
mkdocs build
```

The output is a self-contained static site you can host anywhere.

## Hosting Options

### GitHub Pages (recommended for open-source)

One-command deploy to `gh-pages` branch:

```bash
mkdocs gh-deploy --force
```

Or automate with a GitHub Actions workflow (`.github/workflows/docs.yml`):

```yaml
name: Deploy Docs
on:
  push:
    branches: [main]
    paths: ["docs/**", "mkdocs.yml"]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install "mkdocs>=1.6" "mkdocs-material>=9.5"
      - run: mkdocs gh-deploy --force
```

Once deployed, enable GitHub Pages in the repo settings → **Source: Deploy
from a branch** → **Branch: `gh-pages` / `/(root)`**.

### S3 + CloudFront

```bash
mkdocs build
aws s3 sync site/ s3://your-docs-bucket --delete
aws cloudfront create-invalidation --distribution-id XXXXX --paths "/*"
```

### Netlify / Vercel

| Setting | Value |
|---------|-------|
| Build command | `pip install mkdocs mkdocs-material && mkdocs build` |
| Publish directory | `site` |
| Python version | `3.12` |

## Configuration

The site is configured in `mkdocs.yml` at the project root. Key settings:

| Setting | Purpose |
|---------|---------|
| `site_name` | Title shown in the header and browser tab |
| `nav` | Navigation tree — add new pages here |
| `theme.palette` | Light/dark mode color scheme |
| `theme.features` | Navigation tabs, search suggestions, code copy, etc. |
| `markdown_extensions` | Admonitions, code highlighting, tabbed content |

## Adding a New Doc Page

1. Create the `.md` file under `docs/` (e.g., `docs/new-feature.md`).
2. Add an entry to the `nav` section in `mkdocs.yml`.
3. Run `make docs-serve` to preview.
