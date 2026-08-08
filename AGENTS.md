# Repository Guide

This repository publishes an independent research site with MkDocs and the
`mkdocs-simple-blog` theme.

## Structure

- `docs/`: published pages, assets, styles, and browser JavaScript.
- `code/`: supporting source linked from published articles.
- `mkdocs.yml`: navigation, theme, extensions, and asset configuration.
- `pyproject.toml`: Python requirements; this repository does not use a lockfile.
- `.github/workflows/static.yml`: GitHub Pages build and deployment.
- `.memory/`: private notes and plans; ignored by Git. Read its relevant files
  before substantial work and keep new implementation plans there.
- `site/`: generated output; never edit it by hand.

## Updating The Blog

1. Read `mkdocs.yml`, the affected page, and relevant `.memory/` context.
2. Edit content under `docs/`; keep article-specific assets under
	`docs/assets/` and supporting code under `code/`.
3. Keep public writing self-contained and cite public sources. Do not expose
	private memory or planning material in published pages.
4. Update navigation in `mkdocs.yml` when adding or removing pages.
5. Run `uv run --python 3.13 mkdocs build --strict` before finishing. Also run
	`node --check` for changed JavaScript and preview responsive or interactive
	changes.

Use `uv run --python 3.13 mkdocs serve` for local preview; `uv` resolves the
dependencies from `pyproject.toml`. Keep changes focused and preserve the
existing text-forward visual style.
