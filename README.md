# elan-elan.github.io

An independent MkDocs site for research notes on geospatial AI, structured
geometry, diffusion, 3D reconstruction, tokenization, and VLM reliability. The
site uses the
[mkdocs-simple-blog](https://github.com/FernandoCelmer/mkdocs-simple-blog)
theme and is deployed through GitHub Pages.

## Setup

Install [uv](https://docs.astral.sh/uv/). Project dependencies are declared in
`pyproject.toml` and resolved automatically by `uv run`.

Python 3.13 is recommended.

## Preview

Run the local MkDocs server:

```bash
uv run --python 3.13 mkdocs serve
```

Open [http://127.0.0.1:8000/](http://127.0.0.1:8000/). If that port is busy, use another one:

```bash
uv run --python 3.13 mkdocs serve -a 127.0.0.1:8001
```

## Build

Build the static site before publishing:

```bash
uv run --python 3.13 mkdocs build --strict
```

The generated output is written to `site/`, which is ignored by git and should not be edited by hand.

## Deploy

GitHub Pages should be configured to use **GitHub Actions**. On pushes to `main`, `.github/workflows/static.yml` installs dependencies, runs the strict MkDocs build, uploads `site/`, and deploys it.

For local checks, use the same build command as CI:

```bash
uv run --python 3.13 mkdocs build --strict
```
