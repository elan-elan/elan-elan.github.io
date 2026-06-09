# elan-elan.github.io

Personal website and research blog for [https://elan-elan.github.io](https://elan-elan.github.io), built with MkDocs and the [mkdocs-simple-blog](https://github.com/FernandoCelmer/mkdocs-simple-blog) theme. The CVPR 2026 section turns conference notes into public, mechanism-focused writeups about geospatial AI, structured geometry, diffusion, 3D reconstruction, tokenization, and VLM reliability.

## Setup

Install the locked environment:

```bash
uv sync
```

## Preview

Run the local MkDocs server:

```bash
uv run mkdocs serve
```

Open [http://127.0.0.1:8000/](http://127.0.0.1:8000/). If that port is busy, use another one:

```bash
uv run mkdocs serve -a 127.0.0.1:8001
```

## Build

Build the static site before publishing:

```bash
uv run mkdocs build --strict
```

The generated output is written to `site/`, which is ignored by git and should not be edited by hand.

## Deploy

GitHub Pages should be configured to use **GitHub Actions**. On pushes to `main`, `.github/workflows/static.yml` installs dependencies, runs the strict MkDocs build, uploads `site/`, and deploys it.

For local checks, use the same build command as CI:

```bash
uv run mkdocs build --strict
```
