# Contributing Models

Please keep each model branch easy to run and easy to compare.

Recommended branch naming:

```text
network/hybrid
network/transformer
network/resnet
```

Each model branch should include:

- `README.md` with a short method description.
- A train entry point.
- An evaluate entry point.
- A predict or submission entry point.
- Required shared utilities or dependency notes.

Avoid committing:

- FITS data files.
- `.pt`, `.pth`, or other checkpoints.
- `outputs/`, `runs/`, and temporary diagnostics.
- Local notebooks unless they are cleaned and intentionally documented.

