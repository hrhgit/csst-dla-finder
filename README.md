# CSST DLA Model Zoo

This repository is a lightweight model-zoo workspace for CSST DLA detection
experiments. Each network can live on its own branch, while shared data format
and evaluation conventions stay documented on `main`.

## Branch Layout

- `main`: repository overview, contribution rules, and shared conventions.
- `network/hybrid`: the hybrid ensemble implementation.
- `network/<name>`: proposed branch naming pattern for other submitted models.

For a new model, branch from `main`:

```bash
git checkout main
git checkout -b network/your-model-name
```

Keep model-specific code self-contained and document the expected train,
evaluate, and predict commands in that branch's README.

## Data Policy

Do not commit challenge data, model checkpoints, generated predictions, or run
outputs. The `.gitignore` excludes common large artifacts such as FITS files,
PyTorch checkpoints, and output folders.

