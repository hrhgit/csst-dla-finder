# Dilated ResNet Five-Head Ensemble

This branch keeps the hybrid pipeline's data preparation, decoding, evaluation,
and prediction scripts, while replacing its CNN with the local dilated
five-head residual architecture:

- multi-channel 1D spectrum features
- 1D dilated residual backbone with layer normalization
- heatmap, broad-region, `LOGNHI`, count, and optional offset heads
- weighted ensemble evaluation and prediction
- configurable DLA redshift lower bound via `--min-z-dla`

No data files or trained checkpoints are included.

## Expected FITS Layout

Training FITS:

- `WAVELENGTH`
- `FLUX`
- `FLUX_CLEAN`
- `LABELS`

Test FITS:

- `WAVELENGTH`
- `FLUX`
- `FLUX_CLEAN`
- `META`

The lightweight FITS reader in `src/csst_dla/fits_utils.py` expects the same
fixed challenge table layout used by this project.

## Prepare Targets

Create a split:

```bash
PYTHONPATH=src python3 scripts/make_split.py \
  --train-fits /path/to/train.fits \
  --out splits/split_seed42.npz \
  --seed 42
```

Create dense targets. For lower-resolution spectra, reduce pixel radii so the
physical wavelength width stays comparable.

```bash
PYTHONPATH=src python3 scripts/make_cnn_targets.py \
  --train-fits /path/to/train.fits \
  --split splits/split_seed42.npz \
  --out outputs/cnn_targets_seed42.npz \
  --sigma-pixels 1.0 \
  --low-lognhi-radius-pixels 3 \
  --mid-lognhi-radius-pixels 6 \
  --high-lognhi-radius-pixels 13 \
  --very-high-lognhi-radius-pixels 20
```

## Train Members

For a low-redshift dataset such as `QSO=1.10-2.35`, use `--min-z-dla 1.10`.
For the original higher-redshift range, the default `--min-z-dla 1.55` can be
left unchanged.

```bash
PYTHONPATH=src python3 hybrid_ensemble/train_hybrid.py \
  --targets outputs/cnn_targets_seed42.npz \
  --train-fits /path/to/train.fits \
  --out-dir hybrid_ensemble/runs/member_all_seed42 \
  --input-mode all \
  --hidden 96 \
  --num-blocks 4 \
  --epochs 25 \
  --batch-size 128 \
  --lr 1e-3 \
  --seed 42 \
  --threshold 0.40 \
  --min-z-dla 1.10 \
  --count-loss-weight 0.35 \
  --region-loss-weight 0.2 \
  --lognhi-loss-weight 0.05 \
  --offset-loss-weight 0.05 \
  --device auto
```

Train diverse members by changing `--input-mode` and `--seed`, for example
`residual` with seed `43` and `flux` with seed `44`.

## Evaluate Ensemble

```bash
PYTHONPATH=src python3 hybrid_ensemble/evaluate_hybrid.py \
  --models \
    hybrid_ensemble/runs/member_all_seed42/best_model.pt \
    hybrid_ensemble/runs/member_residual_seed43/best_model.pt \
    hybrid_ensemble/runs/member_flux_seed44/best_model.pt \
  --weights 1.0 1.0 0.7 \
  --targets outputs/cnn_targets_seed42.npz \
  --train-fits /path/to/train.fits \
  --threshold 0.25 \
  --min-distance 3 \
  --min-z-dla 1.10 \
  --count-bias -0.8 1.0 0.4 \
  --count-min-prob 0.0 \
  --soft-radius 1 \
  --soft-power 3.0 \
  --fit-lognhi-calibration \
  --out hybrid_ensemble/runs/ensemble_eval.json \
  --device auto
```

The evaluator writes both:

- `ensemble_eval.json`
- `ensemble_eval_config.json`

The config file is consumed by prediction.

## Predict

```bash
PYTHONPATH=src python3 hybrid_ensemble/predict_hybrid.py \
  --ensemble-config hybrid_ensemble/runs/ensemble_eval_config.json \
  --test-fits /path/to/test.fits \
  --out hybrid_ensemble/runs/submission_hybrid.csv \
  --device auto
```
