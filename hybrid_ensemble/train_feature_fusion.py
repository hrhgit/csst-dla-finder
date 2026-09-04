#!/usr/bin/env python3
"""Train one feature-level, dual-backbone DLA fusion model."""
from __future__ import annotations

from pathlib import Path
import argparse
import json
import sys

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "hybrid_ensemble"))
sys.path.insert(0, "/home/heruihua")

from csst_dla.scoring import labels_to_truth, score_catalog
from decode import decode_validation_catalog
from evaluate_hybrid import load_checkpoint, resolve_device
from feature_fusion import DualFusionTrainDataset, DualTowerFusionNet
from csst_dla_wzx_pkg.inference import load_model_from_checkpoint


def focal_bce_with_logits(logits, target, alpha=0.85, gamma=2.0, weight=None):
    bce = nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none")
    prob = torch.sigmoid(logits)
    p_t = prob * target + (1.0 - prob) * (1.0 - target)
    alpha_t = alpha * target + (1.0 - alpha) * (1.0 - target)
    loss = alpha_t * (1.0 - p_t).pow(gamma) * bce
    if weight is not None:
        return (loss * weight).sum() / weight.sum().clamp_min(1.0)
    return loss.mean()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", required=True)
    parser.add_argument("--train-fits", required=True)
    parser.add_argument("--dilated-checkpoint", required=True)
    parser.add_argument("--wzx-checkpoint", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--merge-mode", choices=["plain", "residual_dilated", "residual_wzx"], required=True)
    parser.add_argument("--fusion-width", type=int, default=128)
    parser.add_argument("--fusion-depth", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.45)
    parser.add_argument("--min-z-dla", type=float, default=1.10)
    parser.add_argument("--region-loss-weight", type=float, default=0.2)
    parser.add_argument("--lognhi-loss-weight", type=float, default=0.05)
    parser.add_argument("--offset-loss-weight", type=float, default=0.1)
    parser.add_argument("--count-loss-weight", type=float, default=0.25)
    parser.add_argument("--high-lognhi-threshold", type=float, default=22.0)
    parser.add_argument("--high-lognhi-center-weight", type=float, default=4.0)
    parser.add_argument("--high-lognhi-log-weight", type=float, default=4.0)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-val-samples", type=int)
    parser.add_argument("--train-backbones", action="store_true", help="Unfreeze both pretrained towers; off by default.")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu", "mps"], default="auto")
    return parser.parse_args()


def loss_for_batch(model, batch, device, args):
    hybrid, wzx, zq, center, region, lognhi, mask, offset, offset_weight, count, _ = batch
    hybrid = hybrid.to(device, non_blocking=True)
    wzx = wzx.to(device, non_blocking=True)
    zq = zq.to(device, non_blocking=True)
    center = center.to(device, non_blocking=True)
    region = region.to(device, non_blocking=True)
    lognhi = lognhi.to(device, non_blocking=True)
    mask = mask.to(device, non_blocking=True)
    offset = offset.to(device, non_blocking=True)
    offset_weight = offset_weight.to(device, non_blocking=True)
    count = count.to(device, non_blocking=True)
    output = model(hybrid, wzx, zq)
    high_mask = ((lognhi >= args.high_lognhi_threshold) & (mask > 0)).float()
    center_weight = 1.0 + (args.high_lognhi_center_weight - 1.0) * high_mask
    center_loss = focal_bce_with_logits(output["center_logits"], center, weight=center_weight)
    region_loss = focal_bce_with_logits(output["region_logits"], region, alpha=0.75)
    if mask.sum() > 0:
        log_weight = mask * (1.0 + (args.high_lognhi_log_weight - 1.0) * high_mask)
        log_loss = (((20.3 + output["lognhi_raw"] - lognhi) ** 2) * log_weight).sum() / log_weight.sum().clamp_min(1.0)
    else:
        log_loss = torch.zeros((), device=device)
    if offset_weight.sum() > 0:
        offset_loss = (((output["offset_raw"] - offset) ** 2) * offset_weight).sum() / offset_weight.sum().clamp_min(1.0)
    else:
        offset_loss = torch.zeros((), device=device)
    count_loss = nn.functional.cross_entropy(output["count_logits"], count)
    loss = (
        center_loss
        + args.region_loss_weight * region_loss
        + args.lognhi_loss_weight * log_loss
        + args.offset_loss_weight * offset_loss
        + args.count_loss_weight * count_loss
    )
    return loss, output


@torch.inference_mode()
def predict_validation(model, loader, device):
    model.eval()
    heat, lognhi, offset, count_logits, rows = [], [], [], [], []
    for batch in loader:
        hybrid, wzx, zq = batch[:3]
        output = model(hybrid.to(device), wzx.to(device), zq.to(device))
        heat.append(torch.sigmoid(output["center_logits"]).cpu().numpy())
        lognhi.append((20.3 + output["lognhi_raw"]).cpu().numpy())
        offset.append(output["offset_raw"].cpu().numpy())
        count_logits.append(output["count_logits"].cpu().numpy())
        rows.append(np.asarray(batch[-1], dtype=np.int64))
    return {
        "heatmap": np.concatenate(heat),
        "lognhi": np.concatenate(lognhi),
        "offset": np.concatenate(offset),
        "count_logits": np.concatenate(count_logits),
        "rows": np.concatenate(rows),
    }


def save_checkpoint(path: Path, model, args, epoch_row: dict, dilated_config: dict, wzx_config: dict) -> None:
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": {
                "merge_mode": args.merge_mode,
                "fusion_width": args.fusion_width,
                "fusion_depth": args.fusion_depth,
                "freeze_backbones": not args.train_backbones,
                "dilated_checkpoint": args.dilated_checkpoint,
                "wzx_checkpoint": args.wzx_checkpoint,
                "dilated_config": dilated_config,
                "wzx_config": wzx_config,
            },
            "threshold": args.threshold,
            "score": epoch_row,
            "training_config": vars(args),
        },
        path,
    )


def score_validation(model, val_loader, val_ds, device, args, epoch: int, loss: float | None = None) -> dict:
    prediction = predict_validation(model, val_loader, device)
    catalog = decode_validation_catalog(
        prediction,
        val_ds.indices,
        val_ds.labels,
        val_ds.wavelength,
        threshold=args.threshold,
        min_distance=10,
        min_z_dla=args.min_z_dla,
        lognhi_min=20.3,
        lognhi_max=22.5,
    )
    truth = labels_to_truth(val_ds.labels, val_ds.indices, min_lognhi=20.3)
    score = score_catalog(truth, catalog)
    return {
        "epoch": epoch,
        "loss": loss,
        "final_score": score.final_score,
        "detection_score": score.detection_score,
        "parameter_score": score.parameter_score,
        "completeness": score.completeness,
        "purity": score.purity,
        "n_pred": score.n_pred,
        "n_match": score.n_match,
        "std_dv": score.std_dv,
        "std_dlognhi": score.std_dlognhi,
    }


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    print(json.dumps({"stage": "loading_datasets", "merge_mode": args.merge_mode}, ensure_ascii=False), flush=True)
    train_ds = DualFusionTrainDataset(args.targets, args.train_fits, "train", args.max_train_samples)
    val_ds = DualFusionTrainDataset(args.targets, args.train_fits, "val", args.max_val_samples)
    print(json.dumps({"stage": "datasets_ready", "train": len(train_ds), "val": len(val_ds)}, ensure_ascii=False), flush=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=False)
    dilated_wrapper, dilated_config = load_checkpoint(args.dilated_checkpoint, device)
    wzx_model, wzx_config = load_model_from_checkpoint(args.wzx_checkpoint, device)
    model = DualTowerFusionNet(
        dilated_wrapper.model,
        wzx_model,
        merge_mode=args.merge_mode,
        width=args.fusion_width,
        depth=args.fusion_depth,
        freeze_backbones=not args.train_backbones,
    ).to(device)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr, weight_decay=1e-4)
    history = []
    best_score = -float("inf")
    # Residual modes are intentionally initialized as a source model plus a
    # zero correction.  Record this epoch-0 control so training cannot hide a
    # regression by overwriting the usable base checkpoint.
    if args.merge_mode != "plain":
        initial = score_validation(model, val_loader, val_ds, device, args, epoch=0)
        history.append(initial)
        best_score = initial["final_score"]
        save_checkpoint(out_dir / "initial_model.pt", model, args, initial, dilated_config, wzx_config)
        print(json.dumps(initial, ensure_ascii=False), flush=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        progress_every = max(1, len(train_loader) // 5)
        for batch_index, batch in enumerate(train_loader, start=1):
            loss, _ = loss_for_batch(model, batch, device, args)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * len(batch[0])
            if batch_index == 1 or batch_index % progress_every == 0:
                print(json.dumps({"progress": "train", "epoch": epoch, "batch": batch_index, "total_batches": len(train_loader)}, ensure_ascii=False), flush=True)
        row = score_validation(model, val_loader, val_ds, device, args, epoch=epoch, loss=total_loss / len(train_ds))
        history.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        if row["final_score"] > best_score:
            best_score = row["final_score"]
            save_checkpoint(out_dir / "best_model.pt", model, args, row, dilated_config, wzx_config)
    save_checkpoint(out_dir / "model.pt", model, args, history[-1], dilated_config, wzx_config)
    (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (out_dir / "training_args.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    print(f"wrote {out_dir / 'best_model.pt'}")


if __name__ == "__main__":
    main()
