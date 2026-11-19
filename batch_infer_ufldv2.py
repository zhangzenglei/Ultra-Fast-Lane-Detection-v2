#!/usr/bin/env python3
"""Batch UFLDv2 inference, optionally comparing recorded undistorted contours."""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps
from torchvision import transforms
from tqdm import tqdm

from utils.config import Config
from utils.lane_comparison import (
    CameraCalibration, RecordedContourIndex,
    draw_comparison_legend, draw_recorded_contours,
)
from model.model_culane import get_model as get_culane_model
from model.model_tusimple import get_model as get_tusimple_model

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
COLORS = [(0, 255, 0), (0, 165, 255), (255, 255, 0), (255, 0, 255)]


def get_anchors(cfg):
    if cfg.dataset == "CULane":
        row = np.linspace(0.42, 1.0, cfg.num_row)
    elif cfg.dataset == "Tusimple":
        row = np.linspace(160, 710, cfg.num_row) / 720.0
    else:
        raise ValueError("Only CULane and Tusimple configs are supported")
    return row, np.linspace(0, 1, cfg.num_col)


def decode(pred, row_anchor, col_anchor, width, height, local_width=1):
    """Match the repo's demo.py pred2coords() decoding, retaining lane IDs."""
    loc_row = pred["loc_row"].detach().float().cpu()
    loc_col = pred["loc_col"].detach().float().cpu()
    ext_row = pred["exist_row"].detach().float().cpu().argmax(1)[0]
    ext_col = pred["exist_col"].detach().float().cpu().argmax(1)[0]
    max_row = loc_row.argmax(1)[0]
    max_col = loc_col.argmax(1)[0]
    n_grid_row = loc_row.shape[1]
    n_grid_col = loc_col.shape[1]
    n_cls_row = loc_row.shape[2]
    n_cls_col = loc_col.shape[2]
    lanes = []

    for lane_id in (1, 2):
        points = []
        if int(ext_row[:, lane_id].sum().item()) > n_cls_row / 2:
            for k in range(n_cls_row):
                if not bool(ext_row[k, lane_id]):
                    continue
                index = int(max_row[k, lane_id])
                indices = torch.arange(max(0, index - local_width), min(n_grid_row - 1, index + local_width) + 1)
                logits = loc_row[0, indices, k, lane_id]
                weighted = (torch.softmax(logits, dim=0) * indices.float()).sum() + 0.5
                x = int(weighted.item() / (n_grid_row - 1) * width)
                y = int(row_anchor[k] * height)
                points.append([min(width - 1, max(0, x)), min(height - 1, max(0, y))])
        lanes.append({"lane_id": lane_id, "points": points})

    for lane_id in (0, 3):
        points = []
        if int(ext_col[:, lane_id].sum().item()) > n_cls_col / 4:
            for k in range(n_cls_col):
                if not bool(ext_col[k, lane_id]):
                    continue
                index = int(max_col[k, lane_id])
                indices = torch.arange(max(0, index - local_width), min(n_grid_col - 1, index + local_width) + 1)
                logits = loc_col[0, indices, k, lane_id]
                weighted = (torch.softmax(logits, dim=0) * indices.float()).sum() + 0.5
                y = int(weighted.item() / (n_grid_col - 1) * height)
                x = int(col_anchor[k] * width)
                points.append([min(width - 1, max(0, x)), min(height - 1, max(0, y))])
        lanes.append({"lane_id": lane_id, "points": points})
    return lanes


def load_checkpoint(path):
    # weights_only=True avoids unpickling arbitrary Python objects. If the
    # checkpoint is older and needs pickle, convert it in a trusted environment.
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
    if not isinstance(state, dict):
        raise TypeError("Checkpoint does not contain a model state dict")
    return {k.removeprefix("module."): v for k, v in state.items()}


def infer(args):
    if args.recorded_contours and not args.camera_param:
        raise ValueError("--recorded-contours requires --camera-param")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    calibration = CameraCalibration(args.camera_param) if args.camera_param else None
    recorded_index = (RecordedContourIndex(args.recorded_contours, calibration.camera_id,
                                           args.match_tolerance_ms)
                      if args.recorded_contours else None)
    cfg = Config.fromfile(args.config)
    if cfg.dataset not in ("CULane", "Tusimple"):
        raise ValueError("Use a CULane or Tusimple model config")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU unavailable. Check nvidia-smi and PyTorch CUDA install")

    print(f"Loading {cfg.dataset} {cfg.backbone=}, checkpoint: {args.weights}")
    model = (get_culane_model if cfg.dataset == "CULane" else get_tusimple_model)(cfg)
    model.load_state_dict(load_checkpoint(args.weights), strict=True)
    model.eval()
    row_anchor, col_anchor = get_anchors(cfg)
    prep = transforms.Compose([
        transforms.Resize((int(cfg.train_height / cfg.crop_ratio), cfg.train_width)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    src = Path(args.input).expanduser().resolve()
    dst = Path(args.output).expanduser().resolve()
    dst.mkdir(parents=True, exist_ok=True)
    if src.is_file():
        image_paths = [src]
    elif src.is_dir():
        image_paths = sorted(p for p in src.rglob("*") if p.is_file() and p.suffix.lower() in EXTENSIONS)
    else:
        raise FileNotFoundError(src)
    if not image_paths:
        raise RuntimeError("No images found")
    if args.limit is not None:
        image_paths = image_paths[:args.limit]
    done, failed, elapsed = 0, 0, 0.0
    matched, unmatched = 0, 0
    for path in tqdm(image_paths, desc="UFLDv2"):
        try:
            with Image.open(path) as raw:
                if calibration and raw.getexif().get(274, 1) != 1:
                    raise ValueError("Calibrated images must not have EXIF rotation/reflection")
                image = ImageOps.exif_transpose(raw).convert("RGB")
                width, height = image.size
                if calibration:
                    image = Image.fromarray(calibration.undistort(np.asarray(image)))
                # Inference and visualization use the same undistorted image.
                tensor = prep(image)[:, -int(cfg.train_height):, :].unsqueeze(0).cuda()
                vis = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR).copy()
            recorded = (recorded_index.match(path, calibration.scale(width, height))
                        if recorded_index else None)
            t0 = time.perf_counter()
            with torch.inference_mode():
                output = model(tensor)
            torch.cuda.synchronize()
            elapsed += time.perf_counter() - t0
            lanes = decode(output, row_anchor, col_anchor, width, height)
            for lane in lanes:
                points = lane["points"]
                if not points:
                    continue
                color = (0, 165, 255) if recorded is not None else COLORS[lane["lane_id"]]
                # Dots show true sampled predictions; connect for easier review.
                cvpts = np.asarray(points, dtype=np.int32).reshape(-1, 1, 2)
                if len(points) >= 2:
                    cv2.polylines(vis, [cvpts], False, color, 2, cv2.LINE_AA)
                for x, y in points:
                    cv2.circle(vis, (x, y), 3, color, -1, cv2.LINE_AA)
            if recorded is not None:
                draw_recorded_contours(vis, recorded)
                draw_comparison_legend(vis, recorded)
            relative = path.relative_to(src) if src.is_dir() else Path(path.name)
            target = dst / relative.parent / (relative.stem + "_lanes.jpg")
            target.parent.mkdir(parents=True, exist_ok=True)
            ok, buf = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if not ok:
                raise RuntimeError("JPEG encode failed")
            buf.tofile(str(target))
            data = {"source": str(path), "width": width, "height": height,
                    "model_dataset": cfg.dataset, "lanes": lanes,
                    "coordinate_space": "undistorted" if calibration else "input_image"}
            if calibration:
                data["calibration"] = calibration.metadata(width, height)
            if recorded is not None:
                data["recorded"] = recorded
            target.with_suffix(".json").write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
            done += 1
            if recorded is not None:
                matched += int(recorded["matched"])
                unmatched += int(not recorded["matched"])
        except Exception as exc:
            failed += 1
            print(f"\n[FAIL] {path}: {exc}", file=sys.stderr)
    print(f"Saved {done} images + JSON under {dst}; failed={failed}; mean model inference={elapsed / max(done,1)*1000:.1f} ms/image (excludes I/O & decoding)")
    if recorded_index:
        print(f"Recorded contours: matched={matched}, unmatched={unmatched}; tolerance={args.match_tolerance_ms:g} ms")
    return 1 if failed else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UFLDv2 inference on an image or directory")
    parser.add_argument("--config", default="configs/culane_res34.py")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--camera-param", help="Fixed camera calibration JSON; undistort in memory")
    parser.add_argument("--recorded-contours", help="Directory of recorded front-mono contour JSON files")
    parser.add_argument("--match-tolerance-ms", type=float, default=1.0,
                        help="Maximum image/contour timestamp difference (default: 1 ms)")
    parser.add_argument("--limit", type=int, help="Process only the first N images (for a preview)")
    sys.exit(infer(parser.parse_args()))
