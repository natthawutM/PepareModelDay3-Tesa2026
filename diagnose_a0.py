"""Measure A0's binary foreground before watershed and diagnose final instances."""
import argparse
import csv
import json
from pathlib import Path
from statistics import mean

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from data import Coco, letterbox, normalize
from infer import Predictor
from run_test import ROOT, evaluate


def foreground_iou(predicted, expected):
    intersection = np.count_nonzero(predicted & expected)
    union = np.count_nonzero(predicted | expected)
    return intersection / union if union else 1.0


def diagnose(source, masks_folder, predictor, output):
    rows = []
    overlaps_for_panels = {}
    confidence = predictor.manifest["watershed"]["foreground"]
    if not 0 < confidence < 1:
        raise ValueError("Foreground threshold must be strictly between 0 and 1")
    threshold = np.log(confidence / (1 - confidence))
    for index, image in enumerate(source.images):
        rgb, instances = source.load(index)
        name = image["file_name"]
        ids = cv2.imread(str(masks_folder / f"{Path(name).stem}.png"), cv2.IMREAD_UNCHANGED)
        canvas, (top, left, height, width) = letterbox(rgb, predictor.size)
        logits = predictor.segmenter.run(None, {"image": normalize(canvas)[None]})[0][0, 0]
        raw = (logits[top:top + height, left:left + width] >= threshold).astype(np.uint8)
        raw = cv2.resize(raw, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
        truth = np.zeros(ids.shape, dtype=bool)
        masks = [mask for _, _, mask in instances if mask.any()]
        n = int(ids.max())
        overlap = np.zeros((len(masks), n), dtype=np.int64)
        for i, mask in enumerate(masks):
            truth |= mask
            overlap[i] = np.bincount(ids[mask], minlength=n + 1)[1:]
        gt_area = np.array([np.count_nonzero(mask) for mask in masks], dtype=np.int64)
        pred_area = np.bincount(ids.ravel(), minlength=n + 1)[1:]
        # Ignore narrow boundary overlaps when flagging probable merges and splits.
        gt_coverage = overlap / gt_area[:, None]
        pred_coverage = overlap / pred_area[None, :].clip(min=1)
        merges = np.count_nonzero(((gt_coverage >= 0.20) & (pred_coverage >= 0.10)).sum(axis=0) >= 2)
        splits = np.count_nonzero(((gt_coverage >= 0.10) & (pred_coverage >= 0.20)).sum(axis=1) >= 2)
        matrix = overlap / (gt_area[:, None] + pred_area[None, :].clip(min=1) - overlap)
        rr, cc = linear_sum_assignment(matrix, maximize=True)
        pairs = matrix[rr, cc]
        pairs = pairs[pairs > 0]
        foreground = ids > 0
        rows.append({
            "filename": name, "pile": "_pile_" in name.lower(),
            "gt": len(masks), "predicted": n, "matched": len(pairs),
            "raw_foreground_iou": foreground_iou(raw, truth),
            "final_foreground_iou": foreground_iou(foreground, truth),
            "instance_iou": float(pairs.sum() / max(len(masks), n)) if masks or n else 1.0,
            "matched_mean_iou": float(pairs.mean()) if len(pairs) else None,
            "matched_iou_075": int((pairs >= 0.75).sum()),
            "matched_iou_090": int((pairs >= 0.90).sum()),
            "possible_merges": int(merges), "possible_splits": int(splits),
            "final_missing_pixels": int(np.count_nonzero(truth & ~foreground)),
            "final_extra_pixels": int(np.count_nonzero(foreground & ~truth)),
        })
        difference = np.zeros_like(rgb)
        difference[truth & foreground] = (0, 170, 0)
        difference[truth & ~foreground] = (255, 65, 65)
        difference[foreground & ~truth] = (65, 110, 255)
        splits_view = np.zeros_like(rgb)
        for i in range(1, n + 1):
            splits_view[ids == i] = np.random.default_rng(i).integers(64, 256, 3)
        gt_view = rgb.copy()
        for mask in masks:
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(gt_view, contours, -1, (0, 255, 255), 1)
        sections = []
        for label, part in (("Ground truth outlines", gt_view),
                            ("Overlap / Missing / Extra", difference),
                            ("Predicted IDs", splits_view)):
            resized = cv2.resize(part, (320, 320), interpolation=cv2.INTER_NEAREST)
            section = cv2.copyMakeBorder(resized, 26, 0, 0, 0, cv2.BORDER_CONSTANT)
            cv2.putText(section, label, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            sections.append(section)
        panel = np.hstack(sections)
        overlaps_for_panels[name] = cv2.cvtColor(panel, cv2.COLOR_RGB2BGR)

    worst = sorted(rows, key=lambda row: row["instance_iou"])[:5]
    for row in worst:
        name = row["filename"]
        cv2.imwrite(str(output / f"{Path(name).stem}_comparison.jpg"), overlaps_for_panels[name])
    return rows


def summarize(rows):
    if not rows:
        return None
    counts = ("gt", "predicted", "matched", "matched_iou_075", "matched_iou_090",
              "possible_merges", "possible_splits", "final_missing_pixels", "final_extra_pixels")
    ious = ("raw_foreground_iou", "final_foreground_iou", "instance_iou")
    matched_rows = [r["matched_mean_iou"] for r in rows if r["matched_mean_iou"] is not None]
    return {"images": len(rows), **{k: mean(r[k] for r in rows) for k in ious},
            **{k: sum(r[k] for r in rows) for k in counts},
            "mean_per_image_matched_iou": mean(matched_rows) if matched_rows else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--weights", type=Path, default=ROOT / "models/A0")
    parser.add_argument("--watershed", type=Path)
    parser.add_argument("--data-root", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path, default=ROOT / "outputs_a0_diagnosis")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    # Validate the saved run before comparing against the raw foreground.
    previous = evaluate(args.predictions, "valid", args.data_root)
    settings = json.loads(args.watershed.read_text(encoding="utf-8")) if args.watershed else None
    predictor = Predictor(args.weights, "selected", 4, watershed=settings)
    if predictor.manifest.get("family") != "unet":
        parser.error("--weights must point to a U-Net runtime")
    rows = diagnose(Coco(args.data_root, "valid"), args.predictions / "seg_mask", predictor, args.out)
    assert abs(mean(r["instance_iou"] for r in rows) - previous["instance_iou"]) < 1e-9
    with (args.out / "per_image.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "all": summarize(rows),
        "pile_by_filename": summarize([r for r in rows if r["pile"]]),
        "other": summarize([r for r in rows if not r["pile"]]),
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
