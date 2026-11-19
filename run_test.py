"""Run the submission CLI on the included Test set and evaluate its on-disk outputs."""
import argparse
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from statistics import mean
from time import perf_counter

ROOT = Path(__file__).resolve().parent


def evaluate(output, split="test", data_root=ROOT):
    import cv2
    import numpy as np
    from data import Coco
    from post import match

    dataset = Coco(data_root, split)
    payload = json.loads((output / "predictions.json").read_text(encoding="utf-8"))
    by_name = {record["image_name"]: record["objects"] for record in payload}
    if len(payload) != len(dataset.images) or set(by_name) != {i["file_name"] for i in dataset.images}:
        raise ValueError("Prediction image names differ from the Test set")
    rows = []
    confusion = {name: dict.fromkeys(dataset.classes, 0) for name in dataset.classes}
    gt_by_class = dict.fromkeys(dataset.classes, 0)
    pred_by_class = dict.fromkeys(dataset.classes, 0)
    correct = matched = total = 0
    for index, image in enumerate(dataset.images):
        image_gt = dict.fromkeys(dataset.classes, 0)
        image_pred = dict.fromkeys(dataset.classes, 0)
        _, gt = dataset.load(index)
        name = image["file_name"]
        objects = by_name[name]
        path = output / "seg_mask" / (Path(name).stem + ".png")
        labels = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        ids = [obj["id"] for obj in objects]
        if labels is None or labels.dtype != np.uint16 or labels.shape != (image["height"], image["width"]):
            raise ValueError(f"Invalid mask in {path}")
        if ids != list(range(1, len(ids)+1)) or np.unique(labels[labels > 0]).tolist() != ids:
            raise ValueError(f"PNG and JSON IDs disagree for {name}")
        scores = match(labels, gt)
        for _, category, mask in gt:
            if mask.any():
                category_name = dataset.names[category]
                gt_by_class[category_name] += 1
                image_gt[category_name] += 1
        for obj in objects:
            pred_by_class[obj["class"]] += 1
            image_pred[obj["class"]] += 1
        for _, category, predicted_id, iou in scores["pairs"]:
            if iou >= .5:
                matched += 1
                expected = dataset.names[category]
                predicted = objects[predicted_id-1]["class"]
                confusion[expected][predicted] += 1
                correct += int(predicted == expected)
        total += scores["gt"] + scores["predicted"]
        rows.append({
            "filename": name, **{k: v for k, v in scores.items() if k != "pairs"},
            "class_counts": {
                category: {"gt": image_gt[category], "predicted": image_pred[category],
                           "count_error": abs(image_pred[category] - image_gt[category])}
                for category in dataset.classes
            },
        })
    return {
        "images": len(rows), "gt_instances": sum(r["gt"] for r in rows),
        "instance_iou": mean(r["instance_iou"] for r in rows),
        "matched_counting": mean(r["counting"] for r in rows),
        "count_mae": mean(r["count_error"] for r in rows),
        "class_accuracy_at_iou_50": correct / matched if matched else None,
        "class_aware_f1_at_iou_50": 2 * correct / total if total else 1.,
        "confusion_at_iou_50": confusion,
        "per_class": {
            name: {
                "gt": gt_by_class[name],
                "predicted": pred_by_class[name],
                "count_mae": mean(row["class_counts"][name]["count_error"] for row in rows),
                "correct_at_iou_50": confusion[name][name],
                "f1_at_iou_50": 2 * confusion[name][name] / (gt_by_class[name] + pred_by_class[name])
                if gt_by_class[name] + pred_by_class[name] else None,
            }
            for name in dataset.classes
        },
        "rows": rows
    }


def run(weights, output, split="test", threads=4, log=None, confidence=None,
        *, data_root=ROOT, watershed=None, roi_margin=None):
    cmd = [sys.executable, str(ROOT / "infer.py"), "--input", str(data_root / split),
           "--weights", str(weights), "--out", str(output), "--threads", str(threads)]
    if confidence is not None:
        cmd.extend(("--confidence", str(confidence)))
    if watershed is not None:
        cmd.extend(("--watershed", str(watershed)))
    if roi_margin is not None:
        cmd.extend(("--roi-margin", str(roi_margin)))
    env = dict(os.environ, YOLO_OFFLINE="true", YOLO_AUTOINSTALL="false",
               OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads))
    start = perf_counter()
    subprocess.run(cmd, check=True, env=env, stdout=log,
                   stderr=subprocess.STDOUT if log is not None else None)
    elapsed = perf_counter() - start
    report = evaluate(output, split, data_root)
    benchmark = json.loads((output / "benchmark.json").read_text(encoding="utf-8"))
    report.update(platform=platform.platform(), split=split, threads=threads, confidence_override=confidence,
                  median_seconds_per_image=benchmark["median_seconds"], total_seconds=elapsed,
                  end_to_end_seconds_per_image=elapsed / report["images"])
    (output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "outputs_test")
    output = parser.parse_args().out.resolve()
    report = run(ROOT / "weights", output)
    print(f"Images: {report['images']} | GT beans: {report['gt_instances']}")
    for key in ("instance_iou", "matched_counting", "class_accuracy_at_iou_50", "class_aware_f1_at_iou_50"):
        value = report[key]
        print(f"{key}: {value:.2%}" if value is not None else f"{key}: no matched pairs")
    print(f"Count MAE: {report['count_mae']:.2f} beans/image")
    print(f"End-to-end: {report['total_seconds']:.3f} s / {report['images']} = "
          f"{report['end_to_end_seconds_per_image']:.3f} s/image")


if __name__ == "__main__":
    main()
