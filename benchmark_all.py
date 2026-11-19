"""Benchmark all packaged models on the same COCO split."""
import argparse
import csv
import json
import subprocess
from pathlib import Path

from run_test import ROOT, run

METRICS = ("images", "gt_instances", "instance_iou", "matched_counting", "count_mae",
           "class_accuracy_at_iou_50", "class_aware_f1_at_iou_50", "total_seconds",
           "end_to_end_seconds_per_image", "median_seconds_per_image", "platform", "threads")

DAY3_THRESHOLDS = {
    "yolo_640": (0.30, 0.35),
    "yolo_640_ncnn": (0.30, 0.35),
    "yolo_640_int8": (0.15, 0.175, 0.20, 0.35),
}
DAY3_WATERSHED = (
    ("baseline", {}),
    ("foreground_045", {"foreground": 0.45}),
    ("foreground_055", {"foreground": 0.55}),
    ("boundary_045", {"boundary": 0.45}),
    ("boundary_055", {"boundary": 0.55}),
    ("marker_8", {"min_marker": 8}),
    ("marker_24", {"min_marker": 24}),
    ("area_8", {"min_area": 8}),
    ("area_20", {"min_area": 20}),
)


def day3_trials():
    for label, settings in DAY3_WATERSHED:
        yield "A0", label, None, settings
    for name, thresholds in DAY3_THRESHOLDS.items():
        for confidence in thresholds:
            yield name, f"conf_{confidence}", confidence, None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("valid", "test"), default="valid")
    parser.add_argument("--model", nargs="+")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--data-root", type=Path, default=ROOT,
                        help="COCO dataset directory containing valid/ and optionally test/")
    parser.add_argument("--day3-tune", action="store_true", help="Sweep the four Day 3 finalists on Valid")
    parser.add_argument("--confidence", nargs="+", type=float, help="Evaluate each of these YOLO thresholds")
    parser.add_argument("--watershed", type=Path, help="Reuse a watershed.json with one U-Net model")
    args = parser.parse_args()
    models = {p.parent.name: p.parent for p in sorted((ROOT / "models").glob("*/manifest.json"))}
    if args.list:
        print("\n".join(models))
        return
    if args.day3_tune and (args.split != "valid" or args.model or args.confidence is not None or args.watershed):
        parser.error("--day3-tune runs its four fixed candidates on Valid")
    data_root = args.data_root.resolve()
    if not (data_root / args.split / "_annotations.coco.json").is_file():
        parser.error(f"Missing COCO annotations in {data_root / args.split}")
    names = list(dict.fromkeys(args.model or models))
    if not names or any(name not in models for name in names) or args.threads < 1:
        parser.error("Use positive threads and valid model names from --list")
    if args.confidence is not None:
        if not args.model or any(not 0 <= value <= 1 for value in args.confidence):
            parser.error("Select YOLO models with --model and use confidence values between 0 and 1")
        if any(json.loads((models[n] / "manifest.json").read_text(encoding="utf-8"))["family"] != "yolo"
               for n in names):
            parser.error("--confidence applies only to YOLO models")
    if args.watershed:
        if not args.model or len(names) != 1 or args.confidence is not None or not args.watershed.is_file():
            parser.error("--watershed requires a JSON file and one U-Net model")
        if json.loads((models[names[0]] / "manifest.json").read_text(encoding="utf-8"))["family"] != "unet":
            parser.error("--watershed applies only to U-Net models")
    thresholds = list(dict.fromkeys(args.confidence)) if args.confidence is not None else [None]
    trials = (list(day3_trials()) if args.day3_tune else
              [(name, f"conf_{c}" if c is not None else "", c, None)
               for name in names for c in thresholds])
    if any(name not in models for name, *_ in trials):
        parser.error("Missing model folders required for this benchmark")
    default_out = "outputs_day3_tune" if args.day3_tune else f"outputs_all_{args.split}"
    out = (args.out or ROOT / default_out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, (name, label, confidence, settings) in enumerate(trials, 1):
        weights = models[name]
        target = out / name / label
        target.mkdir(parents=True, exist_ok=True)
        row = {"model": name, "trial": label, "split": args.split, "confidence_override": confidence}
        print(f"[{i}/{len(trials)}] {name} {label}".rstrip(), flush=True)
        try:
            cfg = json.loads((weights / "manifest.json").read_text(encoding="utf-8"))
            paths = [weights / cfg["variants"]["selected"]["file"]]
            if "classifier" in cfg:
                paths.append(weights / cfg["classifier"])
            row["model_bytes"] = sum(f.stat().st_size for path in paths
                                     for f in (path.rglob("*") if path.is_dir() else [path]) if f.is_file())
            watershed = args.watershed.resolve() if args.watershed else None
            if settings is not None:
                watershed = target / "watershed.json"
                watershed.write_text(json.dumps(settings, indent=2), encoding="utf-8")
            with (target / "infer.log").open("w", encoding="utf-8") as log:
                report = run(weights, target, args.split, args.threads, log, confidence,
                             data_root=data_root, watershed=watershed)
            row.update({key: report[key] for key in METRICS}, status="ok")
            row["day3_performance_out_of_30"] = 15 * (report["instance_iou"] + report["matched_counting"])
            row["under_1s_both"] = max(report["median_seconds_per_image"],
                                       report["end_to_end_seconds_per_image"]) <= 1
            print(f"  IoU {row['instance_iou']:.2%} | Count {row['matched_counting']:.2%} | "
                  f"F1 {row['class_aware_f1_at_iou_50']:.2%} | {row['end_to_end_seconds_per_image']:.3f} s/image", flush=True)
        except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
            row.update(status="error", error=str(error))
            print(f"  Failed: {error}. See {target / 'infer.log'}", flush=True)
        rows.append(row)
        tmp = out / "comparison.tmp"
        with tmp.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=("model", "trial", "split", "status", "model_bytes",
                                                       *METRICS, "day3_performance_out_of_30", "under_1s_both",
                                                       "confidence_override", "error"))
            writer.writeheader()
            writer.writerows(rows)
        tmp.replace(out / "comparison.csv")
    print(out / "comparison.csv")
    if args.day3_tune:
        ranking = sorted((r for r in rows if r["status"] == "ok"),
                         key=lambda r: (not r["under_1s_both"], -r["day3_performance_out_of_30"],
                                        r["end_to_end_seconds_per_image"]))
        (out / "ranking.json").write_text(json.dumps(ranking, indent=2), encoding="utf-8")
        for row in ranking[:3]:
            print(f"{row['model']} {row['trial']} | {row['day3_performance_out_of_30']:.2f}/30"
                  f" | <=1s (both): {row['under_1s_both']}")
    if any(row["status"] == "error" for row in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
