"""Benchmark all packaged models on the same COCO split."""
import argparse
import csv
import json
import subprocess
from pathlib import Path

from run_test import ROOT, run

METRICS = ("images", "gt_instances", "instance_iou", "matched_counting", "count_mae",
           "class_accuracy_at_iou_50", "class_aware_f1_at_iou_50", "total_seconds",
           "end_to_end_seconds_per_image", "platform", "threads")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("valid", "test"), default="valid")
    parser.add_argument("--model", nargs="+")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    models = {p.parent.name: p.parent for p in sorted((ROOT / "models").glob("*/manifest.json"))}
    if args.list:
        print("\n".join(models))
        return
    names = list(dict.fromkeys(args.model or models))
    if not names or any(name not in models for name in names) or args.threads < 1:
        parser.error("Use positive threads and valid model names from --list")
    out = (args.out or ROOT / f"outputs_all_{args.split}").resolve()
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, name in enumerate(names, 1):
        weights = models[name]
        target = out / name
        target.mkdir(exist_ok=True)
        row = {"model": name, "split": args.split}
        print(f"[{i}/{len(names)}] {name}", flush=True)
        try:
            cfg = json.loads((weights / "manifest.json").read_text(encoding="utf-8"))
            paths = [weights / cfg["variants"]["selected"]["file"]]
            if "classifier" in cfg:
                paths.append(weights / cfg["classifier"])
            row["model_bytes"] = sum(f.stat().st_size for path in paths
                                     for f in (path.rglob("*") if path.is_dir() else [path]) if f.is_file())
            with (target / "infer.log").open("w", encoding="utf-8") as log:
                report = run(weights, target, args.split, args.threads, log)
            row.update({key: report[key] for key in METRICS}, status="ok")
            print(f"  IoU {row['instance_iou']:.2%} | Count {row['matched_counting']:.2%} | "
                  f"F1 {row['class_aware_f1_at_iou_50']:.2%} | {row['end_to_end_seconds_per_image']:.3f} s/image", flush=True)
        except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
            row.update(status="error", error=str(error))
            print(f"  Failed: {error}. See {target / 'infer.log'}", flush=True)
        rows.append(row)
        tmp = out / "comparison.tmp"
        with tmp.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=("model", "split", "status", "model_bytes", *METRICS, "error"))
            writer.writeheader()
            writer.writerows(rows)
        tmp.replace(out / "comparison.csv")
    print(out / "comparison.csv")
    if any(row["status"] == "error" for row in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
