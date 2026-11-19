"""Review and improve the A0 classifier using matched Train crops."""
import argparse
import csv
import hashlib
import html
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from data import Coco, normalize, read_rgb
from infer import Predictor
from post import match
from run_test import ROOT

TRAIN_SOURCE = ROOT.parent / "overnight_01" / "src"
sys.path.append(str(TRAIN_SOURCE))
from model import Classifier  # Uses the original training architecture.


def crops(rgb, labels, boxes, predictor):
    return [predictor.prep_crop(rgb[box], labels[box] == object_id) for object_id, box, _ in boxes]


def valid_examples(source, outputs, predictor, review):
    from scipy.ndimage import find_objects
    payload = json.loads((outputs / "predictions.json").read_text(encoding="utf-8"))
    by_name = {r["image_name"]: r["objects"] for r in payload}
    records, samples, cards = [], [], []
    review.mkdir()
    for i, image in enumerate(source.images):
        rgb, gt = source.load(i)
        name = image["file_name"]
        ids = cv2.imread(str(outputs / "seg_mask" / (Path(name).stem + ".png")), cv2.IMREAD_UNCHANGED)
        boxes = [(j, box, None) for j, box in enumerate(find_objects(ids), 1)]
        samples.extend(crops(rgb, ids, boxes, predictor))
        objects = by_name[name]
        matched = {pid: (category, iou) for _, category, pid, iou in match(ids, gt)["pairs"]}
        for pid, box, _ in boxes:
            obj = objects[pid - 1]
            category, iou = matched.get(pid, (None, 0.))
            old = obj["class"]
            expected = source.names[category] if category is not None else None
            records.append((name, pid, expected, old, iou))
            if iou >= .75 and expected != old:
                patch = rgb[box].copy()
                patch[ids[box] != pid] = 114
                thumbnail = cv2.resize(patch, (125, 125))
                rel = f"image_{i + 1}_id_{pid}.jpg"
                cv2.imwrite(str(review / rel), cv2.cvtColor(thumbnail, cv2.COLOR_RGB2BGR))
                caption = f"{expected} -> {old} | IoU {iou:.2f} | ID {pid}"
                cards.append((expected, old, f'<figure><img loading="lazy" src="{rel}"><figcaption>'
                              f'{html.escape(caption)}<br>{html.escape(name)}</figcaption></figure>'))
    body = "".join(card for _, _, card in sorted(cards))
    (review / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>A0 class errors</title><h1>Valid | Good mask, wrong class</h1>'
        '<p>Ground truth → A0 label. IoU ≥ 0.75. Crop from predicted mask.</p>'
        '<style>body{font:16px system-ui}main{display:flex;flex-wrap:wrap}figure{margin:12px;width:200px;overflow-wrap:anywhere}</style>'
        f'<main>{body}</main>', encoding="utf-8")
    return records, np.stack(samples)


def train_examples(source, predictor):
    names = predictor.classes
    patches, labels = [], []
    sources = Counter()
    for i in range(len(source.images)):
        rgb, gt = source.load(i)
        for _, category, mask in gt:
            ys, xs = np.nonzero(mask)
            if not len(xs):
                continue
            box = np.s_[ys.min():ys.max()+1, xs.min():xs.max()+1]
            patches.append(predictor.prep_crop(rgb[box], mask[box]))
            labels.append(names.index(source.names[category]))
            sources["gt"] += 1
        ids, _, boxes = predictor.predict_instances(rgb)
        accepted = {pid: category for _, category, pid, iou in match(ids, gt)["pairs"] if iou >= .75}
        for pid, box, mask in boxes:
            if pid in accepted:
                patches.append(predictor.prep_crop(rgb[box], mask))
                labels.append(names.index(source.names[accepted[pid]]))
                sources["matched_prediction"] += 1
    return np.stack(patches), np.array(labels), dict(sources)


def score(records, labels, truth_objects, classes):
    correct = matches = 0
    class_hits = Counter()
    class_pred = Counter(labels)
    for (_, _, expected, _, iou), label in zip(records, labels, strict=True):
        if iou >= .5:
            matches += 1
            if expected == label:
                correct += 1
                class_hits[label] += 1
    f1 = {name: 2 * class_hits[name] / max(1, truth_objects[name] + class_pred[name]) for name in classes}
    return {"matched_accuracy": correct / matches, "class_aware_f1": 2 * correct / (sum(truth_objects.values()) + len(labels)),
            "macro_f1": sum(f1.values()) / len(f1), "f1_by_class": f1}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--data-root", type=Path, default=ROOT.parent / "coffee_ready_reviewed_v1" / "coco")
    p.add_argument("--valid-root", type=Path, default=ROOT, help="COCO root with an explicit valid split")
    p.add_argument("--adapt-root", type=Path, help="Additional COCO train split, sampled at 25%%")
    p.add_argument("--crop-root", type=Path, help="Prepared classifier directory with train crops and manifest.csv")
    p.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/A0_cls/classifier.pt")
    p.add_argument("--baseline-run", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=12)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(41)
    cv2.setNumThreads(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest_path = ROOT / "models/A0/manifest.json"
    predictor = Predictor(manifest_path.parent, "selected", 4, watershed={"boundary": 0.425, "min_marker": 20})
    classes = predictor.classes
    train = Coco(args.data_root, "train")
    valid = Coco(args.valid_root, "valid")
    adaptation = Coco(args.adapt_root, "train") if args.adapt_root else None
    train_ids = {hashlib.sha256((train.folder / x["file_name"]).read_bytes()).digest() for x in train.images}
    valid_ids = {hashlib.sha256((valid.folder / x["file_name"]).read_bytes()).digest() for x in valid.images}
    if adaptation is not None:
        train_ids.update(hashlib.sha256((adaptation.folder / x["file_name"]).read_bytes()).digest()
                         for x in adaptation.images)
    if train_ids & valid_ids:
        raise ValueError("Train and Valid share image contents")
    records, val = valid_examples(valid, args.baseline_run, predictor, args.out / "review")
    truth = Counter()
    for annotations in valid.by_image.values():
        truth.update(valid.names[a["category_id"]] for a in annotations)
    base = score(records, [r[3] for r in records], truth, classes)
    print("Baseline", base, flush=True)
    with (args.out / "review/class_errors.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(("image", "predicted_id", "gt_class", "predicted_class", "iou"))
        w.writerows(r for r in records if r[4] >= .75 and r[2] != r[3])
    x, y, sources = train_examples(train, predictor)
    print("Training crops", sources, flush=True)
    sample_groups = [(x, y, 1.0)]
    if adaptation is not None:
        adapt_x, adapt_y, adapt_sources = train_examples(adaptation, predictor)
        sample_groups = [(x, y, .75), (adapt_x, adapt_y, .25)]
        sources = {"original": sources, "day3": adapt_sources, "day3_sampling_fraction": .25}
        print("Training with Day 3 crops", adapt_sources, flush=True)
    if args.crop_root is not None:
        with (args.crop_root / "manifest.csv").open(encoding="utf-8", newline="") as handle:
            crop_rows = list(csv.DictReader(handle))
        crop_x = np.stack([normalize(read_rgb(args.crop_root / row["filename"])) for row in crop_rows])
        crop_y = np.array([classes.index(row["class"]) for row in crop_rows])
        sample_groups = [(patches, labels, share * .75) for patches, labels, share in sample_groups]
        sample_groups.append((crop_x, crop_y, .25))
        sources["prepared_crops"] = {"count": len(crop_y), "sampling_fraction": .25,
                                     "root": str(args.crop_root)}
        print(f"Prepared Train crops: {len(crop_y)}", flush=True)
    sampler = None
    if len(sample_groups) > 1:
        sampling_weights = np.concatenate([np.full(len(labels), share / len(labels))
                                           for _, labels, share in sample_groups])
        sampler = WeightedRandomSampler(sampling_weights, len(sampling_weights), replacement=True,
                                        generator=torch.Generator().manual_seed(41))
        x = np.concatenate([patches for patches, _, _ in sample_groups])
        y = np.concatenate([labels for _, labels, _ in sample_groups])
        del sample_groups
    state = torch.load(args.checkpoint, map_location="cpu")
    assert state["classes"] == classes
    model = Classifier().to(device)
    model.load_state_dict(state["model"])
    weights = torch.tensor((len(y) / (len(classes) * np.bincount(y, minlength=len(classes)))) ** .5,
                           dtype=torch.float32, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=0.00035)
    train_data = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    val_x = torch.from_numpy(val)
    history = []
    best_score = -1.
    for epoch in range(args.epochs):
        model.train()
        for batch, target in DataLoader(train_data, batch_size=48, shuffle=sampler is None, sampler=sampler):
            batch, target = batch.to(device), target.to(device)
            if torch.rand(()).item() < .5:
                batch = batch.flip((-1,))
            if torch.rand(()).item() < .5:
                batch = batch.flip((-2,))
            # Mild exposure variation on normalized RGB, preserving the color signal.
            batch = batch + torch.empty((len(batch), 1, 1, 1), device=device).uniform_(-.14, .14)
            opt.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(model(batch), target, weight=weights)
            loss.backward()
            opt.step()
        model.eval()
        with torch.inference_mode():
            pred = torch.cat([model(batch.to(device)).argmax(1).cpu() for batch in val_x.split(64)]).tolist()
        scores = score(records, [classes[i] for i in pred], truth, classes)
        history.append({"epoch": epoch + 1, **scores})
        print(epoch + 1, scores["class_aware_f1"], scores["macro_f1"], flush=True)
        if scores["class_aware_f1"] > best_score:
            best_score = scores["class_aware_f1"]
            torch.save({"model": model.state_dict(), "classes": classes, "size": 96, "epoch": epoch+1},
                       args.out / "classifier.pt")
    runtime = args.out / "runtime"
    shutil.copytree(manifest_path.parent, runtime)
    manifest = json.loads(manifest_path.read_text(encoding="utf8"))
    manifest["watershed"].update(boundary=0.425, min_marker=20)
    (runtime / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf8")
    model.cpu().load_state_dict(torch.load(args.out / "classifier.pt", map_location="cpu")["model"])
    torch.onnx.export(model.eval(), (torch.zeros(1, 3, 96, 96),), runtime / "classifier.onnx",
                      input_names=["image"], output_names=["logits"],
                      dynamic_axes={"image": {0: "instances"}, "logits": {0: "instances"}},
                      opset_version=17, dynamo=False)
    (args.out / "training_summary.json").write_text(json.dumps(
        {"train_crops": sources, "baseline": base, "epochs": history}, indent=2), encoding="utf8")


if __name__ == "__main__":
    main()
