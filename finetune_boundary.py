"""Fine-tune A0 with paired boundary-loss weights on the existing training split."""
import argparse
import json
import math
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TRAIN_SOURCE = ROOT.parent / "overnight_01" / "src"
if str(TRAIN_SOURCE) not in sys.path:
    sys.path.insert(0, str(TRAIN_SOURCE))

import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data import Coco
from model import Segmenter, loss_fn
from train import TrainingSet, evaluate, save_weights


def weighted_boundary_loss(logits, target, valid, boundary_weight):
    if boundary_weight == 1.0:
        return loss_fn(logits, target, valid)
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    probs = logits.float().sigmoid() * valid
    masked_target = target * valid
    axes = (0, 2, 3)
    intersection = (probs * masked_target).sum(axes)
    dice = (2 * intersection + 1) / (probs.sum(axes) + masked_target.sum(axes) + 1)
    losses = (bce * valid).sum(axes) / valid.sum() + 1 - dice
    return (losses[0] + boundary_weight * losses[1]) / (1 + boundary_weight)


def fit(state, config, dataset, valid, device, destination, weight, epochs):
    destination.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(41)
    model = Segmenter(widths=config["widths"]).to(device)
    model.load_state_dict(state["model"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])
    best_score = -1.
    log_path = destination / "training.jsonl"
    for epoch in range(epochs):
        model.train()
        dataset.epoch = state["epoch"] + 1 + epoch
        batches = DataLoader(dataset, batch_size=4, shuffle=True,
                             generator=torch.Generator().manual_seed(41 + epoch))
        step_losses = []
        for x, y, valid_pixels in batches:
            x, y, valid_pixels = x.to(device), y.to(device), valid_pixels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = weighted_boundary_loss(model(x), y, valid_pixels, weight)
            value = loss.item()
            if not math.isfinite(value):
                raise ValueError("Non-finite loss")
            loss.backward()
            optimizer.step()
            step_losses.append(value)
        model.eval()
        with torch.inference_mode():
            metrics = evaluate(valid, config, lambda x: model(torch.from_numpy(x).to(device)).cpu().numpy())
        score = 15 * (metrics["instance_iou"] + metrics["counting"])
        row = {"epoch": epoch + 1, "steps": len(step_losses), "loss": sum(step_losses) / len(step_losses),
               "boundary_loss_weight": weight, "day3_performance_out_of_30": score,
               **{k: v for k, v in metrics.items() if k != "images"}}
        print(json.dumps(row), flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        checkpoint = {"model": model.state_dict(), "config": config, "epoch": epoch,
                      "boundary_loss_weight": weight, "optimizer": optimizer.state_dict()}
        if score > best_score:
            best_score = score
            save_weights(destination / "best.pt", checkpoint)
            (destination / "valid.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        save_weights(destination / "last.pt", checkpoint)
    best = torch.load(destination / "best.pt", map_location="cpu")
    model.cpu().load_state_dict(best["model"])
    return model.eval()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT.parent / "coffee_ready" / "coco")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints" / "A0" / "best.pt")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--copy-paste-data", action="store_true",
                        help="Train one A0 variant on a prepared mixture with Copy-Paste scenes")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.epochs < 1 or args.out.exists():
        parser.error("Use positive epochs and a new output directory")
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(args.checkpoint, map_location="cpu")
    config = dict(state["config"], lr=0.0001, epochs=args.epochs)
    manifest = json.loads((ROOT / "models/A0/manifest.json").read_text(encoding="utf-8"))
    manifest["watershed"].update(boundary=0.425, min_marker=20)
    config["watershed"] = manifest["watershed"]
    config["copy_paste"] = 0.0
    dataset = TrainingSet(Coco(args.data_root, "train"), config, None)
    valid = Coco(args.data_root, "valid")
    print(f"Device={device}; train={len(dataset)}; valid={len(valid.images)}", flush=True)
    trials = (("A0_copy_paste", 1.0),) if args.copy_paste_data else (
        ("A0_boundary_control", 1.0), ("A0_boundary_15", 1.5),
    )
    for label, weight in trials:
        run_dir = args.out / label
        print(f"Training {label}", flush=True)
        model = fit(state, config, dataset, valid, device, run_dir, weight, args.epochs)
        runtime = run_dir / "runtime"
        runtime.mkdir()
        torch.onnx.export(model, (torch.zeros(1, 3, config["size"], config["size"]),),
                          runtime / "model.onnx", dynamo=False, opset_version=17,
                          input_names=["image"], output_names=["logits"])
        shutil.copy2(ROOT / "models/A0/classifier.onnx", runtime / "classifier.onnx")
        (runtime / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
