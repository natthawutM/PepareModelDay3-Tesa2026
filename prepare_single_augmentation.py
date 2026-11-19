"""Prepare Train-only classifier crops and COCO Copy-Paste scenes."""
import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as coco_mask

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "overnight_01" / "src"))
from data import Coco, letterbox

CLASSES = "black broken good husk immature parchment shell".split()
SIZE = 512
CAT = {name: i + 1 for i, name in enumerate(CLASSES)}


def trim(rgb, mask):
    yy, xx = np.nonzero(mask)
    box = np.s_[yy.min():yy.max() + 1, xx.min():xx.max() + 1]
    return rgb[box].copy(), mask[box].copy()


def augment(rgb, mask, rng, long_side=None):
    h, w = mask.shape
    span = int(np.ceil(np.hypot(h, w))) + 4
    offset = ((span - h) // 2, (span - w) // 2)
    canvas = np.full((span, span, 3), 114, np.uint8)
    alpha = np.zeros((span, span), np.uint8)
    y, x = offset
    canvas[y:y+h, x:x+w][mask] = rgb[mask]
    alpha[y:y+h, x:x+w] = mask
    angle = float(rng.uniform(-180, 180))
    transform = cv2.getRotationMatrix2D(((span - 1) / 2, (span - 1) / 2), angle, 1)
    warped = cv2.warpAffine(canvas, transform, (span, span), borderValue=(114, 114, 114))
    alpha = cv2.warpAffine(alpha, transform, (span, span), flags=cv2.INTER_NEAREST).astype(bool)
    warped, alpha = trim(warped, alpha)
    if rng.random() < .5:
        warped, alpha = warped[:, ::-1], alpha[:, ::-1]
    gain, bias = float(rng.uniform(.90, 1.10)), float(rng.uniform(-9, 9))
    warped = np.clip(warped.astype(np.float32) * gain + bias, 0, 255).astype(np.uint8)
    if long_side is not None:
        scale = long_side / max(alpha.shape)
        shape = (max(1, round(alpha.shape[1] * scale)), max(1, round(alpha.shape[0] * scale)))
        warped = cv2.resize(warped, shape, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
        alpha = cv2.resize(alpha.astype(np.uint8), shape, interpolation=cv2.INTER_NEAREST).astype(bool)
    return warped, alpha, angle, gain, bias


def load_donors(root, origin, limits=None):
    ds = Coco(root, "train")
    donors = defaultdict(list)
    colors = []
    indices = range(len(ds.images))
    if limits:
        indices = sorted(indices, key=lambda i: (
            "_sep_" not in ds.images[i]["file_name"].lower(),
            "_pile_" in ds.images[i]["file_name"].lower(),
        ))
    for idx in indices:
        image = ds.images[idx]
        if limits and not any(ds.names[a["category_id"]] in limits for a in ds.by_image[image["id"]]):
            continue
        rgb, instances = ds.load(idx)
        if origin == "new":
            colors.extend(np.median(tile.reshape(-1, 3), axis=0) for tile in
                          (rgb[:32, :32], rgb[-32:, -32:]))
        for aid, category, mask in instances:
            name = ds.names[category]
            if not mask.any() or (limits and (name not in limits or len(donors[name]) >= limits[name])):
                continue
            pix, mask = trim(rgb, mask)
            donors[name].append((origin, image["file_name"], aid, pix, mask))
        if limits and all(len(donors[k]) >= v for k, v in limits.items()):
            break
    return donors, np.asarray(colors)


def write_rgb(path, rgb):
    ok = cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise OSError(f"Unable to write {path}")


def make_crops(donors, root, rng):
    rows = []
    for category, bank in sorted(donors.items()):
        folder = root / "train" / category
        folder.mkdir(parents=True)
        for origin, image, aid, pix, mask in bank:
            for variant in range(4):
                if variant:
                    patch, alpha, angle, gain, bias = augment(pix, mask, rng)
                else:
                    patch, alpha, angle, gain, bias = pix, mask, 0, 1, 0
                crop = letterbox(np.where(alpha[..., None], patch, 114).astype(np.uint8), 96)[0]
                filename = folder / f"bean_{origin}_{aid:04d}_v{variant}.png"
                write_rgb(filename, crop)
                rows.append([filename.relative_to(root).as_posix(), category, origin, image, aid,
                             variant, round(angle, 4), round(gain, 6), round(bias, 6)])
    with (root / "manifest.csv").open("w", encoding="utf8", newline="") as file:
        out = csv.writer(file)
        out.writerow(["filename", "class", "source", "source_image", "source_id",
                      "variant", "rotation_degrees", "brightness_gain", "brightness_bias"])
        out.writerows(rows)
    return len(rows)


def make_background(colors, rng):
    color = colors[rng.integers(len(colors))].clip(130, 245).astype(np.float32)
    coordinates = np.linspace(-1, 1, SIZE, dtype=np.float32)
    gradient = coordinates[:, None] * rng.uniform(-13, 13) + coordinates[None, :] * rng.uniform(-13, 13)
    noise = rng.normal(0, .85, (SIZE, SIZE))
    return np.clip(color + (gradient + noise)[..., None], 0, 255).astype(np.uint8)


def paste(canvas, layers, occupied, donor, category, rng):
    origin, source, source_id, pix, mask = donor
    pix, mask, _, _, _ = augment(pix, mask, rng, int(rng.integers(44, 101)))
    h, w = mask.shape
    area = int(mask.sum())
    for trial in range(125):
        nearby = bool(layers) and trial < 105
        if nearby:
            anchor = layers[rng.integers(len(layers))]["center"]
            angle = rng.uniform(0, 2 * np.pi)
            radius = rng.uniform(29, 78)
            cx, cy = anchor + radius * np.array([np.cos(angle), np.sin(angle)])
        else:
            cx, cy = rng.uniform(76, SIZE - 76, 2)
        x, y = round(cx - w / 2), round(cy - h / 2)
        if x < 12 or y < 12 or x + w > SIZE - 12 or y + h > SIZE - 12:
            continue
        box = np.s_[y:y+h, x:x+w]
        overlap = int(np.count_nonzero(occupied[box] & mask))
        if nearby and not .015 * area <= overlap <= .22 * area:
            continue
        if not nearby and overlap:
            continue
        if any(np.count_nonzero(layer["mask"][box] & mask) >
               layer["mask"].sum() - .70 * layer["original_area"] for layer in layers):
            continue
        instance = np.zeros_like(occupied)
        instance[box] = mask
        for layer in layers:
            layer["mask"] &= ~instance
        canvas[box][mask] = pix[mask]
        occupied |= instance
        layers.append({"mask": instance, "class": category, "center": np.array([cx, cy]),
                       "source": origin, "file": source, "source_id": source_id,
                       "original_area": area, "overlap_pixels": overlap})
        return


def create_scenes(donors, colors, root, rng, count):
    folder = root / "train"
    folder.mkdir(parents=True)
    preview = root.parent / "previews"
    preview.mkdir()
    images, annotations = [], []
    totals = Counter()
    while len(images) < count:
        canvas = make_background(colors, rng)
        layers = []
        occupied = np.zeros((SIZE, SIZE), bool)
        order = rng.permutation(CLASSES)
        for category in order:
            bank = donors[category]
            paste(canvas, layers, occupied, bank[rng.integers(len(bank))], category, rng)
        if len(layers) != len(CLASSES) or not any(layer["overlap_pixels"] for layer in layers):
            continue
        img_id = len(images) + 1
        name = f"copy_paste_{img_id:04d}.png"
        write_rgb(folder / name, canvas)
        images.append({"id": img_id, "file_name": name, "width": SIZE, "height": SIZE})
        shown = canvas.copy()
        combined = np.zeros_like(occupied)
        for label, layer in enumerate(layers, 1):
            mask = layer["mask"]
            assert mask.any() and not np.any(combined & mask)
            combined |= mask
            rle = coco_mask.encode(np.asfortranarray(mask.astype(np.uint8)))
            assert np.array_equal(coco_mask.decode(rle).astype(bool), mask)
            rle["counts"] = rle["counts"].decode("ascii")
            annotations.append({"id": len(annotations) + 1, "image_id": img_id,
                                "category_id": CAT[layer["class"]], "segmentation": rle,
                                "area": int(mask.sum()), "bbox": coco_mask.toBbox(rle).tolist(),
                                "iscrowd": 0, "instance_id": label,
                                "donor_source": layer["source"], "donor_image": layer["file"],
                                "donor_annotation_id": layer["source_id"]})
            totals[layer["class"]] += 1
            if img_id <= 12:
                contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(shown, contours, -1, (30, 230, 70), 1)
                yy, xx = np.nonzero(mask)
                cv2.putText(shown, f"{label} {layer['class']}", (int(xx.mean()), int(yy.mean())),
                            cv2.FONT_HERSHEY_SIMPLEX, .39, (25, 25, 25), 1, cv2.LINE_AA)
        assert np.array_equal(occupied, combined)
        if img_id <= 12:
            write_rgb(preview / name, np.hstack((canvas, shown)))
    (folder / "_annotations.coco.json").write_text(json.dumps({
        "images": images, "annotations": annotations,
        "categories": [{"id": v, "name": k, "supercategory": "bean"} for k, v in CAT.items()],
    }), encoding="utf8")
    return dict(totals)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=BASE / "d3_single_aug_v2")
    parser.add_argument("--scenes", type=int, default=160)
    args = parser.parse_args()
    if args.scenes < 1:
        parser.error("--scenes must be positive")
    args.out.mkdir(parents=True, exist_ok=False)
    cv2.setNumThreads(1)
    rng = np.random.default_rng(41)
    donors, colors = load_donors(BASE / "d3_single_ready_v1", "new")
    previous, _ = load_donors(BASE / "coffee_ready_reviewed_v1" / "coco", "old",
                              limits={"black": 40, "good": 40})
    for category, bank in previous.items():
        donors[category].extend(bank)
    assert all(donors[category] for category in CLASSES)
    crops = make_crops(donors, args.out / "classifier", rng)
    totals = create_scenes(donors, colors, args.out / "coco", rng, args.scenes)
    summary = {"seed": 41, "train_donors": {k: len(v) for k, v in donors.items()},
               "crops": crops, "synthetic_images": args.scenes, "synthetic_instances": totals,
               "resolution": [SIZE, SIZE], "source_splits": ["d3_single_ready_v1/train",
                                                                "coffee_ready_reviewed_v1/coco/train"]}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf8")
    print(summary, flush=True)


if __name__ == "__main__":
    main()
