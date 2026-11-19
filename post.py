import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from skimage.segmentation import watershed

DEFAULTS = {'foreground': .5, 'boundary': .5, 'min_area': 12, 'min_marker': 8}


def instances_from_logits(logits, box, shape, settings, debug=False):
    y, x, h, w = box
    fg_prob, boundary = 1 / (1 + np.exp(-np.clip(logits[:, y:y+h, x:x+w], -40, 40)))
    fg = (fg_prob >= settings['foreground']).astype(np.uint8)
    count, components, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    allowed = stats[:, cv2.CC_STAT_AREA] >= settings['min_area']
    allowed[0] = False
    # Optional rejection of broad, shallow foreground strips at image borders
    # (e.g. exposed tabletop beyond the paper). Never infer an expected count.
    strip_span = settings.get('border_strip_span', 0)
    if strip_span:
        strip_depth = settings.get('border_strip_depth', .2)
        fh, fw = fg.shape
        for component in range(1, count):
            left, top, width, height, _ = stats[component]
            horizontal = (top == 0 or top + height == fh) and width >= strip_span*fw and height <= strip_depth*fh
            vertical = (left == 0 or left + width == fw) and height >= strip_span*fh and width <= strip_depth*fw
            if horizontal or vertical:
                allowed[component] = False
    fg = allowed[components]
    dist = cv2.distanceTransform(np.pad(fg.astype(np.uint8), 1), cv2.DIST_L2, 5)[1:-1, 1:-1]
    core = (fg & (boundary < settings['boundary'])).astype(np.uint8)
    _, markers, stats, _ = cv2.connectedComponentsWithStats(core, connectivity=8)
    keep = stats[:, cv2.CC_STAT_AREA] >= settings['min_marker']
    keep[0] = False
    markers[~keep[markers]] = 0
    separation = settings.get('marker_separation', 0)
    if separation:
        # Retain the larger seed when two seed centroids in the same object are close.
        retained = {}
        for marker in sorted(np.flatnonzero(keep), key=lambda i: (-stats[i, cv2.CC_STAT_AREA], i)):
            yy, xx = np.where(markers == marker)
            region = int(components[yy[0], xx[0]])
            center = np.array([yy.mean(), xx.mean()])
            if any(np.linalg.norm(center - prior) < separation for prior in retained.get(region, [])):
                markers[yy, xx] = 0
            else:
                retained.setdefault(region, []).append(center)
    next_id = int(markers.max()) + 1
    seeded = set(np.unique(components[markers > 0]))
    for label in range(1, count):
        if allowed[label] and label not in seeded:
            peak = np.argmax(np.where(components == label, dist, -1))
            markers.flat[peak] = next_id
            next_id += 1
    elevation = boundary - dist / max(float(dist.max()), 1)
    labels = watershed(elevation, markers, mask=fg, connectivity=2)
    final_min_area = settings.get('final_min_area', 1)
    if final_min_area > 1:
        # Merge tiny watershed regions into adjacent seeds by reflooding the same foreground.
        sizes = np.bincount(labels.ravel(), minlength=int(markers.max()) + 1)
        filtered = markers.copy()
        filtered[sizes[markers] < final_min_area] = 0
        filtered_fg = fg & np.isin(components, np.unique(components[filtered > 0]))
        labels = watershed(elevation, filtered, mask=filtered_fg, connectivity=2)
        markers = filtered
    # Compact IDs before upscaling: scanning a 12 MP restored mask is expensive
    # on the Pi. Nearest-neighbour upscaling preserves every source label.
    upscaling = shape[0] >= labels.shape[0] and shape[1] >= labels.shape[1]
    source = labels if upscaling else cv2.resize(
        labels.astype(np.float32), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST).astype(np.int32)
    present = np.unique(source[source > 0])
    if len(present) > np.iinfo(np.uint16).max:
        raise ValueError('Too many instances for a 16-bit ID mask')
    lookup = np.zeros(int(source.max()) + 1, np.uint16)
    lookup[present] = np.arange(1, len(present) + 1, dtype=np.uint16)
    output = lookup[source]
    if upscaling:
        output = cv2.resize(output, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    if debug:
        return output, {'foreground_probability': fg_prob, 'foreground': fg,
                        'boundary': boundary, 'markers': markers,
                        'foreground_pixels': int(fg.sum()), 'assigned_pixels': int((labels > 0).sum())}
    return output


def match(labels, instances):
    gt = [(i, c, m) for i, c, m in instances if m.any()]
    n = int(labels.max())
    areas = np.bincount(labels.ravel(), minlength=n+1)[1:]
    matrix = np.zeros((len(gt), n))
    for row, (_, _, mask) in enumerate(gt):
        overlap = np.bincount(labels[mask], minlength=n+1)[1:]
        matrix[row] = overlap / (areas + mask.sum() - overlap)
    rows, cols = linear_sum_assignment(matrix, maximize=True)
    pairs = [(gt[r][0], gt[r][1], int(c+1), float(matrix[r, c]))
             for r, c in zip(rows, cols) if matrix[r, c] > 0]
    return {'instance_iou': sum(p[3] for p in pairs) / max(n, len(gt)) if n or gt else 1.,
            'counting': len(pairs) / max(n, len(gt)) if n or gt else 1.,
            'count_error': abs(n - len(gt)), 'predicted': n, 'gt': len(gt),
            'pairs': pairs}

