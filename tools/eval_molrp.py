"""
Compute moLRP and its components from COCO-format GT and detection JSON files.

GT format  (gt_refined.json / gt_ori.json):
    {"annotations": [{"image_id": int, "category_id": int, "bbox": [x,y,w,h]}, ...], ...}

Det format (refined_pred.json / ori_pred.json):
    [{"image_id": int, "category_id": int, "bbox": [x,y,w,h], "score": float}, ...]

Usage:
    python tools/eval_molrp.py --gt gt_refined.json --det refined_pred.json --num_classes 30
"""

import json
import argparse
import numpy as np
from collections import defaultdict
from loguru import logger

VID_CLASSES = [
    'airplane', 'antelope', 'bear', 'bicycle', 'bird',
    'bus', 'car', 'cattle', 'dog', 'domestic_cat',
    'elephant', 'fox', 'giant_panda', 'hamster', 'horse',
    'lion', 'lizard', 'monkey', 'motorcycle', 'rabbit',
    'red_panda', 'sheep', 'snake', 'squirrel', 'tiger',
    'train', 'turtle', 'watercraft', 'whale', 'zebra',
]


def xywh_to_xyxy(boxes):
    """Convert [x, y, w, h] → [xmin, ymin, xmax, ymax]. boxes: (N,4) ndarray."""
    out = boxes.copy()
    out[:, 2] = boxes[:, 0] + boxes[:, 2]
    out[:, 3] = boxes[:, 1] + boxes[:, 3]
    return out


def calculate_iou(boxes1, boxes2):
    """
    IoU between two sets of boxes in [xmin, ymin, xmax, ymax] format.
    Returns (N, M) matrix.
    """
    if len(boxes1) == 0 or len(boxes2) == 0:
        return np.zeros((len(boxes1), len(boxes2)))

    b1 = np.array(boxes1, dtype=np.float32)  # (N, 4)
    b2 = np.array(boxes2, dtype=np.float32)  # (M, 4)

    ix_min = np.maximum(b1[:, 0:1], b2[:, 0])
    iy_min = np.maximum(b1[:, 1:2], b2[:, 1])
    ix_max = np.minimum(b1[:, 2:3], b2[:, 2])
    iy_max = np.minimum(b1[:, 3:4], b2[:, 3])

    iw = np.maximum(0.0, ix_max - ix_min)
    ih = np.maximum(0.0, iy_max - iy_min)
    intersection = iw * ih  # (N, M)

    area1 = (b1[:, 2] - b1[:, 0]) * (b1[:, 3] - b1[:, 1])  # (N,)
    area2 = (b2[:, 2] - b2[:, 0]) * (b2[:, 3] - b2[:, 1])  # (M,)
    union = area1[:, None] + area2[None, :] - intersection

    return intersection / (union + 1e-6)


def evaluate_lrp_for_class(gt_boxes_all, det_boxes_all, det_scores_all, tau=0.5):
    """
    Compute oLRP and components for one class across images.
    gt_boxes_all / det_boxes_all : list of (N,4) ndarrays in xyxy format, one per image.
    det_scores_all               : list of (N,) score arrays, one per image.
    Returns (oLRP, components) or None if no GT exists for this class.
    """
    N_GT = sum(len(g) for g in gt_boxes_all)
    if N_GT == 0:
        return None

    thresholds = np.linspace(0.0, 1.0, 101)
    best_oLRP = 1.0
    best_components = {"oLRP_IoU": 1.0, "oLRP_FP": 1.0, "oLRP_FN": 1.0, "s_star": 0.0}

    for s in thresholds:
        total_tp = 0
        total_fp = 0
        total_iou_sum = 0.0

        for gt_boxes, det_boxes, det_scores in zip(gt_boxes_all, det_boxes_all, det_scores_all):
            keep = det_scores >= s
            s_det_boxes = det_boxes[keep]
            s_det_scores = det_scores[keep]

            if len(s_det_boxes) == 0:
                continue

            if len(gt_boxes) == 0:
                total_fp += len(s_det_boxes)
                continue

            iou_matrix = calculate_iou(s_det_boxes, gt_boxes)  # (D, G)
            matched_gt = set()

            for d_idx in np.argsort(-s_det_scores):
                best_gt_idx = np.argmax(iou_matrix[d_idx])
                max_iou = iou_matrix[d_idx, best_gt_idx]

                if max_iou >= tau and best_gt_idx not in matched_gt:
                    total_tp += 1
                    total_iou_sum += max_iou
                    matched_gt.add(best_gt_idx)
                else:
                    total_fp += 1

        N_TP = total_tp
        N_FP = total_fp
        N_FN = N_GT - N_TP
        denominator = N_TP + N_FP + N_FN
        if denominator == 0:
            continue

        lrp_iou = (N_TP - total_iou_sum) / (N_TP * (1 - tau)) if N_TP > 0 else 0.0
        lrp_fp  = N_FP / (N_TP + N_FP) if (N_TP + N_FP) > 0 else 1.0
        lrp_fn  = N_FN / N_GT

        lrp_total = ((total_tp - total_iou_sum) / (1 - tau) + N_FP + N_FN) / denominator

        if lrp_total < best_oLRP:
            best_oLRP = lrp_total
            best_components = {
                "oLRP_IoU": lrp_iou,
                "oLRP_FP":  lrp_fp,
                "oLRP_FN":  lrp_fn,
                "s_star":   float(s),
            }

    return best_oLRP, best_components


def load_coco_gt(gt_path):
    """Return {image_id: {cat_id: [xyxy boxes]}} from a COCO-format GT file."""
    with open(gt_path) as f:
        data = json.load(f)

    gt = defaultdict(lambda: defaultdict(list))
    for ann in data['annotations']:
        img_id = ann['image_id']
        cat_id = ann['category_id']
        x, y, w, h = ann['bbox']
        gt[img_id][cat_id].append([x, y, x + w, y + h])
    return gt


def load_coco_det(det_path):
    """Return {image_id: {cat_id: [(xyxy box, score)]}} from a COCO detection file."""
    with open(det_path) as f:
        data = json.load(f)

    det = defaultdict(lambda: defaultdict(list))
    for ann in data:
        img_id = ann['image_id']
        cat_id = ann['category_id']
        x, y, w, h = ann['bbox']
        det[img_id][cat_id].append(([x, y, x + w, y + h], ann['score']))
    return det


def compute_molrp(gt_path, det_path, num_classes, tau=0.5, class_names=None):
    gt_data  = load_coco_gt(gt_path)
    det_data = load_coco_det(det_path)

    all_image_ids = sorted(set(gt_data.keys()) | set(det_data.keys()))

    # Per-class lists of (boxes, scores) indexed by image, in the same order
    class_gt         = {c: [] for c in range(num_classes)}
    class_det_boxes  = {c: [] for c in range(num_classes)}
    class_det_scores = {c: [] for c in range(num_classes)}

    for img_id in all_image_ids:
        for c in range(num_classes):
            gt_boxes = np.array(gt_data[img_id][c], dtype=np.float32).reshape(-1, 4)
            class_gt[c].append(gt_boxes)

            entries = det_data[img_id][c]
            if entries:
                boxes  = np.array([e[0] for e in entries], dtype=np.float32)
                scores = np.array([e[1] for e in entries], dtype=np.float32)
            else:
                boxes  = np.zeros((0, 4), dtype=np.float32)
                scores = np.zeros(0, dtype=np.float32)
            class_det_boxes[c].append(boxes)
            class_det_scores[c].append(scores)

    all_olrp     = []
    all_olrp_iou = []
    all_olrp_fp  = []
    all_olrp_fn  = []

    header = f"{'Class':<16} | {'oLRP':>6} | {'oLRP_IoU':>8} | {'oLRP_FP':>7} | {'oLRP_FN':>7} | {'s*':>5}"
    lines = [header, "-" * len(header)]

    for c in range(num_classes):
        result = evaluate_lrp_for_class(
            class_gt[c], class_det_boxes[c], class_det_scores[c], tau
        )
        if result is None:
            continue

        olrp, comp = result
        all_olrp.append(olrp)
        all_olrp_iou.append(comp["oLRP_IoU"])
        all_olrp_fp.append(comp["oLRP_FP"])
        all_olrp_fn.append(comp["oLRP_FN"])

        name = (class_names[c] if class_names and c < len(class_names) else str(c))
        lines.append(f"{name:<16} | {olrp:>6.4f} | {comp['oLRP_IoU']:>8.4f} | "
                      f"{comp['oLRP_FP']:>7.4f} | {comp['oLRP_FN']:>7.4f} | {comp['s_star']:>5.2f}")

    lines.append("\n" + "=" * len(header))
    lines.append(f"moLRP     : {np.mean(all_olrp):.4f}  (lower is better)")
    lines.append(f"moLRP_IoU : {np.mean(all_olrp_iou):.4f}  "
                  f"(avg tightness = {100*(1-np.mean(all_olrp_iou)):.1f}%)")
    lines.append(f"moLRP_FP  : {np.mean(all_olrp_fp):.4f}")
    lines.append(f"moLRP_FN  : {np.mean(all_olrp_fn):.4f}")
    lines.append("=" * len(header))

    logger.info("moLRP evaluation results:\n" + "\n".join(lines))

    return {
        "moLRP":     float(np.mean(all_olrp)),
        "moLRP_IoU": float(np.mean(all_olrp_iou)),
        "moLRP_FP":  float(np.mean(all_olrp_fp)),
        "moLRP_FN":  float(np.mean(all_olrp_fn)),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute moLRP from COCO-format JSON files")
    parser.add_argument("--gt",          default="gt_refined.json",   help="COCO GT JSON path")
    parser.add_argument("--det",         default="refined_pred.json", help="COCO detection JSON path")
    parser.add_argument("--num_classes", type=int, default=30,        help="Number of classes")
    parser.add_argument("--tau",         type=float, default=0.5,     help="IoU matching threshold")
    parser.add_argument("--no_names",    action="store_true",         help="Print class index instead of name")
    args = parser.parse_args()

    names = None if args.no_names else VID_CLASSES[:args.num_classes]
    compute_molrp(args.gt, args.det, args.num_classes, tau=args.tau, class_names=names)
