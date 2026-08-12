"""
Draw per-class RP curves with the optimal oLRP operating point (s*) marked.
Supports overlaying multiple models on each class subplot.

GT format  (gt_refined.json / gt_ori.json):
    {"annotations": [{"image_id": int, "category_id": int, "bbox": [x,y,w,h]}, ...], ...}

Det format (refined_pred.json / ori_pred.json):
    [{"image_id": int, "category_id": int, "bbox": [x,y,w,h], "score": float}, ...]

Usage — all classes, two models:
    python tools/plot_rp_curve_per_class.py \
        --gt gt_refined.json \
        --det refined_pred.json ori_pred.json \
        --names "Refined" "Original" \
        --out_dir rp_per_class/

Usage — single class by index:
    python tools/plot_rp_curve_per_class.py \
        --gt gt_refined.json \
        --det refined_pred.json \
        --names "Refined" \
        --class_id 2
"""

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
from collections import defaultdict

VID_CLASSES = [
    'airplane', 'antelope', 'bear', 'bicycle', 'bird',
    'bus', 'car', 'cattle', 'dog', 'domestic_cat',
    'elephant', 'fox', 'giant_panda', 'hamster', 'horse',
    'lion', 'lizard', 'monkey', 'motorcycle', 'rabbit',
    'red_panda', 'sheep', 'snake', 'squirrel', 'tiger',
    'train', 'turtle', 'watercraft', 'whale', 'zebra',
]

LINESTYLE = ['-', '--', '-.', ':', (0, (3, 1, 1, 1))]
MARKERS   = ['x', '+', 'D', 's', '^']

def _class_colors(n):
    """Return n visually distinct colors, one per class."""
    cmap = plt.colormaps['tab20'].resampled(n) if n <= 20 else plt.colormaps['hsv'].resampled(n)
    return [cmap(i) for i in range(n)]


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_coco_gt(gt_path):
    with open(gt_path) as f:
        data = json.load(f)
    gt = defaultdict(lambda: defaultdict(list))
    for ann in data['annotations']:
        x, y, w, h = ann['bbox']
        gt[ann['image_id']][ann['category_id']].append([x, y, x + w, y + h])
    return gt


def load_coco_det(det_path):
    with open(det_path) as f:
        data = json.load(f)
    det = defaultdict(lambda: defaultdict(list))
    for ann in data:
        x, y, w, h = ann['bbox']
        det[ann['image_id']][ann['category_id']].append(
            ([x, y, x + w, y + h], ann['score'])
        )
    return det


def build_class_data(gt_data, det_data, num_classes):
    all_image_ids = sorted(set(gt_data.keys()) | set(det_data.keys()))
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

    return class_gt, class_det_boxes, class_det_scores


# ---------------------------------------------------------------------------
# IoU
# ---------------------------------------------------------------------------

def calculate_iou(boxes1, boxes2):
    """(N,4) × (M,4) in [xmin,ymin,xmax,ymax] → (N,M) IoU matrix."""
    if len(boxes1) == 0 or len(boxes2) == 0:
        return np.zeros((len(boxes1), len(boxes2)))

    b1 = np.array(boxes1, dtype=np.float32)
    b2 = np.array(boxes2, dtype=np.float32)

    ix_min = np.maximum(b1[:, 0:1], b2[:, 0])
    iy_min = np.maximum(b1[:, 1:2], b2[:, 1])
    ix_max = np.minimum(b1[:, 2:3], b2[:, 2])
    iy_max = np.minimum(b1[:, 3:4], b2[:, 3])

    iw = np.maximum(0.0, ix_max - ix_min)
    ih = np.maximum(0.0, iy_max - iy_min)
    intersection = iw * ih

    area1 = (b1[:, 2] - b1[:, 0]) * (b1[:, 3] - b1[:, 1])
    area2 = (b2[:, 2] - b2[:, 0]) * (b2[:, 3] - b2[:, 1])
    union = area1[:, None] + area2[None, :] - intersection
    return intersection / (union + 1e-6)


# ---------------------------------------------------------------------------
# Core curve computation
# ---------------------------------------------------------------------------

def generate_class_curve_data(gt_boxes_all, det_boxes_all, det_scores_all, tau=0.5):
    """
    Sweep 101 thresholds and collect (precision, recall) pairs plus the
    optimal oLRP operating point (s*).

    Returns (precisions, recalls, (opt_recall, opt_precision)) or None.
    """
    N_GT = sum(len(g) for g in gt_boxes_all)
    if N_GT == 0:
        return None

    thresholds  = np.linspace(0.0, 1.0, 101)
    precisions  = []
    recalls     = []
    best_oLRP   = 1.0
    optimal_pt  = (0.0, 1.0)   # (recall, precision) at s*

    for s in thresholds:
        total_tp      = 0
        total_fp      = 0
        total_iou_sum = 0.0

        for gt_boxes, det_boxes, det_scores in zip(gt_boxes_all, det_boxes_all, det_scores_all):
            keep         = det_scores >= s
            s_det_boxes  = det_boxes[keep]
            s_det_scores = det_scores[keep]

            if len(s_det_boxes) == 0:
                continue
            if len(gt_boxes) == 0:
                total_fp += len(s_det_boxes)
                continue

            iou_matrix = calculate_iou(s_det_boxes, gt_boxes)
            matched_gt = set()

            for d_idx in np.argsort(-s_det_scores):
                best_gt_idx = np.argmax(iou_matrix[d_idx])
                max_iou     = iou_matrix[d_idx, best_gt_idx]

                if max_iou >= tau and best_gt_idx not in matched_gt:
                    total_tp      += 1
                    total_iou_sum += max_iou
                    matched_gt.add(best_gt_idx)
                else:
                    total_fp += 1

        N_TP  = total_tp
        N_FP  = total_fp
        N_FN  = N_GT - N_TP

        precision = N_TP / (N_TP + N_FP) if (N_TP + N_FP) > 0 else 1.0
        recall    = N_TP / N_GT
        precisions.append(precision)
        recalls.append(recall)

        denominator = N_TP + N_FP + N_FN
        if denominator > 0:
            lrp_total = ((total_tp - total_iou_sum) / (1 - tau) + N_FP + N_FN) / denominator
            if lrp_total < best_oLRP:
                best_oLRP  = lrp_total
                optimal_pt = (recall, precision)

    return np.array(precisions), np.array(recalls), optimal_pt


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _draw_class_axes(ax, curves, class_name, class_color, show_legend=True):
    """
    Draw one class panel on `ax`.
    curves      : list of (precisions, recalls, optimal_pt, model_name, model_idx)
    class_color : fixed color for this class; models are distinguished by linestyle
    """
    has_data = False
    for precisions, recalls, (opt_r, opt_p), name, midx in curves:
        ls = LINESTYLE[midx % len(LINESTYLE)]
        mk = MARKERS[midx % len(MARKERS)]

        auc = np.trapz(precisions, recalls)
        ax.plot(recalls, precisions,
                color=class_color, linestyle=ls, linewidth=1.8,
                label=f"{name} (AUC={auc:.3f})")
        ax.scatter(opt_r, opt_p,
                   color=class_color, marker=mk, s=80, linewidths=2.5,
                   zorder=5, label=f"{name} s*")
        has_data = True

    ax.set_title(class_name, fontsize=9, fontweight='bold')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall',    fontsize=7)
    ax.set_ylabel('Precision', fontsize=7)
    ax.tick_params(labelsize=6)
    ax.grid(True, linestyle=':', alpha=0.5)
    if show_legend and has_data:
        ax.legend(fontsize=5, loc='lower left')


def plot_single_class(curves, class_name, class_color, out_path):
    """Full-size figure for one class."""
    fig, ax = plt.subplots(figsize=(5, 5))
    _draw_class_axes(ax, curves, class_name, class_color, show_legend=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved → {out_path}")


def plot_all_classes_grid(all_curves, class_names, class_colors, out_path, ncols=5):
    """
    Grid of subplots: one panel per class that has GT.
    all_curves   : {class_id: list-of-curve-tuples}
    class_colors : list of colors indexed by class_id
    """
    valid_ids = sorted(all_curves.keys())
    n         = len(valid_ids)
    nrows     = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(ncols * 3.2, nrows * 3.2))
    axes = np.array(axes).reshape(-1)

    for plot_idx, cls_id in enumerate(valid_ids):
        _draw_class_axes(axes[plot_idx], all_curves[cls_id],
                         class_names[cls_id], class_colors[cls_id],
                         show_legend=(plot_idx == 0))

    for ax in axes[len(valid_ids):]:
        ax.set_visible(False)

    fig.suptitle("Per-class RP Curves", fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(gt_path, det_paths, model_names, num_classes, tau, class_id, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    gt_data      = load_coco_gt(gt_path)
    class_names  = VID_CLASSES[:num_classes]
    class_colors = _class_colors(num_classes)   # one distinct color per class

    # Build per-model class data
    model_class_data = []
    for det_path in det_paths:
        det_data = load_coco_det(det_path)
        cgt, cdb, cds = build_class_data(gt_data, det_data, num_classes)
        model_class_data.append((cgt, cdb, cds))

    target_classes = [class_id] if class_id is not None else range(num_classes)
    all_curves = {}   # class_id → list of curve tuples

    for c in target_classes:
        curves = []
        for midx, (name, (cgt, cdb, cds)) in enumerate(zip(model_names, model_class_data)):
            result = generate_class_curve_data(cgt[c], cdb[c], cds[c], tau)
            if result is None:
                continue
            precisions, recalls, opt_pt = result
            curves.append((precisions, recalls, opt_pt, name, midx))

        if not curves:
            continue
        all_curves[c] = curves

        if class_id is not None:
            out_path = os.path.join(out_dir, f"rp_class_{class_names[c]}.png")
            plot_single_class(curves, class_names[c], class_colors[c], out_path)

    if class_id is None:
        grid_path = os.path.join(out_dir, "rp_all_classes.png")
        plot_all_classes_grid(all_curves, class_names, class_colors, grid_path, ncols=5)

        for c, curves in all_curves.items():
            out_path = os.path.join(out_dir, f"rp_class_{class_names[c]}.png")
            plot_single_class(curves, class_names[c], class_colors[c], out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Per-class RP curves from COCO JSON files")
    parser.add_argument("--gt",          default="gt_refined.json")
    parser.add_argument("--det",         nargs="+", default=["refined_pred.json"])
    parser.add_argument("--names",       nargs="+", default=None,
                        help="Model name for each --det file")
    parser.add_argument("--num_classes", type=int,   default=30)
    parser.add_argument("--tau",         type=float, default=0.5)
    parser.add_argument("--class_id",   type=int,   default=None,
                        help="Draw only this class index (omit for all classes)")
    parser.add_argument("--out_dir",     default="rp_per_class",
                        help="Directory to save output PNG files")
    args = parser.parse_args()

    names = args.names if args.names else args.det
    if len(names) != len(args.det):
        parser.error("--names must have the same number of entries as --det")

    run(
        gt_path     = args.gt,
        det_paths   = args.det,
        model_names = names,
        num_classes = args.num_classes,
        tau         = args.tau,
        class_id    = args.class_id,
        out_dir     = args.out_dir,
    )
