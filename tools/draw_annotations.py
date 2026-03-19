#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Load my_model_annotation.npy and draw the detections on image frames.

Each annotation entry has shape (N, 6): [x1, y1, x2, y2, score, class_id]
Bounding boxes are in model test-size space and are divided by ratio to map
back to original image coordinates before drawing.

Usage:
    python tools/draw_annotations.py \
        --npy   YOLOX_outputs/yolov_s/vis_res/2026_03_18_17_28_09/my_model_annotation.npy \
        --img_dir  YOLOX_outputs/yolov_s/vis_res/2026_03_18_17_28_09 \
        --output_dir  YOLOX_outputs/yolov_s/vis_res/gt_drawn \
        --test_size 576 \
        --conf 0.05
"""

import argparse
import os
import sys

import cv2
import numpy as np

# ── class names ────────────────────────────────────────────────────────────────
VID_classes = (
    'airplane', 'antelope', 'bear', 'bicycle',
    'bird', 'bus', 'car', 'cattle',
    'dog', 'domestic_cat', 'elephant', 'fox',
    'giant_panda', 'hamster', 'horse', 'lion',
    'lizard', 'monkey', 'motorcycle', 'rabbit',
    'red_panda', 'sheep', 'snake', 'squirrel',
    'tiger', 'train', 'turtle', 'watercraft',
    'whale', 'zebra',
)

IMAGE_EXT = {'.jpg', '.jpeg', '.webp', '.bmp', '.png', '.jpeg'}

# ── color palette (same as yolox/utils/visualize.py) ──────────────────────────
_COLORS = np.array([
    0.000, 0.447, 0.741, 0.850, 0.325, 0.098, 0.929, 0.694, 0.125,
    0.494, 0.184, 0.556, 0.466, 0.674, 0.188, 0.301, 0.745, 0.933,
    0.635, 0.078, 0.184, 0.300, 0.300, 0.300, 0.600, 0.600, 0.600,
    1.000, 0.000, 0.000, 1.000, 0.500, 0.000, 0.749, 0.749, 0.000,
    0.000, 1.000, 0.000, 0.000, 0.000, 1.000, 0.667, 0.000, 1.000,
]).astype(np.float32).reshape(-1, 3)


def draw_boxes(img, boxes, scores, cls_ids, conf=0.05, class_names=VID_classes):
    """Draw bounding boxes with labels on img (in-place). Returns img."""
    for box, score, cls_id in zip(boxes, scores, cls_ids):
        if score < conf:
            continue
        cls_id = int(cls_id)
        x0, y0, x1, y1 = int(box[0]), int(box[1]), int(box[2]), int(box[3])

        color = (_COLORS[cls_id % len(_COLORS)] * 255).astype(np.uint8).tolist()
        txt_color = (0, 0, 0) if np.mean(_COLORS[cls_id % len(_COLORS)]) > 0.5 else (255, 255, 255)
        label = '{}:{:.1f}%'.format(class_names[cls_id], score * 100)

        font = cv2.FONT_HERSHEY_SIMPLEX
        t_size = 0.4
        txt_sz = cv2.getTextSize(label, font, t_size, 1)[0]

        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)
        bk_color = (_COLORS[cls_id % len(_COLORS)] * 255 * 0.7).astype(np.uint8).tolist()
        cv2.rectangle(img, (x0, y0 + 1),
                      (x0 + txt_sz[0] + 1, y0 + int(1.5 * txt_sz[1])), bk_color, -1)
        cv2.putText(img, label, (x0, y0 + txt_sz[1]), font, t_size, txt_color, thickness=1)
    return img


def get_sorted_images(img_dir):
    """Return sorted list of image file paths in img_dir."""
    entries = sorted(os.listdir(img_dir))
    paths = []
    for name in entries:
        ext = os.path.splitext(name)[1].lower()
        if ext in IMAGE_EXT or ext == '.jpeg':
            paths.append(os.path.join(img_dir, name))
    return paths


def main():
    parser = argparse.ArgumentParser(description='Draw my_model_annotation.npy on image frames')
    parser.add_argument('--npy', required=True, help='Path to my_model_annotation.npy')
    parser.add_argument('--img_dir', required=True,
                        help='Directory containing image frames (sorted by name)')
    parser.add_argument('--output_dir', default='annotation_drawn',
                        help='Directory to save output images (default: annotation_drawn)')
    parser.add_argument('--test_size', type=int, default=576,
                        help='Model test size used during inference (default: 576)')
    parser.add_argument('--conf', type=float, default=0.05,
                        help='Confidence threshold for drawing (default: 0.05)')
    args = parser.parse_args()

    # ── load annotations ──────────────────────────────────────────────────────
    annotations = np.load(args.npy, allow_pickle=True)  # shape: (N_frames,)
    print(f'Loaded annotations: {len(annotations)} frames from {args.npy}')

    # ── collect image paths ───────────────────────────────────────────────────
    img_paths = get_sorted_images(args.img_dir)
    print(f'Found {len(img_paths)} images in {args.img_dir}')

    if len(img_paths) != len(annotations):
        print(f'[WARNING] image count ({len(img_paths)}) != annotation count ({len(annotations)}). '
              'Processing min(both).')

    os.makedirs(args.output_dir, exist_ok=True)

    n_frames = min(len(img_paths), len(annotations))
    for idx in range(n_frames):
        img_path = img_paths[idx]
        frame_ann = annotations[idx]  # (N_boxes, 6): x1 y1 x2 y2 score cls_id

        img = cv2.imread(img_path)
        if img is None:
            print(f'[WARNING] Cannot read image: {img_path}, skipping.')
            continue

        h, w = img.shape[:2]
        ratio = min(args.test_size / h, args.test_size / w)

        if frame_ann is not None and len(frame_ann) > 0:
            frame_ann = np.array(frame_ann, dtype=np.float32)
            boxes  = frame_ann[:, 0:4] / ratio   # map from test-size space → original
            scores = frame_ann[:, 4]
            cls_ids = frame_ann[:, 5]
            draw_boxes(img, boxes, scores, cls_ids, conf=args.conf)

        out_name = os.path.basename(img_path)
        out_path = os.path.join(args.output_dir, out_name)
        cv2.imwrite(out_path, img)
        print(f'[{idx+1:4d}/{n_frames}] {out_name}  boxes={len(frame_ann) if frame_ann is not None else 0}')

    print(f'\nDone. Results saved to: {args.output_dir}')


if __name__ == '__main__':
    main()
