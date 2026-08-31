#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

import cv2
import numpy as np

__all__ = ["vis"]


def _box_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    areaB = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])
    union = areaA + areaB - inter
    return inter / union if union > 0 else 0.0


def _text_anchor(x0, y0, x1, y1, txt_w, txt_h, text_pos, img_w=None):
    if text_pos == "center":
        cx = (x0 + x1) // 2
        cy = (y0 + y1) // 2
        half_w = txt_w // 2
        half_h = int(0.75 * txt_h)
        bg_pt1 = [cx - half_w - 1, cy - half_h]
        bg_pt2 = [cx + half_w + 1, cy + half_h]
        org = [cx - half_w, cy + int(0.25 * txt_h)]
    elif text_pos == "top-right":
        bg_pt1 = [x1 - txt_w - 1, y0 + 1]
        bg_pt2 = [x1, y0 + int(1.5 * txt_h)]
        org = [x1 - txt_w, y0 + txt_h]
    elif text_pos == "bottom-left":
        bg_pt1 = [x0, y1 - int(1.5 * txt_h)]
        bg_pt2 = [x0 + txt_w + 1, y1 - 1]
        org = [x0, y1 - int(0.5 * txt_h)]
    elif text_pos == "bottom-right":
        bg_pt1 = [x1 - txt_w - 1, y1 - int(1.5 * txt_h)]
        bg_pt2 = [x1, y1 - 1]
        org = [x1 - txt_w, y1 - int(0.5 * txt_h)]
    else:  # top-left (default)
        bg_pt1 = [x0, y0 + 1]
        bg_pt2 = [x0 + txt_w + 1, y0 + int(1.5 * txt_h)]
        org = [x0, y0 + txt_h]

    # keep the label fully on-screen: shift it right if it runs off the left edge,
    # or left if it runs off the right edge (e.g. a box touching x=0)
    x_shift = 0
    if bg_pt1[0] < 0:
        x_shift = -bg_pt1[0]
    elif img_w is not None and bg_pt2[0] > img_w:
        x_shift = img_w - bg_pt2[0]
    if x_shift:
        bg_pt1[0] += x_shift
        bg_pt2[0] += x_shift
        org[0] += x_shift

    return tuple(bg_pt1), tuple(bg_pt2), tuple(org)


def vis(img, boxes, scores, cls_ids, conf=0.5, class_names=None,t_size = 0.4, color_idx=None, gt_boxes=None,
        text_pos="top-left", one_class=False):

    ious = None
    if gt_boxes:
        ious = []
        for box in boxes:
            x0, y0, x1, y1 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            ious.append(max((_box_iou((x0, y0, x1, y1), gt) for gt in gt_boxes), default=0.0))

    indices = [i for i in range(len(boxes)) if scores[i] >= conf]
    if one_class and ious is not None:
        best_per_cls = {}
        for i in indices:
            cls_id = int(cls_ids[i])
            if cls_id not in best_per_cls or ious[i] > ious[best_per_cls[cls_id]]:
                best_per_cls[cls_id] = i
        indices = sorted(best_per_cls.values())

    for i in indices:
        box = boxes[i]
        cls_id = int(cls_ids[i])
        score = scores[i]
        x0 = int(box[0])
        y0 = int(box[1])
        x1 = int(box[2])
        y1 = int(box[3])
        if color_idx:
            color = (_COLORS[color_idx] * 255).astype(np.uint8).tolist()
        else:
            color = (_COLORS[cls_id%len(_COLORS)] * 255).astype(np.uint8).tolist()
        if ious is not None:
            text = '{}  conf.: {:.2f}  IoU: {:.2f}'.format(class_names[cls_id], score, ious[i])
        else:
            text = '{}:{:.1f}%'.format(class_names[cls_id], score * 100)
        txt_color = (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX

        txt_size = cv2.getTextSize(text, font, t_size, 1)[0]
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)

        txt_bk_color = (0, 100, 0)
        bg_pt1, bg_pt2, org = _text_anchor(x0, y0, x1, y1, txt_size[0], txt_size[1], text_pos, img_w=img.shape[1])
        cv2.rectangle(
            img,
            bg_pt1,
            bg_pt2,
            txt_bk_color,
            -1
        )
        cv2.putText(img, text, org, font, t_size, txt_color, thickness=1)

    return img



_COLORS = np.array(
    [
        0.000, 0.447, 0.741,
        0.850, 0.325, 0.098,
        0.929, 0.694, 0.125,
        0.494, 0.184, 0.556,
        0.466, 0.674, 0.188,
        0.301, 0.745, 0.933,
        0.635, 0.078, 0.184,
        0.300, 0.300, 0.300,
        0.600, 0.600, 0.600,
        1.000, 0.000, 0.000,
        1.000, 0.500, 0.000,
        0.749, 0.749, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 1.000,
        0.667, 0.000, 1.000,
        0.333, 0.333, 0.000,
        0.333, 0.667, 0.000,
        0.333, 1.000, 0.000,
        0.667, 0.333, 0.000,
        0.667, 0.667, 0.000,
        0.667, 1.000, 0.000,
        1.000, 0.333, 0.000,
        1.000, 0.667, 0.000,
        1.000, 1.000, 0.000,
        0.000, 0.333, 0.500,
        0.000, 0.667, 0.500,
        0.000, 1.000, 0.500,
        0.333, 0.000, 0.500,
        0.333, 0.333, 0.500,
        0.333, 0.667, 0.500,
        0.333, 1.000, 0.500,
        0.667, 0.000, 0.500,
        0.667, 0.333, 0.500,
        0.667, 0.667, 0.500,
        0.667, 1.000, 0.500,
        1.000, 0.000, 0.500,
        1.000, 0.333, 0.500,
        1.000, 0.667, 0.500,
        1.000, 1.000, 0.500,
        0.000, 0.333, 1.000,
        0.000, 0.667, 1.000,
        0.000, 1.000, 1.000,
        0.333, 0.000, 1.000,
        0.333, 0.333, 1.000,
        0.333, 0.667, 1.000,
        0.333, 1.000, 1.000,
        0.667, 0.000, 1.000,
        0.667, 0.333, 1.000,
        0.667, 0.667, 1.000,
        0.667, 1.000, 1.000,
        1.000, 0.000, 1.000,
        1.000, 0.333, 1.000,
        1.000, 0.667, 1.000,
        0.333, 0.000, 0.000,
        0.500, 0.000, 0.000,
        0.667, 0.000, 0.000,
        0.833, 0.000, 0.000,
        1.000, 0.000, 0.000,
        0.000, 0.167, 0.000,
        0.000, 0.333, 0.000,
        0.000, 0.500, 0.000,
        0.000, 0.667, 0.000,
        0.000, 0.833, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 0.167,
        0.000, 0.000, 0.333,
        0.000, 0.000, 0.500,
        0.000, 0.000, 0.667,
        0.000, 0.000, 0.833,
        0.000, 0.000, 1.000,
        0.000, 0.000, 0.000,
        0.143, 0.143, 0.143,
        0.286, 0.286, 0.286,
        0.429, 0.429, 0.429,
        0.571, 0.571, 0.571,
        0.714, 0.714, 0.714,
        0.857, 0.857, 0.857,
        0.000, 0.447, 0.741,
        0.314, 0.717, 0.741,
        0.50, 0.5, 0
    ]
).astype(np.float32).reshape(-1, 3)
