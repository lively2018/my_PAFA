import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os
import time
import re
import numpy as np



def parse_scalar(val):
    m = re.search(r'tensor\(([-+]?\d*\.?\d+(?:e[-+]?\d+)?)', str(val))
    return float(m.group(1)) if m else float(val)

def nms(boxes, scores, iou_threshold=0.5):
    """Standard NMS. boxes: (N,4) x1y1x2y2, scores: (N,). Returns kept indices."""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        ix1 = np.maximum(x1[i], x1[order[1:]])
        iy1 = np.maximum(y1[i], y1[order[1:]])
        ix2 = np.minimum(x2[i], x2[order[1:]])
        iy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, ix2 - ix1) * np.maximum(0, iy2 - iy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[1:][iou <= iou_threshold]

    return keep

def diou_nms(boxes, scores, iou_threshold=0.5):
    """DIoU-NMS. Suppresses box j if DIoU(i, j) >= iou_threshold.
    DIoU = IoU - (center_distance^2 / enclosing_diagonal^2)
    Boxes with far-apart centers get lower DIoU → less likely to be suppressed.
    boxes: (N,4) x1y1x2y2, scores: (N,). Returns kept indices."""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        # IoU
        ix1 = np.maximum(x1[i], x1[order[1:]])
        iy1 = np.maximum(y1[i], y1[order[1:]])
        ix2 = np.minimum(x2[i], x2[order[1:]])
        iy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, ix2 - ix1) * np.maximum(0, iy2 - iy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter)

        # Center distance squared
        center_dist2 = (cx[i] - cx[order[1:]]) ** 2 + (cy[i] - cy[order[1:]]) ** 2

        # Enclosing box diagonal squared
        enc_x1 = np.minimum(x1[i], x1[order[1:]])
        enc_y1 = np.minimum(y1[i], y1[order[1:]])
        enc_x2 = np.maximum(x2[i], x2[order[1:]])
        enc_y2 = np.maximum(y2[i], y2[order[1:]])
        enc_diag2 = (enc_x2 - enc_x1) ** 2 + (enc_y2 - enc_y1) ** 2 + 1e-7

        diou = iou - center_dist2 / enc_diag2
        order = order[1:][diou <= iou_threshold]

    return keep

def soft_nms(boxes, scores, sigma=0.5, score_threshold=0.001, method='gaussian'):
    """Soft-NMS. Instead of hard suppression, decays scores of overlapping boxes.
    method='gaussian': score *= exp(-iou^2 / sigma)
    method='linear':   score *= (1 - iou) if iou > sigma else score unchanged
    boxes: (N,4) x1y1x2y2, scores: (N,). Returns kept indices (score >= score_threshold)."""
    x1, y1, x2, y2 = boxes[:, 0].copy(), boxes[:, 1].copy(), boxes[:, 2].copy(), boxes[:, 3].copy()
    areas = (x2 - x1) * (y2 - y1)
    scores = scores.copy()
    order = scores.argsort()[::-1].tolist()

    keep = []
    while order:
        i = order[0]
        keep.append(i)
        order = order[1:]

        for j in order[:]:
            ix1 = max(x1[i], x1[j])
            iy1 = max(y1[i], y1[j])
            ix2 = min(x2[i], x2[j])
            iy2 = min(y2[i], y2[j])
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            iou = inter / (areas[i] + areas[j] - inter)

            if method == 'gaussian':
                scores[j] *= np.exp(-(iou ** 2) / sigma)
            else:  # linear
                if iou > sigma:
                    scores[j] *= (1 - iou)

            if scores[j] < score_threshold:
                order.remove(j)

        # Re-sort remaining by decayed scores
        order = sorted(order, key=lambda x: scores[x], reverse=True)

    return keep

def _box_iou(b1, b2):
    """IoU between two boxes [x1,y1,x2,y2]."""
    ix1, iy1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    ix2, iy2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0

def seq_nms(frames_boxes, frames_scores, link_iou=0.5, suppress_iou=0.3, score_threshold=0.001):
    """Seq-NMS across frames.
    frames_boxes : list of (N_t, 4) float32 arrays, one per frame (x1y1x2y2)
    frames_scores: list of (N_t,)    float32 arrays, one per frame
    link_iou     : min IoU to link a box in frame t to frame t+1
    suppress_iou : IoU threshold to suppress non-path boxes at each frame in path
    score_threshold: stop when best path score <= this
    Returns: list of (frame_idx, box_idx, boosted_score) for all kept detections.
    """
    T = len(frames_boxes)
    scores  = [s.copy() for s in frames_scores]   # working copies (will be boosted)
    active  = [np.ones(len(s), dtype=bool) for s in scores]
    results = []   # (frame_idx, box_idx, score)

    while True:
        # ---- Forward DP ------------------------------------------------
        # dp[t][i]   = best accumulated score for any path ending at box i, frame t
        # prev[t][i] = box index in frame t-1 that leads to dp[t][i], or -1
        dp   = [np.full(len(scores[t]), -np.inf) for t in range(T)]
        prev = [np.full(len(scores[t]), -1, dtype=int) for t in range(T)]

        for i in range(len(scores[0])):
            if active[0][i]:
                dp[0][i] = scores[0][i]

        for t in range(1, T):
            for i in range(len(scores[t])):
                if not active[t][i]:
                    continue
                dp[t][i] = scores[t][i]          # single-frame sequence baseline
                for j in range(len(scores[t - 1])):
                    if not active[t - 1][j] or dp[t - 1][j] < 0:
                        continue
                    if _box_iou(frames_boxes[t][i], frames_boxes[t - 1][j]) >= link_iou:
                        candidate = dp[t - 1][j] + scores[t][i]
                        if candidate > dp[t][i]:
                            dp[t][i] = candidate
                            prev[t][i] = j

        # ---- Find best path end-point ----------------------------------
        best_score, best_t, best_i = -np.inf, -1, -1
        for t in range(T):
            for i in range(len(dp[t])):
                if active[t][i] and dp[t][i] > best_score:
                    best_score, best_t, best_i = dp[t][i], t, i

        if best_score <= score_threshold:
            break

        # ---- Backtrack to recover the path -----------------------------
        path = []
        t, i = best_t, best_i
        while i >= 0:
            path.append((t, i))
            i = prev[t][i]
            t -= 1
        path.reverse()   # chronological order

        # ---- Boost path scores & record --------------------------------
        path_max = max(scores[t][i] for t, i in path)
        for t, i in path:
            scores[t][i] = path_max
            results.append((t, i, path_max))
            active[t][i] = False

        # ---- Suppress overlapping boxes at each path frame -------------
        for t, path_i in path:
            for j in range(len(frames_boxes[t])):
                if not active[t][j]:
                    continue
                if _box_iou(frames_boxes[t][path_i], frames_boxes[t][j]) >= suppress_iou:
                    active[t][j] = False

    return results

def check_nested(boxes):
    """Return list of (outer_idx, inner_idx) where inner box is fully contained in outer box."""
    nested = []
    for i in range(len(boxes)):
        for j in range(len(boxes)):
            if i == j:
                continue
            if (boxes[j, 0] >= boxes[i, 0] and boxes[j, 1] >= boxes[i, 1] and
                    boxes[j, 2] <= boxes[i, 2] and boxes[j, 3] <= boxes[i, 3]):
                nested.append((i, j))
    return nested

def suppress_nested(boxes, scores, containment_threshold=0.8):
    """Suppress inner boxes whose area is largely covered by a higher-scoring outer box.
    containment = intersection / area(inner). If >= threshold, inner is suppressed."""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    suppressed = np.zeros(len(boxes), dtype=bool)

    for i in range(len(order)):
        if suppressed[order[i]]:
            continue
        for j in range(i + 1, len(order)):
            if suppressed[order[j]]:
                continue
            ix1 = max(x1[order[i]], x1[order[j]])
            iy1 = max(y1[order[i]], y1[order[j]])
            ix2 = min(x2[order[i]], x2[order[j]])
            iy2 = min(y2[order[i]], y2[order[j]])
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            # Containment ratio relative to the lower-scoring (inner) box
            inner_idx = order[j]
            containment = inter / areas[inner_idx] if areas[inner_idx] > 0 else 0
            if containment >= containment_threshold:
                suppressed[inner_idx] = True

    return np.where(~suppressed)[0]

def make_parser():
    parser = argparse.ArgumentParser("Memory feat file")
    parser.add_argument("--path", type=str, default="detection_after_nms.csv",\
                         help="path to the CSV file containing memory bank stats")
    return parser

def main(args):
    current_time = time.localtime()
    start_time = time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
    log_file_path = os.path.join("./", f'log_{start_time}.txt')
    log_file = open(log_file_path, 'w')

    df = pd.read_csv(args.path)
    # 1. Feature Number Trace - Separated by Level
    for _, det in df.iterrows():
        image_id = int(parse_scalar(det['image id']))
        feature_idx = int(parse_scalar(det['feature idx']))
        bboxes = [int(float(n)) for n in re.findall(r'[-+]?\d*\.?\d+', str(det['bboxes']))[:4]]
        obj_score = parse_scalar(det['obj_score'])
        class_conf = parse_scalar(det['class_conf'])
        class_pred = int(parse_scalar(det['class_pred']))
        log_entry = f"Image ID: {image_id}, Feature Index: {feature_idx}, BBoxes: {bboxes}, Obj Score: {obj_score}, Class Conf: {class_conf}, Class Pred: {class_pred}\n"
        log_file.write(log_entry)
        print(log_entry.strip())  # Also print to console
    for i in sorted(df['image id'].unique()):
        image_dets = df[df['image id'] == i].copy()
        image_dets['_bboxes'] = image_dets['bboxes'].apply(
            lambda v: [float(n) for n in re.findall(r'[-+]?\d*\.?\d+', str(v))[:4]])
        image_dets['_score'] = image_dets.apply(
            lambda r: parse_scalar(r['obj_score']) * parse_scalar(r['class_conf']), axis=1)
        image_dets['_class'] = image_dets['class_pred'].apply(lambda v: int(parse_scalar(v)))
        print(f"Image ID {i}: Total Detections before  DioU NMS: {len(image_dets)}")
        log_file.write(f"\nDetections after DIoU NMS for Image ID {i}:\n")
        total_keep = 0
        for cls in sorted(image_dets['_class'].unique()):
            cls_dets = image_dets[image_dets['_class'] == cls].reset_index(drop=True)
            boxes = np.array(cls_dets['_bboxes'].tolist(), dtype=np.float32)
            scores = np.array(cls_dets['_score'].tolist(), dtype=np.float32)
            nms_keep = np.array(diou_nms(boxes, scores, iou_threshold=0.50))

            for idx in nms_keep:
                row = cls_dets.iloc[idx]
                bboxes = [int(v) for v in row['_bboxes']]
                log_entry = (f"Image ID: {i}, Class: {cls}, BBoxes: {bboxes}, "
                             f"Score: {row['_score']:.4f}, "
                             f"ObjConf: {parse_scalar(row['obj_score']):.4f}, "
                             f"ClsConf: {parse_scalar(row['class_conf']):.4f}\n")
                log_file.write(log_entry)
                print(log_entry.strip())
            total_keep += len(nms_keep)
            print(f"Image ID {i}: Total Detections after DIoU NMS: {total_keep}")
        total_keep = 0
        for cls in sorted(image_dets['_class'].unique()):
            cls_dets = image_dets[image_dets['_class'] == cls].reset_index(drop=True)
            boxes = np.array(cls_dets['_bboxes'].tolist(), dtype=np.float32)
            scores = np.array(cls_dets['_score'].tolist(), dtype=np.float32)
            nms_keep = np.array(nms(boxes, scores, iou_threshold=0.50))

            for idx in nms_keep:
                row = cls_dets.iloc[idx]
                bboxes = [int(v) for v in row['_bboxes']]
                log_entry = (f"Image ID: {i}, Class: {cls}, BBoxes: {bboxes}, "
                             f"Score: {row['_score']:.4f}, "
                             f"ObjConf: {parse_scalar(row['obj_score']):.4f}, "
                             f"ClsConf: {parse_scalar(row['class_conf']):.4f}\n")
                log_file.write(log_entry)
                print(log_entry.strip())
            total_keep += len(nms_keep)
            print(f"Image ID {i}: Total Detections after NMS: {total_keep}")

        log_file.write(f"\nDetections after Soft-NMS for Image ID {i}:\n")
        total_keep = 0
        for cls in sorted(image_dets['_class'].unique()):
            cls_dets = image_dets[image_dets['_class'] == cls].reset_index(drop=True)
            boxes = np.array(cls_dets['_bboxes'].tolist(), dtype=np.float32)
            scores = np.array(cls_dets['_score'].tolist(), dtype=np.float32)
            soft_keep = soft_nms(boxes, scores, sigma=0.5, score_threshold=0.001, method='gaussian')

            for idx in soft_keep:
                row = cls_dets.iloc[idx]
                bboxes = [int(v) for v in row['_bboxes']]
                log_entry = (f"Image ID: {i}, Class: {cls}, BBoxes: {bboxes}, "
                             f"Score: {row['_score']:.4f}, "
                             f"ObjConf: {parse_scalar(row['obj_score']):.4f}, "
                             f"ClsConf: {parse_scalar(row['class_conf']):.4f}\n")
                log_file.write(log_entry)
                print(log_entry.strip())
            total_keep += len(soft_keep)
        print(f"Image ID {i}: Total Detections after Soft-NMS: {total_keep}")
    # ---- Seq-NMS across all frames -------------------------------------
    frame_ids = sorted(df['image id'].unique())

    # Pre-parse all detections once
    df['_bboxes'] = df['bboxes'].apply(
        lambda v: [float(n) for n in re.findall(r'[-+]?\d*\.?\d+', str(v))[:4]])
    df['_score'] = df.apply(
        lambda r: parse_scalar(r['obj_score']) * parse_scalar(r['class_conf']), axis=1)
    df['_class'] = df['class_pred'].apply(lambda v: int(parse_scalar(v)))

    all_classes = sorted(df['_class'].unique())
    log_file.write(f"\n{'='*60}\nSeq-NMS Results (across {len(frame_ids)} frames)\n{'='*60}\n")
    total_seq_keep = 0

    for cls in all_classes:
        cls_df = df[df['_class'] == cls]

        # Build per-frame arrays
        frames_boxes, frames_scores, frames_rows = [], [], []
        for fid in frame_ids:
            fdet = cls_df[cls_df['image id'] == fid].reset_index(drop=True)
            if len(fdet) == 0:
                frames_boxes.append(np.zeros((0, 4), dtype=np.float32))
                frames_scores.append(np.zeros(0, dtype=np.float32))
            else:
                frames_boxes.append(np.array(fdet['_bboxes'].tolist(), dtype=np.float32))
                frames_scores.append(np.array(fdet['_score'].tolist(), dtype=np.float32))
            frames_rows.append(fdet)

        results = seq_nms(frames_boxes, frames_scores, link_iou=0.5, suppress_iou=0.3)

        for (t, box_idx, boosted_score) in results:
            row   = frames_rows[t].iloc[box_idx]
            bboxes = [int(v) for v in row['_bboxes']]
            fid   = frame_ids[t]
            log_entry = (f"[SeqNMS] Frame: {fid}, Class: {cls}, BBoxes: {bboxes}, "
                         f"BoostedScore: {boosted_score:.4f}, "
                         f"ObjConf: {parse_scalar(row['obj_score']):.4f}, "
                         f"ClsConf: {parse_scalar(row['class_conf']):.4f}\n")
            log_file.write(log_entry)
            print(log_entry.strip())
            total_seq_keep += 1

    print(f"Seq-NMS total kept detections across all frames: {total_seq_keep}")
    log_file.write(f"Seq-NMS total kept: {total_seq_keep}\n")
    log_file.close()
if __name__ == "__main__":
    args = make_parser().parse_args()
    print(f"Arguments: {args}")
    main(args)