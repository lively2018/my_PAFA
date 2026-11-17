"""E1: accuracy of the predicted IoU.

For every val frame, every anchor prediction before pre-NMS / NMS / top-K (no selection by any
score) is paired with the GT of that frame. The actual IoU of a candidate is its largest IoU with
any GT box of the frame (class-agnostic, localization quality). Candidates whose actual IoU is
above `min_iou` (default 0.1) are kept.

Saved per candidate (npz):
  pred_iou   predicted IoU (sigmoid of the IoU head)               -- analysed in E1
  gt_iou     actual IoU (max over GT boxes of the frame)            -- analysed in E1
  obj        objectness                                              -- saved only (E2 / response)
  cls_conf   max class probability, cls_pred its class index         -- saved only
  gt_cls_prob probability of the matched GT's class                  -- saved only
  gt_cls     class of the matched GT (largest IoU)
  gt_idx     index of the matched GT box within the frame (identifies the object; per-object analysis)
  level      FPN level of the anchor (0 = stride 8, 1 = 16, 2 = 32)
  frame      index into `frame_keys` ('<video>/<frame>', same key as the motion/blur metadata)
Also per frame: number of GT boxes and of kept candidates.

Boxes and GT are both in input-image pixels (Vid_Val_Transform scales the GT by the same ratio
as the image), so no rescaling is needed.
"""
import os

import numpy as np
import torch
from loguru import logger
from torchvision.ops import box_iou


def frame_key(path):
    parts = path.replace("\\", "/").split("/")
    return f"{parts[-2]}/{os.path.splitext(parts[-1])[0]}"


class E1IoUCollector:
    def __init__(self, out_path, min_iou=0.1):
        self.out_path = out_path
        self.min_iou = float(min_iou)
        self.cols = {k: [] for k in ('pred_iou', 'gt_iou', 'obj', 'cls_conf', 'cls_pred',
                                     'gt_cls_prob', 'gt_cls', 'gt_idx', 'level', 'frame')}
        self.frame_keys = []
        self.frame_num_gt = []
        self.frame_num_cand = []

    @torch.no_grad()
    def update(self, cache, labels, paths):
        """cache: head.e1_cache of this batch; labels: per-frame [n, 5] (cls, x1, y1, x2, y2);
        paths: per-frame image paths. Frames are in the same order as the batch."""
        pred = cache['pred'].float()                    # [B, N, 4 + 1 + C + 1]
        num_classes = pred.shape[-1] - 6
        sizes = [h * w for h, w in cache['hw']]
        level_of = torch.cat([torch.full((n,), i, dtype=torch.int8) for i, n in enumerate(sizes)])
        assert level_of.numel() == pred.shape[1], (level_of.numel(), pred.shape)
        assert len(labels) == pred.shape[0] == len(paths), (len(labels), pred.shape[0], len(paths))

        for i in range(pred.shape[0]):
            fid = len(self.frame_keys)
            self.frame_keys.append(frame_key(paths[i]))
            lab = torch.as_tensor(labels[i]).float().to(pred.device)
            self.frame_num_gt.append(int(lab.shape[0]))
            if lab.shape[0] == 0:
                self.frame_num_cand.append(0)
                continue
            p = pred[i]
            ious = box_iou(p[:, :4], lab[:, 1:5])         # [N, G]
            g, gi = ious.max(dim=1)
            keep = g > self.min_iou
            n = int(keep.sum())
            self.frame_num_cand.append(n)
            if n == 0:
                continue
            pk = p[keep]
            probs = pk[:, 5:5 + num_classes]
            cls_conf, cls_pred = probs.max(dim=1)
            gt_cls = lab[gi[keep], 0].long()
            self.cols['pred_iou'].append(pk[:, 5 + num_classes].cpu().numpy())
            self.cols['gt_iou'].append(g[keep].cpu().numpy())
            self.cols['obj'].append(pk[:, 4].cpu().numpy())
            self.cols['cls_conf'].append(cls_conf.cpu().numpy())
            self.cols['cls_pred'].append(cls_pred.to(torch.int16).cpu().numpy())
            self.cols['gt_cls_prob'].append(probs.gather(1, gt_cls.unsqueeze(1)).squeeze(1).cpu().numpy())
            self.cols['gt_cls'].append(gt_cls.to(torch.int16).cpu().numpy())
            self.cols['gt_idx'].append(gi[keep].to(torch.int16).cpu().numpy())
            self.cols['level'].append(level_of[keep.cpu()].numpy())
            self.cols['frame'].append(np.full(n, fid, dtype=np.int32))

    def save(self):
        out = {}
        for k, v in self.cols.items():
            if v:
                arr = np.concatenate(v)
            else:
                arr = np.zeros(0, dtype=np.float32)
            out[k] = arr.astype(np.float32) if arr.dtype.kind == 'f' else arr
        out['frame_keys'] = np.asarray(self.frame_keys)
        out['frame_num_gt'] = np.asarray(self.frame_num_gt, dtype=np.int32)
        out['frame_num_cand'] = np.asarray(self.frame_num_cand, dtype=np.int32)
        out['min_iou'] = np.float32(self.min_iou)
        d = os.path.dirname(self.out_path)
        if d:
            os.makedirs(d, exist_ok=True)
        np.savez(self.out_path, **out)
        logger.info(f"[E1] {len(out['gt_iou'])} candidates (actual IoU > {self.min_iou}) of "
                    f"{len(self.frame_keys)} frames -> {self.out_path}")
