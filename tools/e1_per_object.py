"""E1: per-object analysis -- for every GT object, the anchor with the highest predicted IoU.

    python tools/e1_per_object.py --npz E1_outputs/e1_candidates.npz --gt gt_refined.json --out E1_outputs/e1_fullval_obj

A candidate belongs to the GT box it overlaps most (actual IoU > 0.1, see e1_iou_collector.py).
The npz does not store the index of that GT box, only the frame and the GT class, so an object is
identified by (frame, class). This is exact only when the class occurs once in the frame; frames
where the class occurs more than once are left out (count printed). If the npz has 'gt_idx'
(index of the matched GT box in the frame), it is used instead and every object is analysed.

Per object:
  p_best  highest predicted IoU among its candidates
  g_best  actual IoU of that anchor
  g_max   highest actual IoU among its candidates (best box available before NMS)
Measures over objects: Spearman / Pearson (p_best, g_best), MAE, RMSE, mean(p_best - g_best),
10-bin calibration and ECE, share of objects with g_best >= 0.5 / 0.7 / 0.9, mean g_best vs g_max,
share of objects with p_best >= 0.1. Objects with no candidate (no anchor with actual IoU > 0.1) are counted.
"""
import argparse
import collections
import json

import numpy as np
from scipy import stats


def calib(p, g, bins=10):
    edges = np.linspace(0.0, 1.0, bins + 1)
    b = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    rows, ece, n = [], 0.0, len(p)
    for k in range(bins):
        sel = b == k
        cnt = int(sel.sum())
        if cnt == 0:
            rows.append((edges[k], edges[k + 1], 0, float('nan'), float('nan'), float('nan')))
            continue
        mp, mg = float(p[sel].mean()), float(g[sel].mean())
        ece += cnt / n * abs(mp - mg)
        rows.append((edges[k], edges[k + 1], cnt, mp, mg, float(g[sel].std())))
    return rows, ece


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz', required=True)
    ap.add_argument('--gt', required=True, help='gt_refined.json written by the same evaluation')
    ap.add_argument('--out', required=True, help='output prefix')
    args = ap.parse_args()

    d = np.load(args.npz)
    p = d['pred_iou'].astype(np.float64)
    g = d['gt_iou'].astype(np.float64)
    frame = d['frame'].astype(np.int64)
    keys = d['frame_keys']
    has_idx = 'gt_idx' in d.files

    gt = json.load(open(args.gt))
    img_of_key = {im['file_name']: im['id'] for im in gt['images']}
    per_img_cls = collections.Counter((a['image_id'], a['category_id']) for a in gt['annotations'])
    n_obj_all = len(gt['annotations'])

    if has_idx:
        obj = frame * 1000 + d['gt_idx'].astype(np.int64)
        n_obj_eval = n_obj_all
        keep = np.ones(len(p), bool)
    else:
        cls = d['gt_cls'].astype(np.int64)
        # class count per (frame, class) from the GT json, frame index -> image id by key
        img_id = np.array([img_of_key.get(str(k), -1) for k in keys], dtype=np.int64)
        assert (img_id >= 0).all(), 'frame keys not found in GT json'
        uniq = {k for k, v in per_img_cls.items() if v == 1}
        u_codes = np.array(sorted(i * 100 + c for i, c in uniq), dtype=np.int64)
        code = img_id[frame] * 100 + cls
        keep = np.isin(code, u_codes)
        obj = code
        n_obj_eval = len(uniq)
    print(f"GT objects {n_obj_all}; analysed objects {n_obj_eval} "
          f"({'gt_idx' if has_idx else 'class unique in frame'}); candidates used {int(keep.sum())} / {len(p)}")

    p, g, obj = p[keep], g[keep], obj[keep]
    # sort by object, then predicted IoU descending -> first row of each object is its best-p anchor
    order = np.lexsort((-p, obj))
    p, g, obj = p[order], g[order], obj[order]
    first = np.r_[True, obj[1:] != obj[:-1]]
    starts = np.flatnonzero(first)
    p_best, g_best = p[starts], g[starts]
    g_max = np.maximum.reduceat(g, starts)
    n = len(starts)
    n_none = n_obj_eval - n
    err = p_best - g_best

    res = {
        'gt_objects': n_obj_all, 'analysed_objects': int(n_obj_eval),
        'objects_with_candidates': int(n), 'objects_without_candidates': int(n_none),
        'identification': 'gt_idx' if has_idx else 'class unique in frame',
        'spearman_rho': float(stats.spearmanr(p_best, g_best).correlation),
        'pearson_r': float(stats.pearsonr(p_best, g_best)[0]),
        'mae': float(np.mean(np.abs(err))), 'rmse': float(np.sqrt(np.mean(err ** 2))),
        'mean_signed_error': float(np.mean(err)),
        'mean_p_best': float(p_best.mean()), 'mean_g_best': float(g_best.mean()),
        'mean_g_max': float(g_max.mean()),
        'share_g_best_ge': {t: float((g_best >= t).mean()) for t in (0.5, 0.7, 0.9)},
        'share_g_max_ge': {t: float((g_max >= t).mean()) for t in (0.5, 0.7, 0.9)},
        'share_p_best_ge_0.1': float((p_best >= 0.1).mean()),
        'share_best_is_max': float((g_best >= g_max - 1e-6).mean()),
    }
    rows, ece = calib(p_best, g_best)
    res['ece'] = float(ece)
    res['calibration'] = [dict(zip(('lo', 'hi', 'count', 'mean_pred', 'mean_actual', 'std_actual'), r)) for r in rows]
    with open(args.out + '_metrics.json', 'w') as f:
        json.dump(res, f, indent=2)
    with open(args.out + '_calibration.csv', 'w') as f:
        f.write('bin_lo,bin_hi,count,mean_pred_iou,mean_actual_iou,std_actual_iou\n')
        for r in rows:
            f.write('%.1f,%.1f,%d,%.4f,%.4f,%.4f\n' % r)

    print(f"objects with candidates {n}, without {n_none}")
    print(f"Spearman {res['spearman_rho']:.4f} | Pearson {res['pearson_r']:.4f} | MAE {res['mae']:.4f} | "
          f"RMSE {res['rmse']:.4f} | mean(p-g) {res['mean_signed_error']:+.4f} | ECE {res['ece']:.4f}")
    print(f"mean p_best {res['mean_p_best']:.4f}, mean g_best {res['mean_g_best']:.4f}, mean g_max {res['mean_g_max']:.4f}")
    print('g_best >= 0.5/0.7/0.9:', ' / '.join('%.4f' % v for v in res['share_g_best_ge'].values()))
    print('g_max  >= 0.5/0.7/0.9:', ' / '.join('%.4f' % v for v in res['share_g_max_ge'].values()))
    print(f"p_best >= 0.1: {res['share_p_best_ge_0.1']:.4f}; best-p anchor is also the best-IoU anchor: {res['share_best_is_max']:.4f}")
    print(f"{'bin':>10} {'count':>9} {'mean p':>8} {'mean g':>8} {'std g':>7}")
    for lo, hi, cnt, mp, mg, sg in rows:
        print(f"{lo:.1f}-{hi:.1f} {cnt:>9d} {mp:>8.4f} {mg:>8.4f} {sg:>7.4f}")


if __name__ == '__main__':
    main()
