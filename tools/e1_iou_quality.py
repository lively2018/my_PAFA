"""E1: accuracy of the predicted IoU, from the npz written by `tools/vid_eval.py --e1_dump`.

    python tools/e1_iou_quality.py --npz E1_outputs/e1_candidates.npz --out E1_outputs/e1

Candidates: every anchor prediction before pre-NMS / NMS / top-K whose actual IoU (largest IoU
with any GT box of the frame) is above the collection threshold (0.1). p = predicted IoU,
g = actual IoU.

Measures (predicted IoU only):
  Spearman rho (p, g)  main measure -- ranking agreement (candidates are chosen by p)
  Pearson r (p, g)
  MAE, RMSE of p - g, and mean signed error (p - g; negative = under-estimation)
  Calibration curve: p in 10 equal bins [0, 0.1), ..., [0.9, 1.0]; per bin mean p, mean g,
  count; y = x is perfect calibration (points above it: under-estimation)
  ECE = sum_b (n_b / N) * |mean p_b - mean g_b|

Outputs: <out>_metrics.json, <out>_calibration.csv, <out>_calibration.png, printed table.
"""
import argparse
import json

import numpy as np
from scipy import stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz', required=True)
    ap.add_argument('--out', required=True, help='output prefix')
    ap.add_argument('--min_iou', type=float, default=None,
                    help='analyse only actual IoU above this (default: the collection threshold)')
    ap.add_argument('--min_pred', type=float, default=None,
                    help='analyse only predicted IoU >= this (e.g. 0.1; default: no limit)')
    ap.add_argument('--bins', type=int, default=10)
    args = ap.parse_args()

    d = np.load(args.npz)
    p = d['pred_iou'].astype(np.float64)
    g = d['gt_iou'].astype(np.float64)
    min_iou = float(d['min_iou']) if args.min_iou is None else args.min_iou
    m = g > min_iou
    if args.min_pred is not None:
        m &= p >= args.min_pred
    p, g = p[m], g[m]
    n = len(p)
    nf = len(d['frame_keys'])
    cand = d['frame_num_cand']
    if args.min_pred is not None:
        print(f"subset: predicted IoU >= {args.min_pred}")
    print(f"frames {nf}, candidates {n} (actual IoU > {min_iou}), "
          f"per frame mean {cand.mean():.1f} / median {np.median(cand):.0f} / max {cand.max()}")

    err = p - g
    res = {
        'num_frames': int(nf), 'num_candidates': int(n), 'min_actual_iou': min_iou, 'min_pred_iou': args.min_pred,
        'spearman_rho': float(stats.spearmanr(p, g).correlation),
        'pearson_r': float(stats.pearsonr(p, g)[0]),
        'mae': float(np.mean(np.abs(err))),
        'rmse': float(np.sqrt(np.mean(err ** 2))),
        'mean_signed_error': float(np.mean(err)),
        'mean_pred_iou': float(p.mean()), 'mean_actual_iou': float(g.mean()),
    }

    edges = np.linspace(0.0, 1.0, args.bins + 1)
    b = np.clip(np.digitize(p, edges[1:-1]), 0, args.bins - 1)   # p == 1.0 goes to the last bin
    rows, ece = [], 0.0
    for k in range(args.bins):
        sel = b == k
        cnt = int(sel.sum())
        if cnt == 0:
            rows.append((edges[k], edges[k + 1], 0, float('nan'), float('nan'), float('nan')))
            continue
        mp, mg = float(p[sel].mean()), float(g[sel].mean())
        sg = float(g[sel].std())
        ece += cnt / n * abs(mp - mg)
        rows.append((edges[k], edges[k + 1], cnt, mp, mg, sg))
    res['ece'] = float(ece)
    res['calibration'] = [dict(zip(('lo', 'hi', 'count', 'mean_pred', 'mean_actual', 'std_actual'), r)) for r in rows]

    with open(args.out + '_metrics.json', 'w') as f:
        json.dump(res, f, indent=2)
    with open(args.out + '_calibration.csv', 'w') as f:
        f.write('bin_lo,bin_hi,count,mean_pred_iou,mean_actual_iou,std_actual_iou\n')
        for r in rows:
            f.write('%.1f,%.1f,%d,%.4f,%.4f,%.4f\n' % r)

    print(f"Spearman rho {res['spearman_rho']:.4f} | Pearson r {res['pearson_r']:.4f} | "
          f"MAE {res['mae']:.4f} | RMSE {res['rmse']:.4f} | mean(p-g) {res['mean_signed_error']:+.4f} | "
          f"ECE {res['ece']:.4f}")
    print(f"{'bin':>10} {'count':>10} {'mean p':>8} {'mean g':>8} {'std g':>7}")
    for lo, hi, cnt, mp, mg, sg in rows:
        print(f"{lo:.1f}-{hi:.1f} {cnt:>10d} {mp:>8.4f} {mg:>8.4f} {sg:>7.4f}")

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(4.8, 4.2))
        ok = [r for r in rows if r[2] > 0]
        ax.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=1, label='perfect calibration')
        ax.plot([r[3] for r in ok], [r[4] for r in ok], marker='o', color='C0', label='predicted IoU')
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel('Predicted IoU (bin mean)'); ax.set_ylabel('Actual IoU (bin mean)')
        ax2 = ax.twinx()
        ax2.bar([(r[0] + r[1]) / 2 for r in rows], [r[2] for r in rows], width=0.08, alpha=0.2, color='C0')
        ax2.set_ylabel('Candidates'); ax2.set_yscale('log')
        ax.set_zorder(ax2.get_zorder() + 1); ax.patch.set_visible(False)
        ax.legend(loc='upper left', fontsize=8)
        ax.set_title(f"Spearman {res['spearman_rho']:.3f}, ECE {res['ece']:.3f}", fontsize=9)
        fig.tight_layout()
        fig.savefig(args.out + '_calibration.png', dpi=200)
        print('->', args.out + '_calibration.png')
    except Exception as e:  # plotting is optional
        print('plot skipped:', e)


if __name__ == '__main__':
    main()
