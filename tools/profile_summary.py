#!/usr/bin/env python
"""Average the per-forward [PROFILE] lines written by vid_eval.py --profile_time.

Usage:
    python tools/profile_summary.py <eval log> [--warmup 20] [--include-first]

Each [PROFILE] line covers one forward pass of `frames` images. The summary
reports the mean time per forward and per frame for every section, skipping the
first `--warmup` lines and (by default) the first forward of each video, where
the memory is initialized instead of updated.
"""
import argparse
import re
from collections import defaultdict

LINE_RE = re.compile(r"\[PROFILE\] frames=(\d+) first=(True|False) (.*) total=([\d.]+) ms")
PART_RE = re.compile(r"(\w+)=([\d.]+)")
ORDER = ['backbone_fpn', 'head', 'aggregation', 'proposal_select', 'memory_update_pre',
         'msa_refine', 'nms', 'memory_update_post']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('log')
    ap.add_argument('--warmup', type=int, default=20, help='[PROFILE] lines to skip at the start')
    ap.add_argument('--include-first', action='store_true',
                    help='also count the first forward of each video (memory init)')
    args = ap.parse_args()

    rows = []
    with open(args.log, errors='ignore') as f:
        for line in f:
            m = LINE_RE.search(line)
            if m:
                frames, first, parts, total = m.groups()
                rows.append((int(frames), first == 'True', dict(
                    (k, float(v)) for k, v in PART_RE.findall(parts)), float(total)))

    rows = rows[args.warmup:]
    if not args.include_first:
        rows = [r for r in rows if not r[1]]
    if not rows:
        print('no [PROFILE] lines left after filtering')
        return

    sums = defaultdict(float)
    total_sum = 0.0
    frame_sum = 0
    for frames, _, parts, total in rows:
        for k, v in parts.items():
            sums[k] += v
        total_sum += total
        frame_sum += frames

    n = len(rows)
    names = [k for k in ORDER if k in sums] + [k for k in sums if k not in ORDER]
    print(f'forwards counted: {n}  frames counted: {frame_sum}  '
          f'(warmup skipped: {args.warmup}, first-of-video included: {args.include_first})')
    print(f"{'section':<20}{'ms/forward':>12}{'ms/frame':>11}{'share':>8}")
    for k in names:
        print(f'{k:<20}{sums[k] / n:>12.2f}{sums[k] / frame_sum:>11.2f}{sums[k] / total_sum * 100:>7.1f}%')
    print(f"{'total':<20}{total_sum / n:>12.2f}{total_sum / frame_sum:>11.2f}{100.0:>7.1f}%")


if __name__ == '__main__':
    main()
