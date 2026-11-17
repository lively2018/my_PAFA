#!/usr/bin/env bash
# E1: wait for the gt_idx collection run to finish, then run the per-object analysis over all objects.
#
#   cd ~/my_PAFA_e1
#   nohup bash tools/run_e1_per_object_all.sh > E1_outputs_obj_all.log 2>&1 &
#   tail -f E1_outputs_obj_all.log
#
# Waits for the running `vid_eval.py ... --e1_dump <NPZ>` process (or PID given as $1), checks the npz
# was saved (it is written once at the very end), then runs tools/e1_per_object.py (CPU only).
set -euo pipefail

ROOT=$HOME/my_PAFA_e1
NPZ=E1_outputs/e1_candidates_gtidx.npz
RUNLOG=E1_outputs_fullval_gtidx.log
GT=gt_refined.json
OUT=E1_outputs/e1_fullval_obj_all
PY=$HOME/miniforge3/envs/YOLOV_exm_fixfeat/bin/python   # has scipy

cd "$ROOT"

# main process of the collection run: the oldest vid_eval.py with this npz (others are its workers)
PID=${1:-$(pgrep -of "vid_eval.py.*--e1_dump $NPZ" || true)}
if [ -n "$PID" ]; then
    echo "[$(date '+%F %T')] waiting for collection run PID $PID ..."
    while kill -0 "$PID" 2>/dev/null; do sleep 60; done
    echo "[$(date '+%F %T')] collection run finished"
else
    echo "[$(date '+%F %T')] no running collection process found, checking outputs"
fi

# the npz is saved after the last batch; make sure it exists and the run logged the save
if [ ! -s "$NPZ" ]; then
    echo "ERROR: $NPZ not found -- collection did not finish (see $RUNLOG)"; exit 1
fi
if ! grep -q "\[E1\] .* candidates .* -> $NPZ" "$RUNLOG"; then
    echo "WARNING: save line '[E1] N candidates ... -> $NPZ' not in $RUNLOG (continuing, npz exists)"
fi
grep -E "Average Precision.*IoU=0.50 .*all" "$RUNLOG" | head -1 || true   # AP50 of the run (expected ~0.763)

"$PY" - "$NPZ" <<'EOF'
import sys, numpy as np
d = np.load(sys.argv[1])
assert 'gt_idx' in d.files, 'gt_idx missing -- collector without gt_idx was used'
print('npz ok: candidates', len(d['gt_iou']), '| frames', len(d['frame_keys']), '| GT boxes', int(d['frame_num_gt'].sum()))
EOF

echo "[$(date '+%F %T')] per-object analysis -> ${OUT}_*"
"$PY" tools/e1_per_object.py --npz "$NPZ" --gt "$GT" --out "$OUT"
echo "[$(date '+%F %T')] done"
