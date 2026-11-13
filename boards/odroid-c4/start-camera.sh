#!/bin/bash
set -euo pipefail
app="$(cd -- "$(dirname -- "$0")" && pwd)"
mkdir -p "$app/outputs"
exec 9>"$app/outputs/.camera.lock"
flock -n 9 || exit 0
exec >>"$app/outputs/camera.log" 2>&1

export PYTHONDONTWRITEBYTECODE=1
export PYMAF_BOLT_THREADS=4 PYMAF_CAP_DOWNSAMPLE_THREADS=1
export OMP_NUM_THREADS=4 OMP_DYNAMIC=FALSE OMP_WAIT_POLICY=PASSIVE
export PYMAF_ORT_DISABLE_SPIN=1 BOLT_BNN_ZERO_PADDING_FIX=1 BOLT_BNN_NUMERIC_ZERO_FIX=0
unset BOLT_TIMING_FILE BOLT_DIAG_DIR BOLT_DIAG_OP_FILTER PYMAF_PHASE_LOG

exec taskset -c 0-3 /usr/bin/python3 -u "$app/tools/camera_auto_fast.py" \
  --root "$app" --backend "$app/bin/rpc_cap4" --backend-lib-dir "$app/lib" \
  --detector-model "$app/models/person_detector/yolov8n.onnx" \
  --device 0 --width 640 --height 480 --capture-fps 30 --opencv-threads 4 \
  --redetect-seconds 1 --preview-only --review-on-save --review-mode mesh \
  --save-obj --duration 31536000 "$@"
