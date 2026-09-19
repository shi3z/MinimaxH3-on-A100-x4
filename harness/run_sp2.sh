#!/bin/bash
# 2-way sequence-parallel over ONE NVLink pair. Default GPU 4,5 (NV12).
# Usage: run_sp2.sh [port] [gpuA] [gpuB]
cd /mnt/ssdraid/project/h3-opt/harness
PY=/mnt/ssdraid/project/comfy-h3/.venv-exp/bin/python
PORT=${1:-29590}; GA=${2:-4}; GB=${3:-5}
rm -f /mnt/ssdraid/project/h3-opt/logs/sp2_r*.log
pids=()
for r in 0 1; do
  if [ "$r" = "0" ]; then gpu=$GA; else gpu=$GB; fi
  CUDA_VISIBLE_DEVICES=$gpu RANK=$r WORLD_SIZE=2 MASTER_ADDR=127.0.0.1 MASTER_PORT=$PORT \
    "$PY" sp_blockloop.py > /mnt/ssdraid/project/h3-opt/logs/sp2_r${r}.log 2>&1 &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done
echo "=== rank0 (2-GPU NVLink pair $GA,$GB) ==="
grep -E "components|ms  |bench Step C|verify|Error|Traceback" /mnt/ssdraid/project/h3-opt/logs/sp2_r0.log | tail -18
