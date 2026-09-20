#!/bin/bash
cd /mnt/ssdraid/project/h3-opt/harness
PY=/mnt/ssdraid/project/comfy-h3/.venv-exp/bin/python
PORT=${1:-29701}
pids=()
for r in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$((4+r)) RANK=$r WORLD_SIZE=4 MASTER_ADDR=127.0.0.1 MASTER_PORT=$PORT SP_TIMING=1 \
    "$PY" sp_full_step.py > /mnt/ssdraid/project/h3-opt/logs/fullsp_r${r}.log 2>&1 &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done
