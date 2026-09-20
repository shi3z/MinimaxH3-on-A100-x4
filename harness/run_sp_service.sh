#!/bin/bash
# Persistent 4-rank Ulysses-SP H3 service on GPU4-7.
# usage: run_sp_service.sh <manifest.json | --serve queuedir> [master_port] [tag]
cd /mnt/ssdraid/project/h3-opt/harness
PY=/mnt/ssdraid/project/comfy-h3/.venv-exp/bin/python
ARG1=$1
ARG2=$2
PORT=${3:-29721}
LOGTAG=${4:-svc}
mkdir -p /mnt/ssdraid/project/h3-opt/logs
pids=()
for r in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$((4+r)) RANK=$r WORLD_SIZE=4 MASTER_ADDR=127.0.0.1 MASTER_PORT=$PORT \
    SP_RANK0_ENCODE=${SP_RANK0_ENCODE:-0} \
    "$PY" sp_service.py "$ARG1" "$ARG2" > /mnt/ssdraid/project/h3-opt/logs/${LOGTAG}_r${r}.log 2>&1 &
  pids+=($!)
done
rc=0; for p in "${pids[@]}"; do wait "$p" || rc=1; done
exit $rc
