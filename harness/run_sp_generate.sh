#!/bin/bash
# 4-rank Ulysses SP in-process generation on GPU4-7.
# usage: run_sp_generate.sh <job.json> <tag> [master_port]
cd /mnt/ssdraid/project/h3-opt/harness
PY=/mnt/ssdraid/project/comfy-h3/.venv-exp/bin/python
JOB=$1
TAG=${2:-sp4}
PORT=${3:-29711}
mkdir -p /mnt/ssdraid/project/h3-opt/logs
pids=()
for r in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$((4+r)) RANK=$r WORLD_SIZE=4 MASTER_ADDR=127.0.0.1 MASTER_PORT=$PORT \
    "$PY" sp_generate.py "$JOB" "$TAG" > /mnt/ssdraid/project/h3-opt/logs/spgen_${TAG}_r${r}.log 2>&1 &
  pids+=($!)
done
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done
exit $rc
