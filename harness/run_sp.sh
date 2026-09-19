#!/bin/bash
# Launch 4-GPU sequence-parallel block-loop harness on physical GPU 4,5,6,7.
# One process per GPU; each sees only its GPU as cuda:0. NCCL for Ulysses all-to-all.
cd /mnt/ssdraid/project/h3-opt/harness
PY=/mnt/ssdraid/project/comfy-h3/.venv-exp/bin/python
PORT=${1:-29580}
rm -f /mnt/ssdraid/project/h3-opt/logs/stepC_r*.log
pids=()
for r in 0 1 2 3; do
  gpu=$((4+r))
  CUDA_VISIBLE_DEVICES=$gpu RANK=$r WORLD_SIZE=4 MASTER_ADDR=127.0.0.1 MASTER_PORT=$PORT \
    NCCL_P2P_LEVEL=${NCCL_P2P_LEVEL:-SYS} \
    "$PY" sp_blockloop.py > /mnt/ssdraid/project/h3-opt/logs/stepC_r${r}.log 2>&1 &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done
echo "=== rank0 result ==="
grep -E "verify|bench|Error|Traceback" /mnt/ssdraid/project/h3-opt/logs/stepC_r0.log | tail -15
