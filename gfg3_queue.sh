#!/bin/bash
# GFG-v3 后台排队: 冒烟通过 → pilot 训练 (后台, 200 文件, 3000 基因)
export OPENBLAS_NUM_THREADS=8
PY=/home/yuchang/venv-gfg/bin/python
cd /home/yuchang/GFG_velocity_foundation_mousebrain_v1

echo "[queue $(date '+%F %T')] smoke test..."
CUDA_VISIBLE_DEVICES=0 $PY gfg3_train.py \
    --n-files 24 --n-genes 3000 --config original \
    --epochs-a 2 --epochs-b 2 --epochs-c 0 \
    --batch 32 --block 32 \
    --out /tmp/opencode/gfg3_smoke > /home/yuchang/data_download/gfg3_smoke.log 2>&1

if ! grep -q "DONE" /home/yuchang/data_download/gfg3_smoke.log; then
    echo "[queue $(date '+%F %T')] SMOKE FAILED - see gfg3_smoke.log"
    exit 1
fi
echo "[queue $(date '+%F %T')] smoke OK → pilot training (200 files, Stage A+B)"

CUDA_VISIBLE_DEVICES=0,1 $PY gfg3_train.py \
    --n-files 200 --n-genes 3000 --config original \
    --epochs-a 30 --epochs-b 30 --epochs-c 0 \
    --batch 64 --block 64 --lr 1e-3 \
    --out /data/dataset/gfg3_pilot > /home/yuchang/data_download/gfg3_pilot.log 2>&1
echo "[queue $(date '+%F %T')] pilot done"
