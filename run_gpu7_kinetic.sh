#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
experiment_root="${project_dir}/experiments/foundation_pilot_20260830"
scratch_dir="${experiment_root}/gpu7_scratch_mousebrain"
pretrain_dir="${experiment_root}/gpu7_kinetic_pretrain"
transfer_dir="${experiment_root}/gpu7_kinetic_mousebrain"
log_dir="${experiment_root}/logs"

mkdir -p "${scratch_dir}" "${pretrain_dir}" "${transfer_dir}" "${log_dir}"
export CUDA_VISIBLE_DEVICES=7
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg

exec >>"${log_dir}/gpu7_pipeline.log" 2>&1

cd "${project_dir}"
"/home/yuchang/venv-gfg/bin/python" foundation_pilot.py transfer --physical-gpu 7 --output-dir "${scratch_dir}" --tag large_scratch --seed 0 --mouse-batch-size 384 --eval-batch-size 128 --mouse-epochs 10 --mouse-lr 3e-4

"/home/yuchang/venv-gfg/bin/python" foundation_pilot.py pretrain --variant kinetic --physical-gpu 7 --output-dir "${pretrain_dir}" --batch-size 192 --workers 2 --max-steps 15000 --max-hours 6.5 --pretrain-lr 2e-4 --kinetic-weight 0.2 --kinetic-warmup-steps 500 --log-every 25 --save-every 2000

"/home/yuchang/venv-gfg/bin/python" foundation_pilot.py transfer --physical-gpu 7 --output-dir "${transfer_dir}" --tag kinetic_pretrained --seed 0 --checkpoint "${pretrain_dir}/pretrain_final.pt" --mouse-batch-size 384 --eval-batch-size 128 --mouse-epochs 10 --mouse-lr 3e-4
