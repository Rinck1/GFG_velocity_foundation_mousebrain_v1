#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
experiment_root="${project_dir}/experiments/foundation_pilot_20260830"
pretrain_dir="${experiment_root}/gpu6_denoise_pretrain"
transfer_dir="${experiment_root}/gpu6_denoise_mousebrain"
log_dir="${experiment_root}/logs"

mkdir -p "${pretrain_dir}" "${transfer_dir}" "${log_dir}"
export CUDA_VISIBLE_DEVICES=6
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg

exec >>"${log_dir}/gpu6_pipeline.log" 2>&1

cd "${project_dir}"
"/home/yuchang/venv-gfg/bin/python" foundation_pilot.py pretrain --variant denoise --physical-gpu 6 --output-dir "${pretrain_dir}" --batch-size 512 --workers 2 --max-steps 15000 --max-hours 6.5 --pretrain-lr 2e-4 --log-every 25 --save-every 2000

"/home/yuchang/venv-gfg/bin/python" foundation_pilot.py transfer --physical-gpu 6 --output-dir "${transfer_dir}" --tag denoise_pretrained --seed 0 --checkpoint "${pretrain_dir}/pretrain_final.pt" --mouse-batch-size 384 --eval-batch-size 128 --mouse-epochs 10 --mouse-lr 3e-4
