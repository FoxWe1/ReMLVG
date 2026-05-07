#!/usr/bin/env bash
set -euo pipefail

NUM_GPUS="${NUM_GPUS:-8}"
LOG_DIR="${LOG_DIR:-outputs/logs}"
RUN_NAME="${RUN_NAME:-dior_rsvg_visual_idear19-1}"

mkdir -p "${LOG_DIR}"

PIDS=()
for ((gpu = 0; gpu < NUM_GPUS; gpu++)); do
  log_file="${LOG_DIR}/${RUN_NAME}_shard${gpu}.log"
  echo "Starting shard ${gpu}/${NUM_GPUS} on GPU ${gpu}; log=${log_file}"
  CUDA_VISIBLE_DEVICES="${gpu}" nohup bash ./visual.sh \
    --num_shards "${NUM_GPUS}" \
    --shard_id "${gpu}" \
    > "${log_file}" 2>&1 &
  PIDS+=("$!")
done

pid_file="${LOG_DIR}/${RUN_NAME}.pids"
printf "%s\n" "${PIDS[@]}" > "${pid_file}"
echo "Started ${NUM_GPUS} visualization workers. PID file: ${pid_file}"
