#!/usr/bin/env bash

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# -------------------- configurable --------------------
: "${OUTPUT_DIR:=outputs/dior_rsvg_remoteclip_shared_mscma}"
: "${CSV_FILE:=${OUTPUT_DIR}/eval_results.csv}"
: "${GPU_LIST:=4,5,6,7}"                 # comma-separated, e.g. "0,1,2,3"
: "${START_EPOCH:=69}"                   # inclusive
: "${END_EPOCH:=40}"                     # inclusive
: "${DIOR_RSVG_PATH:=../../datasets/DIOR-RSVG}"
: "${BACKBONE:=resnet50}"
: "${TEXT_PRETRAINED:=../../pretrained/roberta-base}"
: "${PYTHON_BIN:=python3}"
: "${DEBUG_FAIL_LINES:=20}"              # print last N lines when a checkpoint fails
# ------------------------------------------------------

set -u

mkdir -p "${OUTPUT_DIR}"
CSV_LOCK="${OUTPUT_DIR}/.eval_results.lock"

IFS=',' read -r -a GPUS <<< "${GPU_LIST}"
if [[ ${#GPUS[@]} -eq 0 ]]; then
  echo "No valid GPU id in GPU_LIST=${GPU_LIST}"
  exit 1
fi

echo "Using GPUs: ${GPU_LIST}"
echo "Checkpoint range: ${START_EPOCH} -> ${END_EPOCH}"
echo "Output CSV: ${CSV_FILE}"

if [[ ! -d "${DIOR_RSVG_PATH}" ]]; then
  echo "DIOR_RSVG_PATH not found: ${DIOR_RSVG_PATH}"
  exit 1
fi
if [[ "${TEXT_PRETRAINED}" == ./* || "${TEXT_PRETRAINED}" == ../* || "${TEXT_PRETRAINED}" == /* ]]; then
  if [[ ! -d "${TEXT_PRETRAINED}" ]]; then
    echo "TEXT_PRETRAINED directory not found: ${TEXT_PRETRAINED}"
    exit 1
  fi
fi
if [[ ! -f "inference_rsvg.py" ]]; then
  echo "inference_rsvg.py not found under: ${SCRIPT_DIR}"
  exit 1
fi

CKPT_PATHS=()
for CKPT in $(seq "${START_EPOCH}" -1 "${END_EPOCH}"); do
  CKPT_ID=$(printf "%04d" "${CKPT}")
  CKPT_PATH="${OUTPUT_DIR}/checkpoint${CKPT_ID}.pth"
  if [[ -f "${CKPT_PATH}" ]]; then
    CKPT_PATHS+=("${CKPT_PATH}")
  else
    echo "Checkpoint not found: ${CKPT_PATH}, skip."
  fi
done

if [[ ${#CKPT_PATHS[@]} -eq 0 ]]; then
  echo "No checkpoints found in range."
  exit 1
fi

echo "ckpt,status,acc05,acc06,acc07,acc08,acc09,meanIoU,cumuIoU" > "${CSV_FILE}"

append_result_line() {
  local line="$1"
  if command -v flock >/dev/null 2>&1; then
    {
      flock -x 200
      echo "${line}" >> "${CSV_FILE}"
    } 200>"${CSV_LOCK}"
  else
    echo "${line}" >> "${CSV_FILE}"
  fi
}

run_eval_worker() {
  local worker_idx="$1"
  local gpu_id="$2"

  local i
  for ((i=worker_idx; i<${#CKPT_PATHS[@]}; i+=${#GPUS[@]})); do
    local ckpt_path ckpt_name ckpt_id
    local cmd_output metric_line status acc05 acc06 acc07 acc08 acc09 meanIoU cumuIoU

    ckpt_path="${CKPT_PATHS[$i]}"
    ckpt_name="$(basename "${ckpt_path}")"
    ckpt_id="${ckpt_name#checkpoint}"
    ckpt_id="${ckpt_id%.pth}"

    echo "[GPU ${gpu_id}] Evaluating checkpoint ${ckpt_id}"

    set +e
    cmd_output=$(
      CUDA_VISIBLE_DEVICES="${gpu_id}" \
      "${PYTHON_BIN}" inference_rsvg.py \
        --dataset_file dior_rsvg\
        --dior_rsvg_path "${DIOR_RSVG_PATH}" \
        --rsvg_hr_path ../../datasets/DIOR-RSVG \
        --num_queries 10 \
        --with_box_refine \
        --binary \
        --backbone "${BACKBONE}" \
        --text_pretrained "${TEXT_PRETRAINED}" \
        --num_frames 1 \
        --resume "${ckpt_path}" \
        --encoder_mode remoteclip \
        --remoteclip_model_name ViT-L-14 \
        --remoteclip_checkpoint ../../pretrained/CLIP/clip-vit-large-patch14 \
        --num_feature_levels 4 \
        --remoteclip_visual_layers 0,7,15,23 \
        --remoteclip_aup_factors 4,2,2,1 \
        --freeze_remoteclip_visual \
        --freeze_remoteclip_text \
        --use_remoteclip_adapter \
        --remoteclip_adapter_scope both \
        --lr_remoteclip_adapter_visual 1e-4 \
        --lr_remoteclip_adapter_text 5e-5 \
        --remoteclip_adapter_dim_visual 512 \
        --remoteclip_adapter_dim_text 384 \
        --use_remoteclip_muller \
        --remoteclip_muller_input_size 800 \
        --remoteclip_input_size 336 \
        --remoteclip_muller_layers 2 \
        --remoteclip_muller_kernel_size 5 \
        --remoteclip_muller_stddev 1.5 \
        --eval 2>&1
    )
    cmd_rc=$?
    set -e

    if [[ ${cmd_rc} -eq 0 ]]; then
      status="ok"
    else
      status="fail"
    fi

    metric_line=$(echo "${cmd_output}" | grep "acc@0.5" | grep -v "Time" | tail -n 1 || true)
    if [[ -z "${metric_line}" ]]; then
      echo "[GPU ${gpu_id}] Warning: no metric line for checkpoint ${ckpt_id} (rc=${cmd_rc})"
      if [[ "${DEBUG_FAIL_LINES}" -gt 0 ]]; then
        echo "[GPU ${gpu_id}] Last ${DEBUG_FAIL_LINES} lines for checkpoint ${ckpt_id}:"
        echo "${cmd_output}" | tail -n "${DEBUG_FAIL_LINES}" | sed "s/^/[GPU ${gpu_id}]   /"
      fi
      append_result_line "${ckpt_id},${status},NA,NA,NA,NA,NA,NA,NA"
      continue
    fi

    acc05=$(echo "${metric_line}" | awk '{print $2}')
    acc06=$(echo "${metric_line}" | awk '{print $4}')
    acc07=$(echo "${metric_line}" | awk '{print $6}')
    acc08=$(echo "${metric_line}" | awk '{print $8}')
    acc09=$(echo "${metric_line}" | awk '{print $10}')
    meanIoU=$(echo "${metric_line}" | awk '{print $12}')
    cumuIoU=$(echo "${metric_line}" | awk '{print $14}')
    append_result_line "${ckpt_id},${status},${acc05},${acc06},${acc07},${acc08},${acc09},${meanIoU},${cumuIoU}"
  done
}

pids=()
for idx in "${!GPUS[@]}"; do
  run_eval_worker "${idx}" "${GPUS[$idx]}" &
  pids+=("$!")
done

for pid in "${pids[@]}"; do
  wait "${pid}"
done

TMP_SORT="${CSV_FILE}.tmp"
{
  head -n 1 "${CSV_FILE}"
  tail -n +2 "${CSV_FILE}" | sort -t, -k1,1r
} > "${TMP_SORT}"
mv "${TMP_SORT}" "${CSV_FILE}"

echo "Done. Results saved to ${CSV_FILE}"
