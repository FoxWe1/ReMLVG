#!/usr/bin/env bash
set -euo pipefail

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python3 -u scripts_fox/draw_gt_pred_boxes.py \
  --dataset_file dior_rsvg \
  --split test \
  --dior_rsvg_path ../../datasets/DIOR-RSVG \
  --num_queries 10 \
  --with_box_refine \
  --binary \
  --num_frames 1 \
  --num_feature_levels 4 \
  --backbone resnet50 \
  --text_pretrained ../../pretrained/roberta-base \
  --encoder_mode remoteclip \
  --remoteclip_model_name ViT-L-14 \
  --remoteclip_checkpoint ../../pretrained/CLIP/RemoteCLIP-ViT-L-14.pt \
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
  --lr_remoteclip_muller 1e-3 \
  --remoteclip_muller_layers 2 \
  --remoteclip_muller_kernel_size 5 \
  --remoteclip_muller_stddev 1.5 \
  --resume outputs/results/dior_rsvg/idear19/idear19-1-rerun1_checkpoint0046.pth \
  --save_dir outputs/results/dior_rsvg/visual/idear19-1 \
  "$@"
