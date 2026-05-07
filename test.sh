#python3 inference_rsvg.py --dataset_file rsvg --num_queries 10 --with_box_refine --binary --freeze_text_encoder \
#--resume rsvg_dirs/r50_bidrection_fusion_10query/checkpoint.pth --backbone resnet50
#--------------------------——--------------------------——rsvg_hr--------------------------——--------------------------——
#python inference_rsvg.py --dataset_file rsvg_hr --num_queries 10 --with_box_refine --binary --freeze_text_encoder \
#--resume outputs/rsvg_hr_6/checkpoint0069.pth --backbone resnet50 --eval

#--------------------------——--------------------------——DIOR-RSVG--------------------------——--------------------------——
#CUDA_VISIBLE_DEVICES=0 python3 inference_rsvg.py --dataset_file dior_rsvg --num_queries 10 --with_box_refine --binary --freeze_text_encoder \
#--resume weights/DIOR-RSVG/checkpoint.pth --backbone resnet50 --eval --rsvg_path ../../datasets/DIOR-RSVG  #作者提供的预训练权重
#CUDA_VISIBLE_DEVICES=0 python3 inference_rsvg.py --dataset_file dior_rsvg --num_queries 10 --with_box_refine --binary --freeze_text_encoder \
#--resume outputs/dior_rsvg_1/checkpoint.pth  --backbone resnet50 --eval --rsvg_path ../../datasets/DIOR-RSVG
CUDA_VISIBLE_DEVICES=4 python3 inference_rsvg.py \
  --dataset_file dior_rsvg \
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
  --resume outputs/dior_rsvg_remoteclip_shared_mscma/checkpoint0069.pth \
  --eval
#--------------------------——--------------------------——rsvg--------------------------——--------------------------——
#CUDA_VISIBLE_DEVICES=0 python3 inference_rsvg.py --dataset_file rsvg --num_queries 10 --with_box_refine --binary --freeze_text_encoder \
#--resume outputs/rsvg_3/checkpoint.pth --backbone resnet50 --rsvg_path ../../datasets/RSVG --eval
#CUDA_VISIBLE_DEVICES=0 python3 inference_rsvg.py --dataset_file rsvg --num_queries 10 --with_box_refine --binary --freeze_text_encoder \
#--resume weights/RSVG/checkpoint.pth --backbone resnet50 --eval  #作者提供的预训练权重

#--------------------------——--------------------------——可视化--------------------------——--------------------------——
# 可视化gt框
# CUDA_VISIBLE_DEVICES=1 python3 scripts_fox/draw_gt_pred_boxes.py \
#   --dataset_file dior_rsvg \
#   --split test \
#   --num_queries 10 \
#   --with_box_refine \
#   --binary \
#   --freeze_text_encoder \
#   --backbone resnet50 \
#   --resume outputs/results/dior_rsvg/replication/replication_new_2_bs4_lr_backbone_5e-5/replication_new_2_bs4_lr_backbone_5e-5_44.pth \
#   --dior_rsvg_path ../../datasets/DIOR-RSVG \
#   --save_dir outputs/dior_rsvg/boxes_gt_and_infer


# 可视化 text_attention
#CUDA_VISIBLE_DEVICES=1 python scripts_fox/visual_text_attention.py \
#  --dataset_file dior_rsvg \
#  --split test \
#  --sample_idx 0 \
#  --num_queries 10 \
#  --with_box_refine \
#  --binary \
#  --freeze_text_encoder \
#  --backbone resnet50 \
#  --resume outputs/results/dior_rsvg/replication/replication_new_2_bs4_lr_backbone_5e-5/replication_new_2_bs4_lr_backbone_5e-5_44.pth \
#  --dior_rsvg_path ../../datasets/DIOR-RSVG \
#  --save_dir outputs/results/dior_rsvg/text_attention_vis \
#  --topk 20 \
#  --all_samples \
#  --max_samples -1 \
#  --print_freq 100
