#CUDA_VISIBLE_DEVICES='1' python -m torch.distributed.launch --nproc_per_node=1  --master_port 29500 --use_env \
#main.py --dataset_file rsvg --binary --with_box_refine \
#--batch_size 2 --num_frames 1 --epochs 70 --lr_drop 40 --num_queries 10 \
#--output_dir outputs/lqvg_1 --backbone resnet50 --lr      \


##调整bs为8,训练DIOR-RSVG
#CUDA_VISIBLE_DEVICES='1' python -m torch.distributed.launch --nproc_per_node=1  --master_port 29500 --use_env \
#main.py --dataset_file rsvg --binary --with_box_refine \
#--batch_size 16 --num_frames 1 --epochs 70 --lr_drop 40 --num_queries 10 \
#--output_dir outputs/lqvg_2 --backbone resnet50 --lr  8e-4 \
#--lr_backbone 4e-4  --lr_poolout 8e-4 \
#--rsvg_path ../../datasets/DIOR-RSVG \
#--resume outputs/lqvg_2/checkpoint0020.pth

#--------------------------------------rsvg-hr-------------------------------------
#CUDA_VISIBLE_DEVICES='0' python main.py \
#--dataset_file rsvg_hr --binary --with_box_refine \
#--batch_size 2 --num_frames 1 --epochs 70 --lr_drop 40 60 --num_queries 10 \
#--output_dir outputs/rsvg_hr_5 --backbone resnet50 \
#--rsvg_hr_path ../../datasets/RSVG-HR \


#CUDA_VISIBLE_DEVICES='0' python main.py \
#--dataset_file rsvg_hr --binary --with_box_refine \
#--batch_size 2 --num_frames 1 --epochs 70 --lr_drop 40 60 --num_queries 10 \
#--output_dir outputs/rsvg_hr_5 --backbone resnet50 \
#--rsvg_hr_path ../../datasets/RSVG-HR \
# --pretrained_weights weights/DIOR-RSVG/checkpoint.pth
#( 复现)

#CUDA_VISIBLE_DEVICES='0' python main.py \
#--dataset_file rsvg_hr --binary --with_box_refine \
#--batch_size 2 --num_frames 1 --epochs 70 --lr_drop 40 60 --num_queries 10 \
#--output_dir outputs/rsvg_hr_6 --backbone resnet50 \
#--rsvg_hr_path ../../datasets/RSVG-HR \
#--pretrained_weights outputs/dior_rsvg_1/checkpoint.pth

# -------------------------------------dior-rsvg-------------------------------------
#CUDA_VISIBLE_DEVICES=1 python -u main.py \
#  --dataset_file dior_rsvg --dior_rsvg_path ../../datasets/DIOR-RSVG \
#  --binary --with_box_refine --backbone resnet50 \
#  --batch_size 4 --num_frames 1 \
#  --epochs 70 --lr_drop 40 60\
#  --num_queries 10 \
#  --output_dir outputs/dior_rsvg_2/ \


# 在V100上4张卡
# CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch \
#   --nproc_per_node=4 --master_port=29501 --use_env \
#   main.py \
#   --dataset_file dior_rsvg --dior_rsvg_path ../../datasets/DIOR-RSVG \
#   --binary --with_box_refine --backbone resnet50 \
#   --batch_size 4 --num_frames 1 \
#   --epochs 70 --lr_drop 40 60 \
#   --num_queries 10 \
#   --output_dir outputs/dior_rsvg_1/ \
#   --text_pretrained ../../pretrained/roberta-base

CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch \
  --nproc_per_node=4 --master_port=29501 --use_env \
  main.py \
  --dataset_file dior_rsvg --dior_rsvg_path ../../datasets/DIOR-RSVG \
  --binary --with_box_refine \
  --batch_size 4 --num_frames 1 --epochs 70 --lr_drop 40 60 \
  --num_queries 10 --num_feature_levels 4 \
  --output_dir outputs/dior_rsvg_1/ \
  --encoder_mode remoteclip \
  --remoteclip_model_name ViT-L-14 \
  --remoteclip_checkpoint ../../pretrained/CLIP/RemoteCLIP-ViT-L-14.pt \
  --remoteclip_visual_layers 0,7,15,23 \
  --remoteclip_aup_factors 4,2,2,1 \
  --freeze_remoteclip_visual \
  --freeze_remoteclip_text \
  --lr_backbone 0 --lr_text_encoder 0 \
  --text_pretrained ../../pretrained/roberta-base \
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



# 目前流程里仍会先初始化一次 LQVG 原文本编码器再替换，所以 --text_pretrained 先给一个可用路径。



#续跑
#CUDA_VISIBLE_DEVICES=1 python -u main.py \
#  --dataset_file dior_rsvg --dior_rsvg_path ../../datasets/DIOR-RSVG \
#  --binary --with_box_refine --backbone resnet50 \
#  --batch_size 4 --num_frames 1 \
#  --epochs 90 --lr_drop 40 60 \
#  --num_queries 10 \
#  --output_dir outputs/dior_rsvg_2/ \
#  --lr_backbone 1e-4 \
#  --resume outputs/dior_rsvg_2/checkpoint0069.pth

# -------------------------------------rsvg-------------------------------------
#CUDA_VISIBLE_DEVICES=1 python -u main.py \
#  --dataset_file rsvg --rsvg_hr_path ../../datasets/RSVG \
#  --binary --with_box_refine --backbone resnet50 \
#  --batch_size 2 --num_frames 1 \
#  --epochs 70 --lr_drop 40 \
#  --num_queries 10 \
#  --output_dir outputs/rsvg_3/ \
#  --pretrained_weights weights/checkpoint.pth
# 根据给出的预训练权重微调

## --------------------------------------------------original--------------------------------------------------

#CUDA_VISIBLE_DEVICES='2,7' python -m torch.distributed.launch --nproc_per_node=2  --master_port 29500 --use_env \
#main.py --dataset_file rsvg --binary --with_box_refine \
#--batch_size 2 --num_frames 1 --epochs 70 --lr_drop 40 --num_queries 10 \
#--output_dir rsvg_dirs/r50_bidrection_fusion_10query_70epo_multiscale --backbone resnet50 \

# CUDA_VISIBLE_DEVICES='2,7' python -m torch.distributed.launch --nproc_per_node=2  --master_port 29500 --use_env \
# main.py --dataset_file rsvg_mm --binary --with_box_refine \
# --batch_size 2 --num_frames 1 --epochs 70 --lr_drop 40 --num_queries 10 \
# --output_dir rsvg_mm_dirs/r50_bidrection_fusion_10query_70epo_pretrain_multiscale --backbone resnet50 \
# --pretrained_weights rsvg_dirs/r50_bidrection_fusion_10query/checkpoint.pth
