import argparse

def get_args_parser():
    parser = argparse.ArgumentParser('ReferFormer training and inference scripts.', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=5e-5, type=float)  #控制是否训练backbone,有学习率就训练
    parser.add_argument('--lr_backbone_names', default=['backbone.0'], type=str, nargs='+')
    parser.add_argument('--lr_text_encoder', default=1e-5, type=float)
    parser.add_argument('--lr_text_encoder_names', default=['text_encoder'], type=str, nargs='+')
    parser.add_argument('--lr_poolout', default=1e-4, type=float)
    parser.add_argument('--lr_poolout_names', default=['poolout_module'], type=str, nargs='+')
    parser.add_argument('--lr_linear_proj_names', default=['reference_points', 'sampling_offsets'], type=str, nargs='+')
    parser.add_argument('--lr_linear_proj_mult', default=1.0, type=float)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--weight_decay', default=5e-4, type=float)
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--lr_drop', default=[6, 8], type=int, nargs='+')
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # Model parameters
    # load the pretrained weights
    parser.add_argument('--pretrained_weights', type=str, default=None,
                        help="Path to the pretrained model.") 

    # Variants of Deformable DETR
    parser.add_argument('--with_box_refine', default=False, action='store_true')
    parser.add_argument('--two_stage', default=False, action='store_true') # NOTE: must be false

    # * Backbone
    # ["resnet50", "resnet101", "swin_t_p4w7", "swin_s_p4w7", "swin_b_p4w7", "swin_l_p4w7"]
    # ["video_swin_t_p4w7", "video_swin_s_p4w7", "video_swin_b_p4w7"]
    parser.add_argument('--backbone', default='resnet50', type=str, 
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--backbone_pretrained', default=None, type=str, 
                        help="if use swin backbone and train from scratch, the path to the pretrained weights") #即便任务上 from scratch，也别让 Swin 从零随机初始化
    parser.add_argument('--use_checkpoint', action='store_true', help='whether use checkpoint for swin/video swin backbone')
    parser.add_argument('--dilation', action='store_true', # DC5
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--num_feature_levels', default=4, type=int, help='number of feature levels')

    # * Transformer
    parser.add_argument('--enc_layers', default=4, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=4, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int, 
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_frames', default=5, type=int,
                        help="Number of clip frames for training")
    parser.add_argument('--num_queries', default=5, type=int,
                        help="Number of query slots, all frames share the same queries") 
    parser.add_argument('--dec_n_points', default=4, type=int)
    parser.add_argument('--enc_n_points', default=4, type=int)
    parser.add_argument('--pre_norm', action='store_true')

    # for text
    parser.add_argument('--freeze_text_encoder', action='store_true') # default: False
    parser.add_argument('--text_pretrained', default="../../pretrained/roberta-base")

    # encoder switch: native LQVG encoders (resnet+roberta) or RemoteCLIP encoders
    parser.add_argument(
        '--encoder_mode',
        default='lqvg',
        type=str,
        choices=('lqvg', 'remoteclip'),
        help='Select encoder stack: lqvg (native) or remoteclip.'
    )
    parser.add_argument('--remoteclip_model_name', default='ViT-L-14', type=str,
                        help='RemoteCLIP model name passed to open_clip.create_model_and_transforms')
    parser.add_argument('--remoteclip_checkpoint', default='', type=str,
                        help='Path to RemoteCLIP checkpoint (.pt)')
    parser.add_argument('--remoteclip_input_size', default=336, type=int,
                        help='Fixed square input size for RemoteCLIP ViT encoders.')
    # muller
    parser.add_argument('--use_remoteclip_muller', action='store_true',
                        help='Use a learnable MULLER resizer before the RemoteCLIP visual tower.')
    parser.add_argument('--remoteclip_muller_input_size', default=800, type=int,
                        help='Fixed square image size before MULLER. MULLER resizes this to --remoteclip_input_size.')
    parser.add_argument('--remoteclip_muller_layers', default=2, type=int,
                        help='Number of Laplacian residual layers in the RemoteCLIP MULLER resizer.')
    parser.add_argument('--remoteclip_muller_kernel_size', default=5, type=int,
                        help='Gaussian kernel size for the RemoteCLIP MULLER resizer.')
    parser.add_argument('--remoteclip_muller_stddev', default=1.0, type=float,
                        help='Gaussian kernel stddev for the RemoteCLIP MULLER resizer.')
    parser.add_argument('--remoteclip_muller_resize_method', default='bicubic', type=str,
                        choices=('bilinear', 'bicubic'),
                        help='Base resize method used by the RemoteCLIP MULLER resizer.')
    parser.add_argument('--remoteclip_muller_antialias', action='store_true',
                        help='Enable antialiasing in the RemoteCLIP MULLER base resizer.') #是否抗锯齿
    parser.add_argument('--lr_remoteclip_muller', default=1e-3, type=float,
                        help='Learning rate for RemoteCLIP MULLER resizer parameters.')
    parser.add_argument('--lr_remoteclip_muller_names', default=['muller_resizer'], type=str, nargs='+',
                        help='Parameter name keywords for RemoteCLIP MULLER LR group.')

    parser.add_argument('--freeze_remoteclip_visual', action='store_true',
                        help='Freeze RemoteCLIP visual encoder parameters')
    parser.add_argument('--freeze_remoteclip_text', action='store_true',
                        help='Freeze RemoteCLIP text encoder parameters')
    parser.add_argument('--disable_remoteclip_renorm', action='store_true',
                        help='Disable ImageNet->CLIP input renormalization before RemoteCLIP visual encoder')
    parser.add_argument('--disable_remoteclip_clip_projection', action='store_true',
                        help='Disable CLIP visual.proj/text_projection before downstream fusion.')
    parser.add_argument('--disable_remoteclip_shared_mscma', action='store_true',
                        help='Disable RemoteCLIP shared-space MSCMA; project to hidden_dim before fusion.')
    parser.add_argument('--remoteclip_visual_layers', default='', type=str,
                        help='Comma-separated ViT block indices for multi-level features, e.g. 0,7,15,23')
    parser.add_argument('--remoteclip_aup_factors', default='4,2,2,1', type=str,
                        help='Comma-separated upsample factors for selected visual layers, e.g. 4,2,2,1')
    parser.add_argument('--use_remoteclip_adapter', action='store_true',
                        help='Attach Houlsby adapters to all RemoteCLIP visual/text transformer blocks.')
    parser.add_argument('--remoteclip_adapter_scope', default='both', type=str,
                        choices=('none', 'visual', 'text', 'both'),
                        help='Where to inject RemoteCLIP adapters when --use_remoteclip_adapter is enabled.')
    parser.add_argument('--remoteclip_adapter_dim', default=64, type=int,
                        help='Bottleneck dim of Houlsby adapters for RemoteCLIP blocks.')
    parser.add_argument('--remoteclip_adapter_dim_visual', default=0, type=int,
                        help='Visual adapter bottleneck dim. <=0 means fallback to --remoteclip_adapter_dim.')
    parser.add_argument('--remoteclip_adapter_dim_text', default=0, type=int,
                        help='Text adapter bottleneck dim. <=0 means fallback to --remoteclip_adapter_dim.')
    parser.add_argument('--lr_remoteclip_adapter', default=1e-4, type=float,
                        help='Fallback learning rate for RemoteCLIP adapter parameters.')
    parser.add_argument('--lr_remoteclip_adapter_visual', default=None, type=float,
                        help='Learning rate for RemoteCLIP visual adapter parameters. Defaults to --lr_remoteclip_adapter.')
    parser.add_argument('--lr_remoteclip_adapter_text', default=None, type=float,
                        help='Learning rate for RemoteCLIP text adapter parameters. Defaults to --lr_remoteclip_adapter.')
    parser.add_argument('--lr_remoteclip_adapter_names', default=['remoteclip_adapter'], type=str, nargs='+',
                        help='Parameter name keywords for RemoteCLIP adapter LR group.')
    parser.add_argument('--lr_remoteclip_adapter_visual_names', default=['backbone'], type=str, nargs='+',
                        help='Parameter name keywords for RemoteCLIP visual adapter LR group.')
    parser.add_argument('--lr_remoteclip_adapter_text_names', default=['text_encoder'], type=str, nargs='+',
                        help='Parameter name keywords for RemoteCLIP text adapter LR group.')

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")
    parser.add_argument('--mask_dim', default=256, type=int, 
                        help="Size of the mask embeddings (dimension of the dynamic mask conv)")
    parser.add_argument('--controller_layers', default=3, type=int, 
                        help="Dynamic conv layer number")
    parser.add_argument('--dynamic_mask_channels', default=8, type=int, 
                        help="Dynamic conv final channel number")
    parser.add_argument('--no_rel_coord', dest='rel_coord', action='store_false',
                        help="Disables relative coordinates")
    
    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # * Matcher
    parser.add_argument('--set_cost_class', default=2, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    # parser.add_argument('--set_cost_mask', default=2, type=float,
    #                     help="mask coefficient in the matching cost")
    # parser.add_argument('--set_cost_dice', default=5, type=float,
    #                     help="mask coefficient in the matching cost")
    # * Loss coefficients
    # parser.add_argument('--mask_loss_coef', default=2, type=float)
    # parser.add_argument('--dice_loss_coef', default=5, type=float)
    parser.add_argument('--cls_loss_coef', default=2, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")
    parser.add_argument('--focal_alpha', default=0.25, type=float)

    # dataset parameters
    # ['ytvos', 'davis', 'a2d', 'jhmdb', 'refcoco', 'refcoco+', 'refcocog', 'dior_rsvg', 'all','rsvg_hr','rsvg']
    # 'all': using the three ref datasets for pretraining
    parser.add_argument('--dataset_file', default='rsvg', help='Dataset name')
    parser.add_argument('--rsvg_path', type=str, default='../../datasets/RSVG')
    parser.add_argument('--dior_rsvg_path', type=str, default='../../datasets/DIOR-RSVG')
    parser.add_argument('--rsvg_mm_path', type=str, default='data/rsvg_mm')
    parser.add_argument('--rsvg_hr_path', type=str, default='../../datasets/RSVG-HR')
    parser.add_argument('--coco_path', type=str, default='data/coco')
    parser.add_argument('--ytvos_path', type=str, default='data/ref-youtube-vos')
    parser.add_argument('--davis_path', type=str, default='data/ref-davis')
    parser.add_argument('--a2d_path', type=str, default='data/a2d_sentences')
    parser.add_argument('--jhmdb_path', type=str, default='data/jhmdb_sentences')
    parser.add_argument('--max_skip', default=3, type=int, help="max skip frame number")
    parser.add_argument('--max_size', default=640, type=int, help="max size for the frame")
    parser.add_argument('--binary', action='store_true')
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=4, type=int)

    # test setting
    parser.add_argument('--threshold', default=0.5, type=float) # binary threshold for mask
    parser.add_argument('--ngpu', default=8, type=int, help='gpu number when inference for ref-ytvos and ref-davis')
    parser.add_argument('--split', default='test', type=str, choices=['valid', 'test'])
    parser.add_argument('--visualize', action='store_true', help='whether visualize the masks during inference')

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--cache_mode', default=False, action='store_true', help='whether to cache images on memory')
    parser.add_argument('--find_unused_parameters', default=False, action='store_true',
                        help='Enable DDP find_unused_parameters. Keep False unless some params are conditionally unused.')
    return parser
