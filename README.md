# ReMLVG: A RemoteCLIP-Based Framework for Remote Sensing Visual Grounding

This repository is for remote sensing visual grounding. Given a natural-language referring expression, the model predicts the bounding box of the described object in a remote sensing image.

Original work:

**Language Query based Transformer with Multi-Scale Cross-Modal Alignment for Visual Grounding on Remote Sensing Images**

Meng Lan, Fu Rong, Hongzan Jiao, Zhi Gao, Lefei Zhang

<p align="center"><img src="docs/framework.png" width="800"/></p>

This codebase keeps the original LQVG training path and adds an experimental RemoteCLIP/CLIP encoder path, multi-layer ViT feature extraction, shared-space MSCMA, MULLER learned resizing, and optional Houlsby adapters. The current default training script uses RemoteCLIP + MULLER without adapters.

## Features

- Supports DIOR-RSVG, RSVG-HR, RSVG, and RSVG-MM dataset entry points.
- Keeps the original LQVG encoder path: ResNet/Swin + RoBERTa.
- Adds a RemoteCLIP/CLIP encoder path via `--encoder_mode remoteclip`.
- Extracts multi-level ViT visual features with options such as `--remoteclip_visual_layers 0,7,15,23`.
- Builds multi-scale features with AUP factors such as `--remoteclip_aup_factors 4,2,2,1`.
- Supports the MULLER learned resizer before the CLIP visual tower.
- Keeps optional Houlsby adapter support for experiments, but adapters are disabled in the default `train.sh`.
- Provides single-checkpoint inference and multi-checkpoint evaluation scripts.

## Repository Layout

```text
.
|-- main.py                         # Training entry point
|-- inference_rsvg.py               # Evaluation / inference entry point
|-- opts.py                         # Command-line options
|-- train.sh                        # Default training script
|-- test.sh                         # Single-checkpoint evaluation examples
|-- one4all.sh                      # Multi-checkpoint evaluation script
|-- datasets/                       # Dataset loaders and transforms
|-- models/
|   |-- LQVG.py                     # Main LQVG model
|   |-- clip_initializer.py         # CLIP/RemoteCLIP replacement, adapters, MULLER integration
|   |-- muller_resizer.py           # PyTorch MULLER implementation
|   |-- deformable_transformer.py   # Transformer modules
|   `-- ops/                        # Deformable Attention CUDA operators
|-- scripts_fox/                    # Visualization and diagnostic scripts
|-- RSVG-HR/Annotations/            # RSVG-HR annotation files
|-- LICENSE                         # Apache License 2.0 text
`-- NOTICE                          # Attribution and modification notice
```

## Installation

The original project was tested with:

- Python 3.7
- PyTorch 1.8.1
- CUDA 11.1

Create the environment and install dependencies:

```bash
conda create -n lqvg python=3.7 -y
conda activate lqvg

conda install pytorch==1.8.1 torchvision==0.9.1 torchaudio==0.8.1 -c pytorch
pip install -r requirements.txt
pip install 'git+https://github.com/facebookresearch/fvcore'
pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
```

For the RemoteCLIP/open_clip path, also install:

```bash
pip install open_clip_torch
```

Build the Deformable Attention CUDA operators:

```bash
cd models/ops
python setup.py build install
cd ../..
```

See [docs/install.md](docs/install.md) for the original installation notes.

## Data Preparation

Run commands from the repository root. Some dataset loaders still depend on relative paths, and the default dataset root is `../../datasets/`.

Recommended layout:

```text
../../datasets/
|-- DIOR-RSVG/
|   |-- JPEGImages/
|   |-- Annotations/
|   |-- train.txt
|   `-- test.txt
|-- RSVG-HR/
|   |-- images/
|   `-- Annotations/
|       |-- rsvg_hr_train.txt
|       `-- rsvg_hr_test.txt
`-- RSVG/
    |-- images/
    |-- rsvg_train.pth
    `-- rsvg_test.pth
```

Dataset sources:

- RSVG-HR uses images from the RSVG dataset. The image source is listed on the [GeoVG page](https://sunyuxi.github.io/publication/GeoVG), and the RSVG-HR annotations are included under `RSVG-HR/Annotations/`.
- DIOR-RSVG can be obtained from the [RSVG-pytorch](https://github.com/ZhanYang-nwpu/RSVG-pytorch?tab=readme-ov-file) project.
- Original LQVG model weights for DIOR-RSVG and RSVG-HR are available from the authors' [Google Drive folder](https://drive.google.com/drive/folders/1uC9TAPOwiIbHcee6hSO_3b2Mwr-zDGtg?usp=drive_link).

## Pretrained Weights

Common paths used by the scripts:

```text
../../pretrained/roberta-base
../../pretrained/CLIP/RemoteCLIP-ViT-L-14.pt
../../pretrained/CLIP/clip-vit-large-patch14
```

Notes:

- `--text_pretrained` must point to a valid RoBERTa path. Even in RemoteCLIP mode, the current build flow first constructs the original LQVG model and then replaces the encoders.
- `--remoteclip_checkpoint` can point to an open_clip/RemoteCLIP `.pt` checkpoint or to a HuggingFace CLIP-format directory.

## Training

The default [train.sh](train.sh) trains on DIOR-RSVG with RemoteCLIP + MULLER and no adapter.

Run:

```bash
bash train.sh
```

Equivalent core command:

```bash
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
  --remoteclip_checkpoint ../../pretrained/CLIP/clip-vit-large-patch14 \
  --remoteclip_visual_layers 0,7,15,23 \
  --remoteclip_aup_factors 4,2,2,1 \
  --freeze_remoteclip_visual \
  --freeze_remoteclip_text \
  --lr_backbone 0 --lr_text_encoder 0 \
  --text_pretrained ../../pretrained/roberta-base \
  --use_remoteclip_muller \
  --remoteclip_muller_input_size 800 \
  --remoteclip_input_size 336 \
  --lr_remoteclip_muller 1e-3 \
  --remoteclip_muller_layers 2 \
  --remoteclip_muller_kernel_size 5 \
  --remoteclip_muller_stddev 1.5
```

Training outputs:

- `outputs/dior_rsvg_1/checkpoint.pth`: latest checkpoint.
- `outputs/dior_rsvg_1/checkpointXXXX.pth`: checkpoint for each epoch.
- `outputs/dior_rsvg_1/log.txt`: training log.

Background run example:

```bash
mkdir -p outputs/logs
nohup bash train.sh > outputs/logs/dior_rsvg_1.log 2>&1 &
```

## Evaluation

Use `inference_rsvg.py` for single-checkpoint evaluation. The evaluation-time model options must match the training-time model options. If a checkpoint was trained without adapters, do not pass adapter options during evaluation.

Example:

```bash
CUDA_VISIBLE_DEVICES=0 python3 inference_rsvg.py \
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
  --remoteclip_checkpoint ../../pretrained/CLIP/clip-vit-large-patch14 \
  --remoteclip_visual_layers 0,7,15,23 \
  --remoteclip_aup_factors 4,2,2,1 \
  --freeze_remoteclip_visual \
  --freeze_remoteclip_text \
  --use_remoteclip_muller \
  --remoteclip_muller_input_size 800 \
  --remoteclip_input_size 336 \
  --remoteclip_muller_layers 2 \
  --remoteclip_muller_kernel_size 5 \
  --remoteclip_muller_stddev 1.5 \
  --resume outputs/dior_rsvg_1/checkpoint0069.pth \
  --eval
```

Reported metrics:

- `acc@0.5`
- `acc@0.6`
- `acc@0.7`
- `acc@0.8`
- `acc@0.9`
- `meanIoU`
- `cumuIoU`

## Multi-Checkpoint Evaluation

Use [one4all.sh](one4all.sh) to evaluate a range of checkpoints and write a CSV summary.

```bash
GPU_LIST=4,5,6,7 \
OUTPUT_DIR=outputs/dior_rsvg_1 \
START_EPOCH=69 \
END_EPOCH=40 \
bash one4all.sh
```

Default result file:

```text
outputs/dior_rsvg_1/eval_results.csv
```

Keep `one4all.sh` model options aligned with the checkpoint structure. The committed script is currently configured for no-adapter checkpoints; if you evaluate an adapter checkpoint, add the adapter options back.

## Important Options

| Option | Description |
| --- | --- |
| `--dataset_file` | Dataset name: `dior_rsvg`, `rsvg_hr`, `rsvg`, or `rsvg_mm` |
| `--encoder_mode` | `lqvg` for the original encoder path, `remoteclip` for CLIP/RemoteCLIP |
| `--remoteclip_checkpoint` | RemoteCLIP/CLIP checkpoint path |
| `--remoteclip_visual_layers` | ViT block indices used as multi-level visual features |
| `--remoteclip_aup_factors` | Upsampling factors for the selected visual layers |
| `--freeze_remoteclip_visual` | Freeze the RemoteCLIP visual encoder |
| `--freeze_remoteclip_text` | Freeze the RemoteCLIP text encoder |
| `--use_remoteclip_muller` | Enable the MULLER learned resizer |
| `--remoteclip_muller_input_size` | External input size before MULLER |
| `--remoteclip_input_size` | Target CLIP/RemoteCLIP visual input size |
| `--use_remoteclip_adapter` | Optional Houlsby adapters inside RemoteCLIP blocks |
| `--disable_remoteclip_shared_mscma` | Disable shared-space MSCMA and use hidden-dim fusion |
| `--resume` | Resume training or evaluate a checkpoint |
| `--pretrained_weights` | Load pretrained model weights for fine-tuning |

## Adapter Experiments

Adapter support remains in the codebase, but the default training script currently disables adapters.

To enable adapters, add:

```bash
--use_remoteclip_adapter \
--remoteclip_adapter_scope both \
--lr_remoteclip_adapter_visual 1e-4 \
--lr_remoteclip_adapter_text 5e-5 \
--remoteclip_adapter_dim_visual 512 \
--remoteclip_adapter_dim_text 384
```

When adapters are enabled, the original CLIP parameters are frozen and the inserted adapter parameters are trainable. Evaluation must use the same adapter configuration as training.

## Visualization and Utilities

The `scripts_fox/` directory contains auxiliary scripts:

- `draw_gt_pred_boxes.py`: draw ground-truth and predicted boxes.
- `visual_attention.py`: visualize visual attention.
- `visual_text_attention.py`: visualize text attention.
- `check_remote_clip.py`: inspect RemoteCLIP weights and module status.

See comments in [test.sh](test.sh) for examples.

## License and Attribution

This repository is based on LQVG and is distributed under the Apache License 2.0. The full license text is included in [LICENSE](LICENSE), and attribution / modification notes are recorded in [NOTICE](NOTICE).

If you redistribute this repository or a derivative work, follow the Apache-2.0 redistribution conditions:

- Include a copy of the Apache License 2.0.
- Retain upstream copyright, patent, trademark, and attribution notices that apply to the distributed code.
- Keep any upstream `NOTICE` contents if the upstream distribution includes a `NOTICE` file.
- Mark modified files with clear notices stating that they were changed.
- Do not imply endorsement by the original LQVG authors.
- You may license your own modifications under additional terms only if your redistribution still complies with Apache-2.0 for the upstream LQVG code.

For authoritative text, consult the official [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0.html) and the Apache Software Foundation's [license application guidance](https://www.apache.org/legal/apply-license).

## Common Notes

1. Run training and evaluation commands from the repository root.
2. `--text_pretrained` must be valid even in RemoteCLIP mode.
3. Training and evaluation options must match, especially adapter, MULLER, RemoteCLIP checkpoint, visual layers, and AUP factors.
4. If GPU count changes, update both `CUDA_VISIBLE_DEVICES` and `--nproc_per_node`.
5. If CUDA operator loading fails, rebuild `models/ops`.
6. The DIOR-RSVG loader currently expects `../../datasets/DIOR-RSVG/train.txt` and `../../datasets/DIOR-RSVG/test.txt`.

## Citation

If you use the original LQVG method, cite the original paper. The entry below is provided as a convenient starting point; for publication, verify it against the official paper metadata.

```bibtex
@article{lan2024language,
  title={Language Query based Transformer with Multi-Scale Cross-Modal Alignment for Visual Grounding on Remote Sensing Images},
  author={Lan, Meng and Rong, Fu and Jiao, Hongzan and Gao, Zhi and Zhang, Lefei},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  year={2024}
}
```

If you use MULLER, also cite the MULLER work referenced in `muller-source/README.md`.
