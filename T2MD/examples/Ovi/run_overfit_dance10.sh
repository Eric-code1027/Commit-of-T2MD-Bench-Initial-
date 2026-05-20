#!/bin/bash
export TOKENIZERS_PARALLELISM=false
export NO_TORCH_COMPILE=1
export VIDEO_ONLY_TRAINING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

VIDEO_ONLY_TRAINING=1 accelerate launch \
    --config_file examples/Ovi/accelerate_zero3_2gpu.yaml \
    examples/Ovi/train_t2av.py \
    --dataset_csv_path ./datasets/dance_10/dance_caption.jsonl \
    --dataset_base_path "" \
    --dataset_num_workers 2 \
    --height 448 --width 832 --num_frames 121 \
    --dataset_repeat 50 \
    --num_epochs 30 \
    --learning_rate 1e-4 \
    --gradient_accumulation_steps 4 \
    --save_steps 200 \
    --remove_prefix_in_ckpt "pipe.model." \
    --output_path "models/train/overfit_dance10" \
    --lora_base_model "model" \
    --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
    --lora_rank 32 \
    --use_gradient_checkpointing_offload \
    2>&1 | tee log.overfit_dance10