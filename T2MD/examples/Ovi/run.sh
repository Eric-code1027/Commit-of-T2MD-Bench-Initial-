accelerate launch --config_file examples/Ovi/accelerate_zero2.yaml examples/Ovi/train_t2av.py \
  --dataset_base_path /root/Ovi \
  --dataset_csv_path ./datasets/mini_100k_filtered_uniq.csv \
  --dataset_num_workers 8 \
  --dynamic_duration \
  --height 640 \
  --width 640 \
  --num_frames 121 \
  --dataset_repeat 1 \
  --learning_rate 1e-5 \
  --gradient_accumulation_steps 64 \
  --num_epochs 10 \
  --remove_prefix_in_ckpt "pipe.model." \
  --output_path "models/train/Ovi_sft_debug" \
  --trainable_models "model" &> log.v4


# accelerate launch examples/Ovi/train_t2av.py \
#   --dataset_base_path /root/Ovi \
#   --dataset_csv_path ./mini_100k_full_info.csv \
#   --dataset_num_workers 8 \
#   --dynamic_duration \
#   --height 640 \
#   --width 640 \
#   --num_frames 121 \
#   --dataset_repeat 1 \
#   --learning_rate 1e-5 \
#   --gradient_accumulation_steps 8 \
#   --save_steps 50 \
#   --num_epochs 2 \
#   --remove_prefix_in_ckpt "pipe.model." \
#   --output_path "models/train/Ovi_sft" \
#   --trainable_models "model" # &> log


## inference
# torchrun --nnodes 1 --nproc_per_node 8 inference.py --config-file ovi/configs/inference/inference_sft.yaml
# torchrun --nnodes 1 --nproc_per_node 8 inference.py --config-file ovi/configs/inference/inference_fusion.yaml


# TODO: large batch training
