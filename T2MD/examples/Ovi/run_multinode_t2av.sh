export PYTHONIOENCODING=UTF-8

ALL_NNODES=$(echo $NODE_IP_LIST|sed 's/,/\n/g'|wc -l)
export NNODES=${NNODES:-${ALL_NNODES}}
export NODE_RANK=${NODE_RANK:-${INDEX}}
export MASTER_ADDR=${MASTER_ADDR:-${CHIEF_IP}}
export MASTER_PORT=${MASTER_PORT:-29500}
export NO_TORCH_COMPILE=1

export NCCL_DEBUG=WARN
NET_TYPE="high"
if [[ "${NET_TYPE}" = "low" ]]; then
    export NCCL_SOCKET_IFNAME=eth1
    export NCCL_IB_GID_INDEX=3
    export NCCL_IB_HCA=mlx5_2:1,mlx5_2:1
    export NCCL_IB_SL=3
    export NCCL_CHECK_DISABLE=1
    export NCCL_P2P_DISABLE=0
    export NCCL_LL_THRESHOLD=16384
    export NCCL_IB_CUDA_SUPPORT=1
else
    export NCCL_IB_TIMEOUT=24
    export NCCL_IB_GID_INDEX=3
    export NCCL_IB_SL=3
    export NCCL_CHECK_DISABLE=1
    export NCCL_P2P_DISABLE=0
    export NCCL_IB_DISABLE=0
    export NCCL_LL_THRESHOLD=16384
    export NCCL_IB_CUDA_SUPPORT=1
    export NCCL_SOCKET_IFNAME=bond1
    export UCX_NET_DEVICES=bond1
    export NCCL_IB_HCA=mlx5_bond_1,mlx5_bond_5,mlx5_bond_3,mlx5_bond_7,mlx5_bond_4,mlx5_bond_8,mlx5_bond_2,mlx5_bond_6
    export NCCL_COLLNET_ENABLE=0
    export SHARP_COLL_ENABLE_SAT=0
    export NCCL_NET_GDR_LEVEL=2
    export NCCL_IB_QPS_PER_CONNECTION=4
    export NCCL_IB_TC=160
    export NCCL_PXN_DISABLE=1
fi

TOTAL_WORKERS=$((NNODES * 8))

echo "Total $NNODES machines, current machine $NODE_RANK, master ${MASTER_ADDR}:${MASTER_PORT}, world size $TOTAL_WORKERS"

accelerate launch --config_file examples/Ovi/accelerate_multinodes_zero2.yaml \
    examples/Ovi/train_t2av.py \
  --dataset_csv_path ./code/4s_pipeline/crawler_code/a_v_caption_filtered.jsonl \
  --dataset_num_workers 8 \
  --height 480 \
  --width 480 \
  --num_frames 117 \
  --dataset_repeat 1 \
  --learning_rate 1e-5 \
  --num_epochs 10 \
  --remove_prefix_in_ckpt "pipe.model." \
  --output_path "models/train/Ovi_sft" \
  --trainable_models "model" \
  --save_steps 1000 \
  --audio_vae_ckpt ./models/vae/g_01240000 \
  --audio_vae_stat ./models/vae/global_mean_var_124w.stat \
  &> log.16gpus
