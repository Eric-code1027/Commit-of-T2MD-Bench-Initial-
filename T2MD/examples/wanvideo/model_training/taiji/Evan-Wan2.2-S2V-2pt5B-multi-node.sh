#!/bin/bash

# +====================================================
# + 多机多卡训练脚本 - Wan2.2-S2V-5B
# + 参考太极平台多机多卡配置
# +====================================================

JOBS_DIR=$(dirname "$0")
source ${JOBS_DIR}/common_env.sh

# ==========================================================================
# 本项目的根目录
PROJECT_BASE=$(cd ${JOBS_DIR}/../../../.. || exit; pwd)
echo "PROJECT_BASE: ${PROJECT_BASE}"
# 定义启动路径
cd ${PROJECT_BASE} || exit 1
export PYTHONPATH=${PROJECT_BASE}:$PYTHONPATH
# 保存模型检查点和推理结果的根目录
export SAVE_BASE=./exp_dir/wan_s2v
echo "SAVE_BASE: ${SAVE_BASE}"
# ==========================================================================

# 检查是否存在 hostfile 环境变量, 如果不存在, 则默认使用 /etc/taiji/hostfile
if [ -z "${hostfile}" ]; then
    hostfile=/etc/taiji/hostfile
fi
cat $hostfile

# 定义环境变量
export TOKENIZERS_PARALLELISM=false
export LOGURU_COLORIZE=true             # 强制 loguru 终端输出彩色

# 计算节点数量 (根据 hostfile 变量获取实际使用的节点数量)
HOST_NUM=$(wc -l "$hostfile" | awk '{print $1}')
# 获取主节点 IP (hostfile 第一行)
MAIN_IP=$(head -n 1 $hostfile | awk '{print $1}')
# 获取当前节点的 machine_rank (从环境变量 RANK 或 NODE_RANK 获取，默认为 0)
MACHINE_RANK=${RANK:-${NODE_RANK:-0}}
# 每个节点的 GPU 数量
NUM_GPUS_PER_NODE=${NUM_GPUS_PER_NODE:-8}
# 总进程数
NUM_PROCESSES=$((HOST_NUM * NUM_GPUS_PER_NODE))

# 获取当前时间
DATATIME=$(date +%Y%m%d_%H%M%S)
YEAR_MONTH=$(date +%Y%m)
# 定义保存模型权重和相关参数的目录
OUTPUT_DIR="${SAVE_BASE}/talk_head_70w/Evan-Wan2.2-S2V-2pt5B_full_multi_node"
LOG_DIR="${SAVE_BASE}/log_log/${YEAR_MONTH}_train"
if [ ! -d "${LOG_DIR}" ]; then
    mkdir -p $LOG_DIR
fi
# 获取一个空闲端口 (优先使用外部传入的 MASTER_PORT)
free_port=${MASTER_PORT:-$(find_free_port)}

echo "========================================"
echo "HOST_NUM: ${HOST_NUM}"
echo "MAIN_IP: ${MAIN_IP}"
echo "MACHINE_RANK: ${MACHINE_RANK}"
echo "NUM_GPUS_PER_NODE: ${NUM_GPUS_PER_NODE}"
echo "NUM_PROCESSES: ${NUM_PROCESSES}"
echo "FREE_PORT: ${free_port}"
echo "========================================"

set -x
EXP_NAME="Evan-Wan2.2-S2V-2pt5B-multi-node"
TASK_FLAG="${DATATIME}_n${HOST_NUM}_${EXP_NAME}"

# PYTHON_PATH=./local/miniconda3/envs/diffsynth-org/bin/python3.12
PYTHON_PATH=/usr/bin/python
# CONDA_BIN_DIR=$(dirname $PYTHON_PATH)
# export PATH=$CONDA_BIN_DIR:$PATH
accelerate launch \
  --config_file ${PROJECT_BASE}/examples/wanvideo/model_training/taiji/accelerate_config_16nodes.yaml \
  --num_machines ${HOST_NUM} \
  --num_processes ${NUM_PROCESSES} \
  --machine_rank ${MACHINE_RANK} \
  --main_process_ip ${MAIN_IP} \
  --main_process_port ${free_port} \
  ${PROJECT_BASE}/examples/wanvideo/model_training/train_s2v.py \
  --dataset_base_path ./train_datasets/talk_head_70w \
  --dataset_metadata_path ./train_datasets/talk_head_70w/evan_metadata_s2v_with_prompt.csv \
  --data_file_keys "video,input_audio" \
  --height 832 \
  --width 448 \
  --num_frames 121 \
  --dataset_repeat 1 \
  --model_paths '["dummy_s2v_2pt5b_model", "./release_models/Wan2.2-S2V-14B/wav2vec2-large-xlsr-53-english/model.safetensors", "./release_models/Wan2.2-S2V-14B/models_t5_umt5-xxl-enc-bf16.pth", "./release_models/Wan2.2-TI2V-5B/Wan2.2_VAE.pth"]' \
  --tokenizer_path "./release_models/Wan2.2-S2V-14B/google/umt5-xxl/" \
  --audio_processor_path "./release_models/Wan2.2-S2V-14B/wav2vec2-large-xlsr-53-english/" \
  --dataset_num_workers 2 \
  --learning_rate 5e-5 \
  --num_epochs 100 \
  --trainable_models "dit" \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "${OUTPUT_DIR}" \
  --save_steps 1000 \
  --extra_inputs "input_image,input_audio" \
  --use_gradient_checkpointing \
  --gradient_accumulation_steps 1 \
  2>&1 | tee "${LOG_DIR}/${TASK_FLAG}.log"

exit 0
