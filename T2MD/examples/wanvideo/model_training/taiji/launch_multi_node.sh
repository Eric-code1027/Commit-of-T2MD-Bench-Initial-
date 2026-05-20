#!/bin/bash

# +====================================================
# + 多节点启动器 - 通过 SSH 在所有节点上启动训练脚本
# + 只需在主节点 (rank 0) 上运行此脚本即可
# +====================================================
# 用法: ./launch_multi_node.sh [-clean]
#   -clean: 清理所有 __pycache__ 目录

# 解析命令行参数
CLEAN_CACHE=false
for arg in "$@"; do
    if [ "$arg" == "-clean" ]; then
        CLEAN_CACHE=true
    fi
done

JOBS_DIR=$(dirname "$0")
REPO_ROOT=$(cd "${JOBS_DIR}/../../../.." && pwd)

# 清理 __pycache__ 目录
if [ "$CLEAN_CACHE" = true ]; then
    echo "[INFO] Cleaning __pycache__ directories in ${REPO_ROOT} ..."
    find ${REPO_ROOT} -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null
    echo "[INFO] Cache cleaned."
    sleep 2
fi
source ${JOBS_DIR}/common_env.sh

# 检查是否存在 hostfile 环境变量, 如果不存在, 则默认使用 /etc/taiji/hostfile
if [ -z "${hostfile}" ]; then
    hostfile=/etc/taiji/hostfile
fi

if [ ! -f "${hostfile}" ]; then
    echo "[ERROR] hostfile not found: ${hostfile}"
    exit 1
fi

echo "========== Hostfile 内容 =========="
cat $hostfile
echo "===================================="

# 从 hostfile 读取节点 IP 列表
NODES=($(awk '{print $1}' $hostfile))
HOST_NUM=${#NODES[@]}
MAIN_IP=${NODES[0]}

# 固定 master port，确保所有节点使用相同端口
MASTER_PORT=${MASTER_PORT:-$(find_free_port)}

# 训练脚本的绝对路径
# TRAIN_SCRIPT="$(cd ${JOBS_DIR} && pwd)/Evan-Wan2.2-S2V-5B-multi-node.sh"
# TRAIN_SCRIPT="$(cd ${JOBS_DIR} && pwd)/Evan-Wan2.2-S2V-DD-5B-multi-node.sh"
TRAIN_SCRIPT="$(cd ${JOBS_DIR} && pwd)/Evan-Wan2.2-S2V-5B-multi-node.sh"
# TRAIN_SCRIPT="$(cd ${JOBS_DIR} && pwd)/Evan-Wan2.2-S2V-2pt5B-multi-node.sh"
# TRAIN_SCRIPT="$(cd ${JOBS_DIR} && pwd)/Evan-Wan2.2-S2V-5B-PIM-multi-node.sh"

echo "========== 多节点启动配置 =========="
echo "HOST_NUM:     ${HOST_NUM}"
echo "MAIN_IP:      ${MAIN_IP}"
echo "MASTER_PORT:  ${MASTER_PORT}"
echo "TRAIN_SCRIPT: ${TRAIN_SCRIPT}"
for i in "${!NODES[@]}"; do
    echo "  Node ${i}: ${NODES[$i]}"
done
echo "===================================="

# 检查训练脚本是否存在
if [ ! -f "${TRAIN_SCRIPT}" ]; then
    echo "[ERROR] Train script not found: ${TRAIN_SCRIPT}"
    exit 1
fi

# 收集所有后台进程的 PID
PIDS=()

# 先启动非主节点 (rank 1, 2, 3, ...)
for ((i=1; i<${HOST_NUM}; i++)); do
    NODE_IP=${NODES[$i]}
    echo "[INFO] Starting rank ${i} on node ${NODE_IP} via SSH ..."
    ssh -o StrictHostKeyChecking=no -f ${NODE_IP} \
        "export RANK=${i} MASTER_PORT=${MASTER_PORT}; bash ${TRAIN_SCRIPT}" &
    PIDS+=($!)
done

# 最后在本地启动主节点 (rank 0)，前台运行
echo "[INFO] Starting rank 0 on local node (${MAIN_IP}) ..."
export RANK=0
export MASTER_PORT=${MASTER_PORT}
bash ${TRAIN_SCRIPT}
LOCAL_EXIT_CODE=$?

# 等待所有后台 SSH 进程完成
echo "[INFO] Waiting for all remote nodes to finish ..."
for pid in "${PIDS[@]}"; do
    wait $pid 2>/dev/null
done

echo "[INFO] All nodes finished. Local exit code: ${LOCAL_EXIT_CODE}"
exit ${LOCAL_EXIT_CODE}
