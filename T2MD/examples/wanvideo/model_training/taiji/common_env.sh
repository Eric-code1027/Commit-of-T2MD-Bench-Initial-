
# +====================================================
# + 太极平台多机多卡设置，网络通信
# +====================================================
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
# 加上该参数避免 NCCL error (由宿主机升级 535 驱动导致的问题). 说是不影响训练性能.
export NCCL_NVLS_ENABLE=0

# +==============================================================================
# + 全局环境变量
# +==============================================================================
export TOKENIZERS_PARALLELISM=false

# +==============================================================================
# + Tools
# +==============================================================================
function find_free_port() {
    # 端口搜索循环
    local start=23456
    local end=33456
    local free_port=23456
    for port in $(seq $start 100 $end)
    do
        # 使用lsof命令检查端口是否被占用，如果未被占用，那么将此端口号赋值给变量并退出搜索
        (echo >/dev/tcp/localhost/$port) >/dev/null 2>&1
        if [[ $? -eq 1 ]]; then
            free_port=$port
            break
        fi
    done
    echo $free_port
}