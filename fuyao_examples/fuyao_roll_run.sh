#!/bin/bash
set -x

################################################################################
# 脚本名称：fuyao_roll_run.sh
# 核心功能：fuyao环境的roll框架运行主入口
# 适用环境：Slurm 集群 + 多GPU（支持NVLink）
# 启动方式：命令行传参指定 RUN_TYPE/CONFIG_PATH/CONFIG_NAME，支持多类型启动
# 使用示例：
#  SWANLAB_API_KEY=xxx bash fuyao_examples/fuyao_roll_run.sh --run-type agentic --config-path /path/to/config.yaml --config-name demo
#  SWANLAB_API_KEY=xxx bash fuyao_examples/fuyao_roll_run.sh --run-type distill --config-path /path/to/distill.yaml --config-name distill_demo
################################################################################

# ========================== 0. 安装roll的megatron支持 ==========================
pip install ./mcore_adapter

# ========================== 1. 定义帮助信息 ==========================
show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo "Required Options:"
    echo "  --run-type         Specify run type (required), valid values: agentic/agentic_rollout/distill/distill_vl/dpo"
    echo "  --config-path      Specify config file path (required), e.g., /path/to/config.yaml"
    echo "  --config-name      Specify config name in YAML file (required), e.g., agent_val_frozen_lake_multi_nodes_demo"
    echo "Environment Variables (required, one of):"
    echo "  SWANLAB_API_KEY        Key for swanlab visualization"
    echo "  WANDB_API_KEY          Key for wandb visualization"
    echo "Example:"
    echo "  SWANLAB_API_KEY=xxx $0 --run-type agentic --config-path /data/config.yaml --config-name demo"
    echo "  WANDB_API_KEY=xxx $0 --run-type distill --config-path /data/distill.yaml --config-name distill_demo"
    exit 0
}

# ========================== 2. 解析命令行入参 ==========================
# 参数变量默认值
RUN_TYPE="agentic"
CONFIG_PATH="../fuyao_examples/agentic"  # 注意要退回上级目录，因为执行的脚本位于examples目录，和fuyao yaml目录不在一个
CONFIG_NAME="frozen_lake_demo"

# 解析参数（参考示例逻辑）
while [[ $# -gt 0 ]]; do
    case $1 in
        --run-type)
            RUN_TYPE="$2"
            shift 2
            ;;
        --config-path)
            CONFIG_PATH="$2"
            shift 2
            ;;
        --config-name)
            CONFIG_NAME="$2"
            shift 2
            ;;
        --help)
            show_help
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# ========================== 3. 基础配置：脚本运行模式 ==========================
# 防止Python/Ray缓冲日志，确保实时输出训练日志
export PYTHONBUFFERED=16
# Ray集群通信固定端口（避免端口冲突）
export RAY_PORT="6379"

# ========================== 4. 环境清理：终止残留进程 ==========================
# 目的：重新运行任务时清理上一次的残留进程，避免资源占用/端口冲突
echo "===== Step 1: Clean up residual processes ====="
pkill -9 sglang || true  # 强制终止sglang推理进程（||true避免进程不存在时报错）
sleep 1                  # 等待进程完全终止
pkill -9 vllm || true    # 强制终止vllm推理进程（||true避免进程不存在时报错）
sleep 1                  # 等待进程完全终止
ray stop --force || true # 强制停止Ray集群
pkill -9 ray || true     # 终止Ray残留进程
pkill -9 python || true  # 终止Python训练进程
# 二次清理，确保无残留
sleep 3
pkill -9 ray || true
pkill -9 python || true

# ========================== 5. 前置校验：入参 + 密钥 + 硬件 ==========================
echo -e "\n===== Step 2: Validate required parameters ====="
# 5.1 校验必填命令行参数
if [[ -z "$RUN_TYPE" ]]; then
    echo "Error: --run-type is required!"
    show_help
    exit 1
fi
if [[ -z "$CONFIG_PATH" ]]; then
    echo "Error: --config-path is required!"
    show_help
    exit 1
fi
if [[ -z "$CONFIG_NAME" ]]; then
    echo "Error: --config-name is required!"
    show_help
    exit 1
fi

# 5.2 校验启动类型合法性
VALID_RUN_TYPES=("agentic" "agentic_rollout" "distill" "distill_vl" "dpo" "reward_fl" "rlvr" "rlvr_rollout" "rlvr_vl" "rlvr_vlmath" "sft")
if [[ ! " ${VALID_RUN_TYPES[@]} " =~ " ${RUN_TYPE} " ]]; then
    echo "Error: Invalid --run-type '$RUN_TYPE'! Valid values: ${VALID_RUN_TYPES[*]}"
    show_help
    exit 1
fi
echo "Validated --run-type: $RUN_TYPE"

# 5.3 校验可视化密钥（SWANLAB_API_KEY 或 WANDB_API_KEY 二选一）
if [ -z "$SWANLAB_API_KEY" ] && [ -z "$WANDB_API_KEY" ]; then
    echo "Error: Either SWANLAB_API_KEY or WANDB_API_KEY environment variable must be set!"
    show_help
    exit 1
fi

# 5.4 检测NVLink支持（GPU高速互联）
NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

# ========================== 6. 路径/依赖配置：加载基础参数 ==========================
echo -e "\n===== Step 3: Load paths & dependencies ====="
# 6.1 获取脚本所在绝对路径（兼容任意目录执行脚本）
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

# ========================== 7. 分布式通信配置：NCCL优化 ==========================
echo -e "\n===== Step 4: Configure NCCL for multi-GPU ====="
# NCCL configuration for multi-GPU training
unset NCCL_NET_GDR_LEVEL
export NCCL_IB_QPS_PER_CONNECTION=8
export NCCL_IB_TIMEOUT=22
export NCCL_IB_HCA=mlx5
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export NCCL_CUMEM_ENABLE=0
export NCCL_MAX_NCHANNELS=16

# ========================== 8. 集群适配配置：路径/节点信息 ==========================
echo -e "\n===== Step 5: Configure cluster settings ====="
# 8.1 集群存储路径适配（不同集群对应不同存储目录）
get_storage_path() {
    case "$FUYAO_CLUSTER" in
        "fuyao-sh-b1"|"fuyao-ppu-sh"|"fuyao-b1")
            echo "/dataset_rc"
            ;;
        "fuyao-cpu-c1"|"fuyao-ppu-c1"|"fuyao-ppu-c2"|"fuyao-ppu-c3"|"fuyao-c1"|"fuyao-c2"|"fuyao-c3")
            echo "/dataset-cpfs3-rc"
            ;;
        *)
            echo "/dataset_rc"  # 未知集群默认路径
            ;;
    esac
}
export CPFS_DIR=$(get_storage_path)
# 8.2 从Slurm获取多节点训练信息
NNODES=${SLURM_JOB_NUM_NODES}          # 总节点数
NUMS_GPUS_PER_NODE=${SLURM_GPUS_PER_NODE}  # 每节点GPU数

# 8.3 解析用户名（用于Checkpoint目录区分，若无需可注释）
if [[ -n "$AUTH_USER" ]]; then
    USERNAME=$(echo "$AUTH_USER" | cut -d'@' -f1)
else
    USERNAME="default"
fi

# ========================== 9. 辅助函数：Ray集群健康检查 ==========================
# 检查Ray是否就绪
check_ray_ready() {
    until ray health-check >/dev/null 2>&1; do
        echo "Waiting for Ray to be ready..."
        sleep 1
    done
    echo "Ray is ready"
}

# 检查所有节点是否接入集群
check_cluster_ready() {
    echo "Checking if all $NNODES nodes are connected..."
    while true; do
        node_count=$(python3 -c "
import ray
try:
    ray.init(address='auto', ignore_reinit_error=True)
    alive_nodes = sum(1 for node in ray.nodes() if node['Alive'])
    print(alive_nodes)
    ray.shutdown()
except Exception as e:
    print(0)
" 2>/dev/null)
        
        echo "Found $node_count alive nodes / $NNODES expected"
        if [[ $node_count -ge $NNODES ]]; then
            echo "All nodes are ready!"
            break
        fi
        sleep 5
    done
}

# ========================== 10. 环境变量补充：Megatron-LM和ROLL路径 ==========================
export PYTHONPATH="/root/Megatron-LM/:$PYTHONPATH"
export CUDA_DEVICE_MAX_CONNECTIONS="1"  # 优化CUDA设备连接
ROLL_PATH="/code"
export PYTHONPATH="$ROLL_PATH:$PYTHONPATH"

# ========================== 11. 启动分布式训练：主/从节点逻辑 ==========================
echo -e "\n===== Step 6: Start Ray cluster & training ====="
# 拼接启动命令（核心：根据RUN_TYPE动态生成脚本路径）
LAUNCH_SCRIPT="examples/start_${RUN_TYPE}_pipeline.py"
echo "Generated launch script: $LAUNCH_SCRIPT"

if [[ ${NODE_RANK} == 0 ]]; then
    # 主节点（NODE_RANK=0）：启动Ray头节点 + 执行动态生成的启动命令
    echo "Starting Ray head node (MASTER_ADDR: $MASTER_ADDR)..."
    ray start --head --include-dashboard=True --dashboard-host=0.0.0.0
    check_ray_ready
    check_cluster_ready
    
    echo "Starting $RUN_TYPE job with config: $CONFIG_PATH -> $CONFIG_NAME..."
    # 核心：动态启动命令，传入命令行参数
    python $LAUNCH_SCRIPT \
       --config_path $CONFIG_PATH \
       --config_name $CONFIG_NAME
    
    # todo: 检查megatron 训练完后自动转为huggingface，注意xbigdata库怎么装
else
    # 工作节点：连接主节点Ray集群，保持存活
    echo "Starting Ray worker node (connect to $MASTER_ADDR:$RAY_PORT)..."
    ray start --address="${MASTER_ADDR}":$RAY_PORT
    check_ray_ready
    sleep infinity  # 保持节点存活，提供计算资源
fi