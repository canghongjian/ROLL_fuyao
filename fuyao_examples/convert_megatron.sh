#!/bin/bash

## 转换megatron格式为huggingface
pip install ./mcore_adapter

CHECKPOINT_PATH="/path/to/your/megatron_checkpoint"
OUTPUT_PATH="/path/to/output/hf_model"

while [[ $# -gt 0 ]]; do
    case $1 in
        --ckpt-path)
            CHECKPOINT_PATH="$2"
            shift 2
            ;;
        --output-path)
            OUTPUT_PATH="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

echo "开始转权重！"
echo "你的megatron路径为：$CHECKPOINT_PATH"
echo "要转移到的目标路径为：$OUTPUT_PATH"

python mcore_adapter/tools/convert.py \
    --checkpoint_path $CHECKPOINT_PATH \
    --output_path $OUTPUT_PATH \