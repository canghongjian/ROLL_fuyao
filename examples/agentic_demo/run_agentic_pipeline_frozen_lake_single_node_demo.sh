#!/bin/bash
set +x

pip install ./mcore_adapter
ROLL_PATH="/workspace/ROLL-main"
CONFIG_PATH=$(basename $(dirname $0))
export PYTHONPATH="$ROLL_PATH:$PYTHONPATH"
python examples/start_agentic_pipeline.py --config_path $CONFIG_PATH  --config_name agent_val_frozen_lake_single_node_demo
