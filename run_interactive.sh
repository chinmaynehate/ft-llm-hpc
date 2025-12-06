#!/bin/bash
# Run this AFTER getting an interactive allocation with salloc

# Set your Railway URL
export WS_BASE="${WS_BASE:-wss://ft-llm-relay-production.up.railway.app}"
export WS_ROOM="${WS_ROOM:-ft-llm}"
export NUM_GPUS="${NUM_GPUS:-4}"
export NODE_ID=$(hostname)

echo "========================================"
echo "FT-LLM Demo - Interactive Mode"
echo "========================================"
echo "Node: $NODE_ID"
echo "GPUs: $NUM_GPUS"
echo "WebSocket: $WS_BASE"
echo "========================================"

python hpc_main.py --num-gpus $NUM_GPUS --ws-base "$WS_BASE" --room "$WS_ROOM"
