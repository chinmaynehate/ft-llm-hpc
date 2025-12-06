#!/bin/bash
# ============================================
# FT-LLM Demo - Interactive Run Script
# ============================================

# Configuration
export WS_BASE="${WS_BASE:-wss://ft-llm-relay-production.up.railway.app}"
export WS_ROOM="${WS_ROOM:-ft-llm}"
export NUM_GPUS="${NUM_GPUS:-4}"
export NODE_ID=$(hostname)

# Model settings
export LLAMA_DEVICE="${LLAMA_DEVICE:-cuda:0}"

# ============================================
# OUTPUT LENGTH FOR DEMO
# ============================================
# 256  = ~10-15 seconds (too short)
# 512  = ~20-30 seconds (good for quick demo)
# 1024 = ~40-60 seconds (good for full demo)
# 2048 = ~90-120 seconds (very long, for extended demo)
export LLAMA_MAX_TOKENS="${LLAMA_MAX_TOKENS:-1024}"

echo "========================================"
echo "FT-LLM Demo - Phase 2 (Real LLaMA)"
echo "========================================"
echo "Node:           $NODE_ID"
echo "GPUs:           $NUM_GPUS"
echo "WebSocket:      $WS_BASE"
echo "LLaMA Device:   $LLAMA_DEVICE"
echo "Max Tokens:     $LLAMA_MAX_TOKENS (longer = more demo time)"
echo "========================================"

# Verify GPU
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Check:"
    nvidia-smi --query-gpu=index,name,memory.total --format=csv
    echo "========================================"
fi

# Run
python hpc_main.py \
    --num-gpus $NUM_GPUS \
    --ws-base "$WS_BASE" \
    --room "$WS_ROOM" \
    --llama-device "$LLAMA_DEVICE" \
    --llama-max-new-tokens "$LLAMA_MAX_TOKENS"
